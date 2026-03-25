"""
Qwen2.5-VL-3B-Instruct — QLoRA Fine-tuning (명함 OCR)
======================================================
변경 사항 (LoRA → QLoRA):
  1. 베이스 모델을 NF4 4-bit로 로드 (BitsAndBytesConfig)
  2. prepare_model_for_kbit_training() 필수 적용
  3. LoRA 어댑터는 bfloat16으로 연산 (compute_dtype)
  4. ViT + LLM 디코더 동시 LoRA 적용 (글자 인식률 직접 향상)
  5. gradient_checkpointing 활성화 (VRAM 추가 절약)

설치:
    pip install git+https://github.com/huggingface/transformers accelerate
    pip install peft "bitsandbytes>=0.43.0" qwen-vl-utils[decord]==0.0.8
    pip install torch torchvision torchaudio  # CUDA 버전에 맞게

실행:
    python qwen25vl_qlora_finetune.py

폴더 구조:
    business_card_dataset/
        images/   card_00000.jpg ...
        labels/   labels.json
"""

# ─────────────────────────────────────────────────────────────
# 0. 임포트
# ─────────────────────────────────────────────────────────────
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

# CUDA 메모리 단편화 완화 (PyTorch 권장 옵션)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm


# ─────────────────────────────────────────────────────────────
# 1. 설정
# ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    # ── 모델 ──
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

    # ── 데이터 ──
    data_root:  str   = "/Users/jisu/Documents/coding_test/LLaMA-Factory/data/business_card_dataset"
    label_file: str   = "/Users/jisu/Documents/coding_test/LLaMA-Factory/data/business_card_dataset/labels/labels_train.json"
    train_ratio: float = 0.8
    val_ratio:   float = 0.1
    test_ratio:  float = 0.1

    # ── QLoRA 핵심 ──
    # NF4: 가중치를 정규분포 기반 4-bit로 압축 → 일반 int4보다 정확도 높음
    bnb_4bit_quant_type:     str  = "nf4"
    # 실제 행렬 연산은 bfloat16으로 dequantize해서 수행 (정밀도 유지)
    bnb_4bit_compute_dtype:  str  = "bfloat16"
    # double quantization: scale factor도 한 번 더 압축 → VRAM 추가 절약
    bnb_4bit_use_double_quant: bool = True

    # ── LoRA (QLoRA의 어댑터 부분) ──
    lora_r:       int   = 8
    lora_alpha:   int   = 32      # 보통 r * 2
    lora_dropout: float = 0.05

    # ViT(비전 인코더) + LLM 디코더 동시 학습
    # ViT 레이어: attn.qkv / attn.proj / mlp.fc1 / mlp.fc2
    # LLM 레이어: q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj
    lora_target_modules: list = field(default_factory=lambda: [ 
      # LLM
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj",

      # Vision encoder (substring match로 안정화)
      "qkv",
      "proj",
      "fc1",
      "fc2",
  ])

    # ── 학습 ──
    output_dir:                    str   = "checkpoints/qwen25vl_qlora"
    num_epochs:                    int   = 6
    per_device_train_batch_size:   int   = 1
    per_device_eval_batch_size:    int   = 1
    # 유효 배치 = 2 * 8 = 16  (QLoRA는 배치를 작게 가져가는 대신 accumulation으로 보완)
    gradient_accumulation_steps:   int   = 8
    learning_rate:                 float = 5e-5
    warmup_ratio:                  float = 0.1
    lr_scheduler_type:             str   = "cosine"
    # QLoRA는 bf16 연산 권장 (fp16은 4-bit와 궁합이 좋지 않음)
    bf16:                          bool  = True
    fp16:                          bool  = False
    max_grad_norm:                 float = 1.0
    logging_steps:                 int   = 10
    save_steps:                    int   = 100
    eval_steps:                    int   = 100
    seed:                          int   = 42

    # ── 이미지 토큰 범위 ──
    min_pixels: int = 256  * 28 * 28
    max_pixels: int = 1024 * 28 * 28

    # ── 메모리 안전 옵션 ──
    eval_every_n_epochs: int = 2
    inference_max_new_tokens: int = 128


CFG = Config()

# compute_dtype 문자열 → torch dtype 변환
COMPUTE_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


# ─────────────────────────────────────────────────────────────
# 2. 프롬프트 템플릿
# ─────────────────────────────────────────────────────────────

# ── System Prompt ──
# 모델의 역할과 출력 규칙을 간결하게 정의
SYSTEM_PROMPT = (
    "You are an OCR system specialized in reading business cards. "
    "Extract structured contact information from the card."
)

USER_PROMPT = (
    "Read the business card image and extract the information.\n\n"
    "Return the result as JSON:\n\n"
    "{\n"
    '  "name": "",\n'
    '  "email": "",\n'
    '  "company_phone": "",\n'
    '  "mobile_phone": ""\n'
    "}\n\n"
    "Rules:\n"
    "- Use mobile_phone for numbers starting with 010 or +82 10.\n"
    "- Use company_phone for regional area codes like 02, 031, 032, etc.\n"
    "- If a field is missing, return an empty string.\n"
    "- Output JSON only."
)

def build_target_json(label: dict) -> str:
    """
    라벨 dict → 모델이 출력해야 할 JSON 문자열
    출력 포맷은 USER_PROMPT의 [Output Format] 규칙과 일치:
      - 명함이면: is_business_card=true + name/email/company_phone/mobile_phone
      - 명함 아니면: is_business_card=false + confidence
    """
    # SFT 스타일 데이터셋(labels_train.json):
    # messages[].assistant.content 에 이미 구조화된 정답이 들어있음
    assistant_content = None
    messages = label.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and isinstance(msg.get("content"), dict)
            ):
                assistant_content = msg["content"]
                break

    if assistant_content is not None:
        return json.dumps(
            {
                "is_business_card": bool(assistant_content.get("is_business_card", True)),
                "name": str(assistant_content.get("name", "") or ""),
                "email": str(assistant_content.get("email", "") or ""),
                "company_phone": str(assistant_content.get("company_phone", "") or ""),
                "mobile_phone": str(assistant_content.get("mobile_phone", "") or ""),
            },
            ensure_ascii=False,
        )

    # 일반 스키마(labels.json): top-level company/mobile phone이 있으면 그대로 사용
    top_company_phone = str(label.get("company_phone", "") or "")
    top_mobile_phone = str(label.get("mobile_phone", "") or "")
    if top_company_phone or top_mobile_phone:
        return json.dumps(
            {
                "is_business_card": True,
                "name": str(label.get("name", "") or ""),
                "email": str(label.get("email", "") or ""),
                "company_phone": top_company_phone,
                "mobile_phone": top_mobile_phone,
            },
            ensure_ascii=False,
        )

    # 레거시 스키마: phone 단일 필드 분류
    raw_phone = str(label.get("phone", "") or "")
    mobile_phone = ""
    company_phone = ""

    phone_label_lower = raw_phone.lower()
    if any(tok in phone_label_lower for tok in ["m.", "mobile", "hp", "h.p", "cell", "휴대폰"]):
        mobile_phone = raw_phone
    elif any(tok in phone_label_lower for tok in ["tel", "office", "대표번호", "전화"]):
        company_phone = raw_phone
    else:
        digits = re.sub(r"\D", "", raw_phone)
        if digits.startswith(("8210", "010", "011", "016", "017", "019")):
            mobile_phone = raw_phone
        else:
            company_phone = raw_phone

    return json.dumps(
        {
            "is_business_card": True,
            "name":          label.get("name", ""),
            "email":         label.get("email", ""),
            "company_phone": company_phone,
            "mobile_phone":  mobile_phone,
        },
        ensure_ascii=False,
    )


def get_record_file(record: dict) -> str:
    """
    레코드에서 이미지 파일명을 꺼내 표준 키("file")로 정규화.
    서로 다른 데이터셋 키명(file, filename, image_path 등)을 허용한다.
    """
    candidate_keys = (
        "file",
        "filename",
        "file_name",
        "image_file",
        "image_filename",
        "image_path",
        "path",
        "image",
        "img",
    )

    file_value = None
    for key in candidate_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            file_value = value.strip()
            break
        if isinstance(value, dict):
            nested = value.get("path") or value.get("filename") or value.get("file")
            if isinstance(nested, str) and nested.strip():
                file_value = nested.strip()
                break
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first.strip():
                file_value = first.strip()
                break
            if isinstance(first, dict):
                nested = (
                    first.get("path")
                    or first.get("filename")
                    or first.get("file")
                    or first.get("image")
                )
                if isinstance(nested, str) and nested.strip():
                    file_value = nested.strip()
                    break

    # SFT 스타일 데이터셋: {"images": [...], "messages": [...], ...}
    if file_value is None:
        images_value = record.get("images")
        if isinstance(images_value, list) and images_value:
            first = images_value[0]
            if isinstance(first, str) and first.strip():
                file_value = first.strip()
            elif isinstance(first, dict):
                nested = (
                    first.get("path")
                    or first.get("filename")
                    or first.get("file")
                    or first.get("image")
                )
                if isinstance(nested, str) and nested.strip():
                    file_value = nested.strip()

    if file_value is None:
        raise KeyError(
            f"이미지 파일 키를 찾을 수 없습니다. 사용 가능한 키: {list(record.keys())}"
        )

    # 'images/foo.jpg' 형태는 images_dir와 중복 결합되지 않도록 정리
    file_value = file_value.replace("\\", "/")
    if file_value.startswith("./"):
        file_value = file_value[2:]
    if file_value.startswith("images/"):
        file_value = file_value[len("images/"):]
    file_value = file_value.lstrip("/")

    record["file"] = file_value
    return file_value


def resolve_image_path(images_dir: Path, record: dict) -> str:
    """
    이미지 경로를 해석한다.
    1) 로컬(images_dir/file) 우선
    2) 없고 HF 레코드 메타가 있으면 해당 파일만 즉시 다운로드
    """
    file_rel = get_record_file(record)
    local_path = images_dir / file_rel
    if local_path.exists():
        return str(local_path)

    repo_id = record.get("_hf_repo_id")
    if isinstance(repo_id, str) and repo_id:
        from huggingface_hub import hf_hub_download

        tried = []
        for filename in (f"images/{file_rel}", file_rel):
            tried.append(filename)
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                )
                return downloaded
            except Exception:
                continue

        raise FileNotFoundError(
            f"HF 이미지 다운로드 실패: repo={repo_id}, file={file_rel}, tried={tried}"
        )

    raise FileNotFoundError(
        f"이미지 파일을 찾을 수 없습니다: {local_path}. "
        "로컬 데이터셋이면 images_dir/labels 매칭을 확인하고, "
        "HF 데이터셋이면 load_records_from_hf 경로를 사용하세요."
    )


# ─────────────────────────────────────────────────────────────
# 3. 데이터셋
#
# [로드 방법 A] HuggingFace Hub (권장)
#   records = load_records_from_hf("ilovelevi/business_card_dataset")
#
# [로드 방법 B] 로컬 폴더
#   with open("business_card_dataset/labels/labels.json") as f:
#       records = json.load(f)
#
# labels.json 레코드 예시:
#   {
#     "file": "card_00000.jpg",
#     "name": "김민준",
#     "email": "info@company.co.kr",
#     "phone": "T. 010-1234-5678",
#     "company_display": "삼성전자 주식회사",
#     ...
#   }
# ─────────────────────────────────────────────────────────────

def load_records_from_hf(repo_id: str = "ilovelevi/business_card_dataset") -> tuple[list[dict], str]:
    """
    HuggingFace Hub에서 labels.json + 이미지를 다운로드.
    반환: (records 리스트, 이미지 캐시 디렉토리 경로)
    """
    from huggingface_hub import hf_hub_download

    print(f"[HF] {repo_id} labels.json 다운로드 중...")
    label_path = hf_hub_download(
        repo_id=repo_id,
        filename="labels/labels.json",
        repo_type="dataset",
    )
    with open(label_path, encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        raise ValueError("labels.json이 비어 있습니다.")
    for rec in records:
        if isinstance(rec, dict):
            rec["_hf_repo_id"] = repo_id

    # 첫 번째 이미지를 다운로드해 캐시 디렉토리 위치 파악
    first_file = get_record_file(records[0])
    first_img = hf_hub_download(
        repo_id=repo_id,
        filename=f"images/{first_file}",
        repo_type="dataset",
    )
    # first_file에 하위 폴더가 있어도 images 루트를 정확히 맞춘다.
    cache_images_root = Path(first_img)
    for _ in Path(first_file).parts:
        cache_images_root = cache_images_root.parent
    cache_images_dir = str(cache_images_root)
    print(f"[HF] {len(records)}건 로드 완료 | 이미지 캐시: {cache_images_dir}")
    return records, cache_images_dir


class BusinessCardDataset(Dataset):
    """
    records : labels.json에서 읽어온 dict 리스트
    processor: Qwen2.5-VL AutoProcessor
    images_dir: 이미지 파일들이 있는 디렉토리 경로
                (HF 캐시 경로 또는 로컬 data_root/images 경로)
    """
    def __init__(self, records: list[dict], processor, images_dir: str):
        self.records    = records
        self.processor  = processor
        self.images_dir = Path(images_dir)

    def __len__(self):
        return len(self.records)

    def _build_item(self, rec: dict) -> dict:
        img_path = resolve_image_path(self.images_dir, rec)
        target   = build_target_json(rec)

        # ── 메시지 구성 ──
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text",  "text": USER_PROMPT},
            ]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=False,
        )

        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            raise ValueError(f"vision inputs 누락 | path={img_path}")

        # qwen-vl 입력 표준화:
        # pixel_values: [num_vision_tokens, dim]
        # image_grid_thw: [num_images, 3]
        pv = inputs["pixel_values"]
        if pv.dim() == 3 and pv.shape[0] == 1:
            pv = pv[0]
        if pv.dim() != 2:
            raise ValueError(f"unexpected pixel_values.shape={tuple(pv.shape)} | path={img_path}")

        grid = inputs["image_grid_thw"]
        if grid.dim() == 1:
            grid = grid.unsqueeze(0)
        if grid.dim() != 2 or grid.shape[-1] != 3:
            raise ValueError(f"unexpected image_grid_thw.shape={tuple(grid.shape)} | path={img_path}")

        # 비정상 이미지 샘플(vision token 0개 혹은 spatial merge 불가) 방어
        if (grid <= 0).any():
            raise ValueError(f"invalid image_grid_thw={grid.tolist()} | path={img_path}")
        if pv.numel() == 0 or pv.shape[0] == 0:
            raise ValueError(f"empty pixel_values | path={img_path}")
        # Qwen2.5-VL spatial_merge_unit=4 가정: 최소 4 token 필요
        if pv.shape[0] < 4:
            raise ValueError(f"too few vision tokens={pv.shape[0]} | path={img_path}")

        target_ids = self.processor.tokenizer(
            target + self.processor.tokenizer.eos_token,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]

        # ── labels 구성 ──
        # 입력 = [프롬프트 토큰 | 정답 토큰]
        # labels = [  -100 ...  | 정답 토큰]  ← 프롬프트는 loss 계산 제외
        input_ids   = inputs["input_ids"][0]
        labels_full = torch.cat([torch.full_like(input_ids, -100), target_ids])
        input_ids   = torch.cat([input_ids, target_ids])

        result = {
            "input_ids":      input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels":         labels_full,
        }
        result["pixel_values"] = pv
        result["image_grid_thw"] = grid
        return result

    def __getitem__(self, idx: int) -> dict:
        max_tries = min(8, len(self.records))
        for offset in range(max_tries):
            rec = self.records[(idx + offset) % len(self.records)]
            try:
                return self._build_item(rec)
            except Exception as e:
                if offset == max_tries - 1:
                    raise RuntimeError(f"유효한 샘플을 찾지 못했습니다. 마지막 오류: {e}") from e
                continue


# ─────────────────────────────────────────────────────────────
# 4. Collator — pixel_values 별도 처리
# ─────────────────────────────────────────────────────────────
def collate_fn(batch: list[dict]) -> dict:
    from torch.nn.utils.rnn import pad_sequence

    result = {
        "input_ids": pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=0
        ),
        "attention_mask": pad_sequence(
            [b["attention_mask"] for b in batch],
            batch_first=True,
            padding_value=0
        ),
        "labels": pad_sequence(
            [b["labels"] for b in batch],
            batch_first=True,
            padding_value=-100
        ),
    }

    # 🔹 Qwen2.5-VL에서는 vision tokens을 concat하는 것이 맞다
    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.cat(
            [b["pixel_values"] for b in batch],
            dim=0
        )

    if "image_grid_thw" in batch[0]:
        result["image_grid_thw"] = torch.cat(
            [b["image_grid_thw"] for b in batch],
            dim=0
        )

    return result


# ─────────────────────────────────────────────────────────────
# 5. QLoRA 모델 로드
#    LoRA와의 차이점:
#      - BitsAndBytesConfig로 베이스 모델 자체를 4-bit로 로드
#      - prepare_model_for_kbit_training() 필수 (gradient checkpointing + casting)
#      - LoRA 어댑터는 여전히 bfloat16 full precision으로 학습
# ─────────────────────────────────────────────────────────────
def load_model_with_qlora(cfg: Config):
    compute_dtype = COMPUTE_DTYPE_MAP[cfg.bnb_4bit_compute_dtype]

    # ── 1) 4-bit 양자화 설정 ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,          # "nf4"
        bnb_4bit_compute_dtype=compute_dtype,                  # bfloat16
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,  # True
    )

    print(f"[모델 로드] {cfg.model_id}  (4-bit NF4 QLoRA)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        # 4-bit 로드 시 device_map="auto" 필수
        device_map="auto",
        # 베이스 가중치 dtype은 compute_dtype과 일치시킴
        torch_dtype=compute_dtype,
    )

    # ── 2) kbit 학습 준비 ──
    #  - gradient checkpointing 활성화 (VRAM 절약)
    #  - 4-bit 레이어를 float32로 업캐스트해서 gradient 흐름 보장
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # ── 3) LoRA 어댑터 설정 (ViT + LLM 동시) ──
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        # 어댑터 가중치는 bfloat16 full precision으로 학습
        # (베이스 모델은 4-bit 고정, 어댑터만 업데이트)
    )

    model = get_peft_model(model, lora_config)
    # 학습 중 KV cache 비활성화로 메모리 절감
    model.config.use_cache = False

    # 학습 파라미터 요약 출력
    model.print_trainable_parameters()
    # 예상 출력:
    # trainable params: ~19M || all params: ~1.86B || trainable%: ~1.02%
    # (베이스 3B 모델이 4-bit로 압축되어 실제 메모리는 ~2GB 수준)

    return model


# ─────────────────────────────────────────────────────────────
# 6. Trainer 서브클래스
# ─────────────────────────────────────────────────────────────
class VLMTrainer(Trainer):
    def collate_fn(batch: list[dict]) -> dict:
      from torch.nn.utils.rnn import pad_sequence

      result = {
          "input_ids": pad_sequence(
              [b["input_ids"] for b in batch],
              batch_first=True,
              padding_value=0,
          ),
          "attention_mask": pad_sequence(
              [b["attention_mask"] for b in batch],
              batch_first=True,
              padding_value=0,
          ),
          "labels": pad_sequence(
              [b["labels"] for b in batch],
              batch_first=True,
              padding_value=-100,
          ),
      }

    # 🔥 Qwen2.5-VL에서는 vision tokens 길이가 이미지마다 다르기 때문에
    # torch.cat() 하면 오류가 발생할 수 있음 → list 그대로 전달
      result["pixel_values"] = [b["pixel_values"] for b in batch]
      result["image_grid_thw"] = [b["image_grid_thw"] for b in batch]

      return result


# ─────────────────────────────────────────────────────────────
# 7. 평가 메트릭
# ─────────────────────────────────────────────────────────────
FIELDS = ["company", "name", "title", "phone", "email", "address"]


def parse_json_safe(text: str) -> dict:
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except Exception:
        return {}


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def _char_error_rate(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    return _levenshtein_distance(pred, ref) / max(1, len(ref))


def compute_metrics(eval_preds):
    predictions, label_ids = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    pred_ids = predictions.argmax(-1) if predictions.ndim == 3 else predictions
    label_ids = label_ids.copy()

    cers = []
    for pred_row, label_row in zip(pred_ids, label_ids):
        label_mask = label_row != -100
        if not label_mask.any():
            continue

        pred_target = pred_row[label_mask]
        label_target = label_row[label_mask]

        pred_text = processor.tokenizer.decode(pred_target, skip_special_tokens=True)  # noqa
        label_text = processor.tokenizer.decode(label_target, skip_special_tokens=True)  # noqa
        cers.append(_char_error_rate(pred_text, label_text))

    mean_cer = float(sum(cers) / len(cers)) if cers else 1.0
    return {"cer": mean_cer}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


class EpochProgressCallback(TrainerCallback):

    def __init__(self):
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        total_epochs = int(math.ceil(float(args.num_train_epochs)))
        self.pbar = tqdm(total=total_epochs, desc="Training Epochs")

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch_idx = int(state.epoch) + 1 if state.epoch is not None else 1
        total_epochs = int(math.ceil(float(args.num_train_epochs)))
        print(f"\n[Epoch {epoch_idx}/{total_epochs}] 학습 진행 중...")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.update(1)

        epoch_idx = int(state.epoch) if state.epoch is not None else -1
        total_epochs = int(math.ceil(float(args.num_train_epochs)))
        print(f"[Epoch {epoch_idx}/{total_epochs}] 학습 완료")

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()


class CERPerEpochCallback(TrainerCallback):
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.trainer = None

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control

        epoch_idx = int(state.epoch) if state.epoch is not None else -1
        if epoch_idx % CFG.eval_every_n_epochs != 0:
            return control

        print(f"\n[CER] epoch {epoch_idx} (val only)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_metrics = self.trainer.evaluate(
            eval_dataset=self.val_dataset,
            metric_key_prefix="val",
        )

        print(f"[CER] val={val_metrics.get('val_cer', float('nan')):.6f}")
        return control


# ─────────────────────────────────────────────────────────────
# 8. 추론 헬퍼
# ─────────────────────────────────────────────────────────────
def inference(model, processor, image_path: str, device="cuda") -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": USER_PROMPT},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=CFG.inference_max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated   = output_ids[:, inputs["input_ids"].shape[1]:]
    result_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return parse_json_safe(result_text)


# ─────────────────────────────────────────────────────────────
# 9. 메인
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(CFG.seed)
    random.seed(CFG.seed)

    # ── 프로세서 ──
    print("[프로세서 로드]")
    processor = AutoProcessor.from_pretrained(
        CFG.model_id,
        min_pixels=CFG.min_pixels,
        max_pixels=CFG.max_pixels,
    )
    globals()["processor"] = processor

    # ── 데이터 로드 (로컬 데이터셋 기본) ──
    with open(CFG.label_file, encoding="utf-8") as f:
        all_records = json.load(f)
    images_dir = str(Path(CFG.data_root) / "images")
    print(f"[로컬 데이터셋] labels={CFG.label_file} | images={images_dir}")

    # HF 데이터셋을 쓰려면 아래 한 줄로 교체:
    # all_records, images_dir = load_records_from_hf("ilovelevi/business_card_dataset")

    if not math.isclose(CFG.train_ratio + CFG.val_ratio + CFG.test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio 는 1.0이어야 합니다.")

    rng = random.Random(CFG.seed)
    rng.shuffle(all_records)

    n_total = len(all_records)
    n_train = max(1, int(n_total * CFG.train_ratio))
    n_val   = max(1, int(n_total * CFG.val_ratio))
    n_test  = n_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
    if n_train + n_val + n_test != n_total:
        n_train = n_total - n_val - n_test

    trn_rec  = all_records[:n_train]
    val_rec  = all_records[n_train:n_train + n_val]
    test_rec = all_records[n_train + n_val:]
    print(f"[데이터] 학습 {len(trn_rec)}건 / 검증 {len(val_rec)}건 / 테스트 {len(test_rec)}건")

    train_dataset = BusinessCardDataset(trn_rec, processor, images_dir)
    val_dataset   = BusinessCardDataset(val_rec,  processor, images_dir)
    test_dataset  = BusinessCardDataset(test_rec, processor, images_dir)

    # ── QLoRA 모델 ──
    model = load_model_with_qlora(CFG)

    # ── TrainingArguments ──
    # QLoRA 주의: optim은 "paged_adamw_8bit" 권장
    #   → 옵티마이저 상태도 8-bit로 압축해 VRAM 추가 절약
    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.num_epochs,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        per_device_eval_batch_size=CFG.per_device_eval_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.learning_rate,
        warmup_ratio=CFG.warmup_ratio,
        lr_scheduler_type=CFG.lr_scheduler_type,
        bf16=CFG.bf16,
        fp16=CFG.fp16,
        max_grad_norm=CFG.max_grad_norm,
        optim="paged_adamw_8bit",      # ← QLoRA 핵심: 옵티마이저도 8-bit
        logging_steps=CFG.logging_steps,
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=3,
        eval_accumulation_steps=1,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=CFG.seed,
        report_to="none",
        disable_tqdm=True,
        # gradient_checkpointing은 prepare_model_for_kbit_training()에서 이미 활성화
        gradient_checkpointing=False,
    )

    epoch_progress_cb = EpochProgressCallback()
    cer_cb = CERPerEpochCallback(train_dataset=train_dataset, val_dataset=val_dataset)

    trainer = VLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[epoch_progress_cb, cer_cb],
    )
    cer_cb.trainer = trainer

    print("\n[학습 시작] QLoRA (ViT + LLM 동시)")
    trainer.train()

    # ── 어댑터 저장 ──
    lora_save_path = Path(CFG.output_dir) / "qlora_adapter"
    model.save_pretrained(lora_save_path)
    processor.save_pretrained(lora_save_path)
    # 필요하면 .pt 단일 파일 저장을 활성화하세요.
    # torch.save(model.state_dict(), Path(CFG.output_dir) / "model_state_dict.pt")
    print(f"\n[저장 완료] {lora_save_path}")

    print("\n[테스트 CER]")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(f"[CER] test={test_metrics.get('test_cer', float('nan')):.6f}")

    # ── 추론 테스트 ──
    print("\n[추론 테스트]")
    sample   = val_rec[0]
    img_path = resolve_image_path(Path(images_dir), sample)
    result   = inference(model, processor, img_path)
    print("예측:", json.dumps(result,            ensure_ascii=False, indent=2))
    print("정답:", build_target_json(sample))


# ─────────────────────────────────────────────────────────────
# 10. 배포용 병합
#     QLoRA는 4-bit 베이스에 병합이 불가 → bfloat16으로 재로드 후 병합
# ─────────────────────────────────────────────────────────────
def merge_and_save(qlora_adapter_path: str, save_path: str):
    """
    QLoRA 어댑터 병합 시 주의:
      4-bit 양자화 모델에는 직접 merge_and_unload() 불가
      → 베이스 모델을 bf16으로 새로 로드한 뒤 어댑터 병합

    사용법:
        merge_and_save("checkpoints/qwen25vl_qlora/qlora_adapter", "merged_model")
    """
    print("[병합] bf16 베이스 모델 로드 (양자화 없이)...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CFG.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        # 양자화 설정 없이 로드
    )
    print("[병합] QLoRA 어댑터 로드...")
    peft_model = PeftModel.from_pretrained(base, qlora_adapter_path)

    print("[병합] 어댑터 병합 중...")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(save_path)
    AutoProcessor.from_pretrained(qlora_adapter_path).save_pretrained(save_path)
    print(f"[병합 완료] {save_path}  (bf16 단일 모델)")
