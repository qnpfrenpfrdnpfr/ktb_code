# ╔══════════════════════════════════════════════════════════════════╗
# ║  Qwen2.5-VL-3B-Instruct — QLoRA Fine-tuning (Google Colab T4)  ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# T4 VRAM : 15GB
# 예상 사용량: ~11~13GB (4-bit + gradient checkpointing)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 : 패키지 설치
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [중요] RuntimeError: duplicate registrations for aten.linspace.Tensor_Tensor
# 원인: bitsandbytes 최신 버전과 torch 버전 충돌
# 해결: bitsandbytes를 0.41.3으로 고정 + 런타임 재시작 필수
#
# !pip install -q \
#     git+https://github.com/huggingface/transformers \
#     accelerate \
#     "peft>=0.10.0" \
#     "bitsandbytes==0.41.3" \
#     huggingface_hub \
#     qwen-vl-utils \
#     pillow
#
# ★ 설치 후 반드시: 런타임 → 런타임 다시 시작 → CELL 2부터 재실행


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 : Google Drive 마운트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# from google.colab import drive
# drive.mount('/content/drive')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 : 경로 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os
from pathlib import Path

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent
else:
    # Jupyter/Colab 셀 실행 환경에서는 __file__가 없으므로 현재 작업 디렉터리를 사용
    PROJECT_ROOT = Path.cwd()

# HF 데이터셋 기본값
# - 요청한 repo: https://huggingface.co/datasets/ilovelevi/data_3_10/tree/main/dataset_3_10
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "ilovelevi/data_3_10")
HF_DATASET_SUBDIR = os.environ.get("HF_DATASET_SUBDIR", "dataset_3_10")
HF_DATASET_CACHE = Path(
    os.environ.get("HF_DATASET_CACHE", str(PROJECT_ROOT / ".hf_datasets_cache"))
).expanduser()

# USE_HF_DATASET=1 이면 HF에서 snapshot 다운로드 후 사용
# USE_HF_DATASET=0 이면 로컬 DATASET_310_ROOT 사용
USE_HF_DATASET = os.environ.get("USE_HF_DATASET", "1").strip().lower() in {
    "1", "true", "yes", "y", "on"
}
LOCAL_DATA_ROOT = Path(
    os.environ.get("DATASET_310_ROOT", str(PROJECT_ROOT / "dataset_3_10"))
).expanduser()

# device_map auto 사용 여부 (Jupyter 기본값 OFF)
# - OFF(기본): accelerate 이슈 회피, 단일 GPU 환경에서 안전
# - ON: USE_DEVICE_MAP_AUTO=1
USE_DEVICE_MAP_AUTO = os.environ.get("USE_DEVICE_MAP_AUTO", "0").strip().lower() in {
    "1", "true", "yes", "y", "on"
}

# 체크포인트 저장 경로 (기본: 프로젝트 하위)
#   QLORA_OUTPUT_DIR=/path/to/output
OUTPUT_DIR = Path(
    os.environ.get("QLORA_OUTPUT_DIR", str(PROJECT_ROOT / "qwen25vl_qlora_output"))
).expanduser()


def resolve_dataset_root() -> Path:
    if not USE_HF_DATASET:
        return LOCAL_DATA_ROOT

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ImportError(
            "huggingface_hub가 필요합니다. `pip install huggingface_hub` 후 다시 실행하세요."
        ) from e

    print(f"[HF DATASET] snapshot 다운로드: {HF_DATASET_REPO}")
    try:
        snapshot_root = Path(
            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                cache_dir=str(HF_DATASET_CACHE),
            )
        )
    except Exception as e:
        if LOCAL_DATA_ROOT.exists():
            print(f"[HF DATASET] 다운로드 실패. 로컬 경로로 fallback: {LOCAL_DATA_ROOT}")
            print(f"[HF DATASET] 원인: {e}")
            return LOCAL_DATA_ROOT
        raise RuntimeError(
            f"HF 데이터셋 다운로드 실패: {HF_DATASET_REPO}. "
            f"네트워크/HF 토큰 권한을 확인하거나 DATASET_310_ROOT를 지정하세요."
        ) from e

    preferred_root = snapshot_root / HF_DATASET_SUBDIR
    if preferred_root.exists():
        return preferred_root
    return snapshot_root


def resolve_label_paths(data_root: Path) -> tuple[Path, Path]:
    json_candidates = [
        data_root / "labels" / "labels.json",
        data_root / "labels.json",
        data_root / "labels" / "labels_train.json",
        data_root / "labels_train.json",
    ]
    jsonl_candidates = [
        data_root / "labels" / "labels.jsonl",
        data_root / "labels.jsonl",
    ]

    label_file = next((p for p in json_candidates if p.exists()), json_candidates[0])
    label_jsonl = next((p for p in jsonl_candidates if p.exists()), jsonl_candidates[0])
    return label_file, label_jsonl


DATA_ROOT = resolve_dataset_root()
IMAGES_DIR = DATA_ROOT / "images"
LABEL_FILE, LABEL_JSONL_FILE = resolve_label_paths(DATA_ROOT)

# ── 경로 확인 ──
for p, name in [
    (Path(HF_DATASET_CACHE), "hf cache"),
    (DATA_ROOT, "dataset_3_10 root"),
    (IMAGES_DIR, "images"),
    (LABEL_FILE, "labels.json"),
    (LABEL_JSONL_FILE, "labels.jsonl"),
]:
    status = "OK" if p.exists() else "NOT FOUND"
    print(f"[{status}] {name}: {p}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 : 임포트 & 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import json, re, random
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from qwen_vl_utils import process_vision_info

# ── accelerate 유무 점검 ──
try:
    import accelerate as _accelerate_check  # noqa: F401
    _ACCELERATE_IMPORT_OK = True
except Exception:
    _ACCELERATE_IMPORT_OK = False

try:
    from transformers.utils import is_accelerate_available as _tf_is_accelerate_available
    _ACCELERATE_TF_OK = bool(_tf_is_accelerate_available())
except Exception:
    _ACCELERATE_TF_OK = False

ACCELERATE_AVAILABLE = bool(_ACCELERATE_IMPORT_OK and _ACCELERATE_TF_OK)
if not ACCELERATE_AVAILABLE:
    print(
        "[경고] accelerate 감지 실패(또는 커널 재시작 필요): "
        "device_map='auto'를 비활성화합니다."
    )
print(
    f"[accelerate] import_ok={_ACCELERATE_IMPORT_OK}, "
    f"transformers_ok={_ACCELERATE_TF_OK}, use_device_map_auto={USE_DEVICE_MAP_AUTO}"
)

# ── 버전 확인 (linspace 충돌 사전 점검) ──
import torch as _torch_check
import bitsandbytes as _bnb_check
_torch_ver  = tuple(int(x) for x in _torch_check.__version__.split(".")[:2])
_bnb_ver    = _bnb_check.__version__

print(f"[torch]         {_torch_check.__version__}")
print(f"[bitsandbytes]  {_bnb_ver}")

if _torch_ver >= (2, 2) and _bnb_ver >= "0.42":
    print()
    print("=" * 60)
    print("[경고] torch>=2.2 + bitsandbytes>=0.42 조합에서")
    print("  'duplicate registrations for aten.linspace' 에러 발생 가능")
    print("  → CELL 1로 돌아가 bitsandbytes==0.41.3 으로 재설치 후")
    print("    런타임을 다시 시작하세요.")
    print("=" * 60)
else:
    print("[OK] 버전 조합 이상 없음")

print(f"[GPU]  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU — 런타임 확인!'}")
if torch.cuda.is_available():
    print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[VRAM] N/A")


@dataclass
class Config:
    # ── 모델 ──
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # ── QLoRA (4-bit NF4) ──
    bnb_4bit_quant_type:       str  = "nf4"
    bnb_4bit_compute_dtype:    str  = "float16"   # T4는 bfloat16 미지원 → float16
    bnb_4bit_use_double_quant: bool = True

    # ── LoRA 어댑터 ──
    lora_r:       int   = 8
    lora_alpha:   int   = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        # ── ViT (비전 인코더) ── 글자 인식률 향상
        "attn.qkv",
        "attn.proj",
        "mlp.fc1",
        "mlp.fc2",
        # ── LLM 디코더 ──
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── 학습 (T4 최적화) ──
    output_dir:                  str   = str(OUTPUT_DIR)
    num_epochs:                  int   = 3
    per_device_train_batch_size: int   = 1     # T4 한계상 1
    per_device_eval_batch_size:  int   = 1
    gradient_accumulation_steps: int   = 16   # 유효 배치 = 1*16 = 16
    learning_rate:               float = 2e-4
    warmup_ratio:                float = 0.05
    lr_scheduler_type:           str   = "cosine"
    fp16:                        bool  = True  # T4는 fp16
    bf16:                        bool  = False
    max_grad_norm:               float = 1.0
    logging_steps:               int   = 10
    save_steps:                  int   = 50
    eval_steps:                  int   = 50
    val_ratio:                   float = 0.1
    seed:                        int   = 42

    # ── 이미지 토큰 범위 (T4 메모리 절약을 위해 축소) ──
    min_pixels: int = 256  * 28 * 28
    max_pixels: int = 512  * 28 * 28   # 1280 → 512 (VRAM 절약)


CFG = Config()
DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 : 프롬프트 & 정답 JSON 빌더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_TEXT = "You are a business card OCR assistant. Output JSON only."

USER_TEXT = (
    "Extract information from this business card.\n"
    "Output a JSON object with these fields: "
    "is_business_card, name, email, company, company_phone, mobile_phone, job_title, department.\n"
    "If not a business card, output: {\"is_business_card\": false}"
)


def clean_text(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def pick_lang_value(rec: dict, ko_key: str, en_key: str, fallback_key: str = "") -> str:
    """
    dataset_3_10의 *_ko/*_en 필드에서 언어 타입에 맞는 값을 선택.
    mixed면 둘 다 존재할 때 "KO | EN" 형태로 유지.
    """
    ko = clean_text(rec.get(ko_key, ""))
    en = clean_text(rec.get(en_key, ""))
    fb = clean_text(rec.get(fallback_key, "")) if fallback_key else ""
    lang = clean_text(rec.get("lang_type", "")).lower()

    if lang == "english_only":
        return en or ko or fb
    if lang == "korean_only":
        return ko or en or fb
    if ko and en and ko != en:
        return f"{ko} | {en}"
    return ko or en or fb


def split_phone_from_string(raw: str) -> tuple[str, str]:
    """단일 phone 문자열 fallback 분리 (구버전 labels 호환)."""
    s = clean_text(raw)
    lower = s.lower()
    if any(t in lower for t in ["m.", "mobile", "hp", "h.p", "cell", "휴대폰"]):
        return "", s
    if any(t in lower for t in ["tel", "office", "대표번호", "전화"]):
        return s, ""
    digits = re.sub(r"\D", "", s)
    if digits.startswith(("8210", "010", "011", "016", "017", "018", "019")):
        return "", s
    return s, ""


def split_phones(rec: dict) -> tuple[str, str]:
    """
    phones(list)에서 T/M을 분리.
    - label이 M/mobile/hp 계열이거나 mobile 번호 패턴이면 mobile_phone
    - 나머지는 company_phone
    """
    company_phone = clean_text(rec.get("company_phone", ""))
    mobile_phone = clean_text(rec.get("mobile_phone", ""))

    phones = rec.get("phones")
    if isinstance(phones, list):
        for phone in phones:
            if not isinstance(phone, dict):
                continue
            label = clean_text(phone.get("label", "")).lower().replace(".", "")
            value = clean_text(phone.get("value", ""))
            if not value:
                continue

            digits = re.sub(r"\D", "", value)
            is_mobile = (
                label in {"m", "mobile", "hp", "h p", "cell", "handphone"}
                or digits.startswith(("8210", "010", "011", "016", "017", "018", "019"))
            )
            if is_mobile:
                if not mobile_phone:
                    mobile_phone = value
            else:
                if not company_phone:
                    company_phone = value

            if company_phone and mobile_phone:
                break

    if not company_phone and not mobile_phone:
        return split_phone_from_string(rec.get("phone", ""))
    return company_phone, mobile_phone


def resolve_image_path(rec: dict, images_dir: Path, data_root: Path) -> Path:
    """
    dataset_3_10(image) / 기존 labels(images,file) 모두 지원.
    """
    candidates = []
    if clean_text(rec.get("image", "")):
        candidates.append(clean_text(rec["image"]))
    if isinstance(rec.get("images"), list) and rec["images"]:
        candidates.append(clean_text(rec["images"][0]))
    if clean_text(rec.get("file", "")):
        candidates.append(clean_text(rec["file"]))

    for raw in candidates:
        p = Path(raw)
        if p.is_absolute() and p.exists():
            return p

        path_candidates = [
            data_root / p,
            images_dir / p,
            images_dir / p.name,
        ]
        for cp in path_candidates:
            if cp.exists():
                return cp

    raise FileNotFoundError(
        f"이미지를 찾지 못했습니다. candidates={candidates}, images_dir={images_dir}, data_root={data_root}"
    )


def load_records(label_json_path: Path, label_jsonl_path: Path) -> tuple[list, Path]:
    if label_json_path.exists():
        with open(label_json_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON 라벨 파일은 list여야 합니다: {label_json_path}")
        return data, label_json_path

    if label_jsonl_path.exists():
        rows = []
        with open(label_jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows, label_jsonl_path

    raise FileNotFoundError(
        f"라벨 파일을 찾지 못했습니다: {label_json_path} 또는 {label_jsonl_path}"
    )


def build_target_json(rec: dict) -> str:
    """labels.json 레코드 → 모델이 출력해야 할 정답 JSON 문자열
    USER_TEXT 필드와 동일:
    is_business_card, name, email, company, company_phone, mobile_phone, job_title, department
    """
    company_phone, mobile_phone = split_phones(rec)
    name = pick_lang_value(rec, "name_ko", "name_en", "name")
    company = pick_lang_value(rec, "company_ko", "company_en", "company")
    job_title = pick_lang_value(rec, "job_title_ko", "job_title_en", "job_title")
    department = pick_lang_value(rec, "department_ko", "department_en", "department")
    return json.dumps(
        {
            "is_business_card": True,
            "name": name,
            "email": clean_text(rec.get("email", "")),
            "company": company,
            "company_phone": company_phone,
            "mobile_phone": mobile_phone,
            "job_title": job_title,
            "department": department,
        },
        ensure_ascii=False,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 : 데이터셋
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BusinessCardDataset(Dataset):
    def __init__(self, records: list, processor, images_dir: Path):
        self.records    = records
        self.processor  = processor
        self.images_dir = images_dir

    def __len__(self):
        return len(self.records)

    def _build_item(self, rec: dict) -> dict:
        img_path = str(resolve_image_path(rec, self.images_dir, DATA_ROOT))

        user_prompt = USER_TEXT
        if isinstance(rec.get("messages"), list) and len(rec["messages"]) > 0:
            first_msg = rec["messages"][0]
            if isinstance(first_msg, dict):
                content = first_msg.get("content")
                if isinstance(content, str) and content.strip():
                    user_prompt = content

        if (
            isinstance(rec.get("messages"), list)
            and len(rec["messages"]) > 1
            and isinstance(rec["messages"][1], dict)
            and "content" in rec["messages"][1]
        ):
            target = json.dumps(rec["messages"][1]["content"], ensure_ascii=False)
        else:
            target = build_target_json(rec)

        messages = [
          {
              "role": "system",
              "content": [{"type":"text","text":SYSTEM_TEXT}]
          },
          {
              "role": "user",
              "content": [
                  {"type":"image","image":img_path},
                  {"type":"text","text":user_prompt}
              ]
          }
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

        # Qwen2.5-VL 입력 표준화
        # pixel_values: [num_vision_tokens, dim]
        # image_grid_thw: [num_images, 3]
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 3 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values[0]
        if pixel_values.dim() != 2:
            raise ValueError(
                f"unexpected pixel_values.shape={tuple(pixel_values.shape)} | path={img_path}"
            )
        if pixel_values.numel() == 0 or pixel_values.shape[0] < 4:
            raise ValueError(
                f"invalid pixel_values.shape={tuple(pixel_values.shape)} | path={img_path}"
            )

        image_grid_thw = inputs["image_grid_thw"]
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        if image_grid_thw.dim() != 2 or image_grid_thw.shape[-1] != 3:
            raise ValueError(
                f"unexpected image_grid_thw.shape={tuple(image_grid_thw.shape)} | path={img_path}"
            )
        if (image_grid_thw <= 0).any():
            raise ValueError(f"invalid image_grid_thw={image_grid_thw.tolist()} | path={img_path}")

        # 정답 토크나이즈
        target_ids = self.processor.tokenizer(
            target + self.processor.tokenizer.eos_token,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]

        # 입력 = [프롬프트 | 정답]
        # labels = [  -100  | 정답]  ← 프롬프트는 loss 제외
        input_ids   = inputs["input_ids"][0]
        labels_ids  = torch.cat([torch.full_like(input_ids, -100), target_ids])
        input_ids   = torch.cat([input_ids, target_ids])

        result = {
            "input_ids":      input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels":         labels_ids,
        }
        result["pixel_values"] = pixel_values
        result["image_grid_thw"] = image_grid_thw

        return result

    def __getitem__(self, idx: int):
        max_tries = min(8, len(self.records))
        for offset in range(max_tries):
            rec = self.records[(idx + offset) % len(self.records)]
            try:
                return self._build_item(rec)
            except Exception as e:
                if offset == max_tries - 1:
                    raise RuntimeError(
                        f"유효한 샘플을 찾지 못했습니다. 마지막 오류: {e}"
                    ) from e
                continue


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7 : Collator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collate_fn(batch: list) -> dict:
    result = {
        "input_ids":      pad_sequence([b["input_ids"]      for b in batch],
                                        batch_first=True, padding_value=0),
        "attention_mask": pad_sequence([b["attention_mask"]  for b in batch],
                                        batch_first=True, padding_value=0),
        "labels":         pad_sequence([b["labels"]          for b in batch],
                                        batch_first=True, padding_value=-100),
    }
    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    if "image_grid_thw" in batch[0]:
        result["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8 : QLoRA 모델 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_model_with_qlora(cfg: Config):
    compute_dtype = DTYPE_MAP[cfg.bnb_4bit_compute_dtype]

    # 1) 4-bit NF4 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,           # nf4
        bnb_4bit_compute_dtype=compute_dtype,                  # float16 (T4)
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )

    print(f"[모델 로드] {cfg.model_id}  (4-bit NF4)")
    load_kwargs = {
        "quantization_config": bnb_config,
        "torch_dtype": compute_dtype,
    }
    if USE_DEVICE_MAP_AUTO and ACCELERATE_AVAILABLE:
        load_kwargs["device_map"] = "auto"
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_id,
            **load_kwargs,
        )
    except ValueError as e:
        # Jupyter에서 accelerate 설치/재시작 타이밍 꼬임으로 발생하는 케이스 안전 fallback
        if "requires `accelerate`" in str(e) and "device_map" in load_kwargs:
            print("[경고] device_map='auto' 재시도 실패 -> device_map 없이 다시 로드합니다.")
            load_kwargs.pop("device_map", None)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.model_id,
                **load_kwargs,
            )
        else:
            raise

    # 2) kbit 학습 준비 (gradient checkpointing 포함)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    # 3) LoRA 어댑터 — ViT + LLM 동시
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,           # 8
        lora_alpha=cfg.lora_alpha,   # 32
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_with_saved_adapter(cfg: Config, adapter_dir: Path):
    compute_dtype = DTYPE_MAP[cfg.bnb_4bit_compute_dtype]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )

    print(f"[어댑터 재로딩] base={cfg.model_id} | adapter={adapter_dir}")
    load_kwargs = {
        "quantization_config": bnb_config,
        "torch_dtype": compute_dtype,
    }
    if USE_DEVICE_MAP_AUTO and ACCELERATE_AVAILABLE:
        load_kwargs["device_map"] = "auto"
    try:
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_id,
            **load_kwargs,
        )
    except ValueError as e:
        if "requires `accelerate`" in str(e) and "device_map" in load_kwargs:
            print("[경고] 재로딩 단계에서 device_map 오류 -> device_map 없이 다시 로드합니다.")
            load_kwargs.pop("device_map", None)
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.model_id,
                **load_kwargs,
            )
        else:
            raise
    reloaded_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    reloaded_model.eval()
    return reloaded_model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9 : Trainer 서브클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 10 : 메인 학습 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

torch.manual_seed(CFG.seed)
random.seed(CFG.seed)

# ── 프로세서 로드 ──
print("[프로세서 로드]")
processor = AutoProcessor.from_pretrained(
    CFG.model_id,
    min_pixels=CFG.min_pixels,
    max_pixels=CFG.max_pixels,
)

# ── 라벨 로드 & train/val 분할 ──
all_records, active_label_path = load_records(LABEL_FILE, LABEL_JSONL_FILE)
print(f"[라벨 로드] {active_label_path}")

random.shuffle(all_records)
n_val   = max(1, int(len(all_records) * CFG.val_ratio))
val_rec = all_records[:n_val]
trn_rec = all_records[n_val:]
print(f"[데이터] 전체 {len(all_records)}건 | 학습 {len(trn_rec)}건 | 검증 {n_val}건")

# ── 데이터셋 ──
train_dataset = BusinessCardDataset(trn_rec, processor, IMAGES_DIR)
val_dataset   = BusinessCardDataset(val_rec,  processor, IMAGES_DIR)

# ── QLoRA 모델 ──
model = load_model_with_qlora(CFG)

# ── TrainingArguments (T4 최적화) ──
training_args = TrainingArguments(
    output_dir=CFG.output_dir,
    num_train_epochs=CFG.num_epochs,
    per_device_train_batch_size=CFG.per_device_train_batch_size,   # 1
    per_device_eval_batch_size=CFG.per_device_eval_batch_size,     # 1
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,  # 16
    learning_rate=CFG.learning_rate,
    warmup_ratio=CFG.warmup_ratio,
    lr_scheduler_type=CFG.lr_scheduler_type,
    fp16=CFG.fp16,       # T4 → fp16
    bf16=CFG.bf16,       # False
    max_grad_norm=CFG.max_grad_norm,
    optim="paged_adamw_8bit",          # 옵티마이저도 8-bit 압축
    logging_steps=CFG.logging_steps,
    save_steps=CFG.save_steps,
    eval_steps=CFG.eval_steps,
    eval_strategy="steps",
    save_total_limit=2,                # 체크포인트 2개만 유지 (드라이브 절약)
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    gradient_checkpointing=False,      # prepare_model_for_kbit_training에서 이미 설정
    seed=CFG.seed,
    report_to="none",
)

trainer = VLMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# ── 학습 시작 ──
print("\n[학습 시작] QLoRA — ViT + LLM 동시 학습")
print(f"  배치 크기 : {CFG.per_device_train_batch_size} × accumulation {CFG.gradient_accumulation_steps} = 유효 {CFG.per_device_train_batch_size * CFG.gradient_accumulation_steps}\n")

trainer.train()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 11 : LoRA 어댑터 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

adapter_path = OUTPUT_DIR / "qlora_adapter"
model.save_pretrained(adapter_path)
processor.save_pretrained(adapter_path)
print(f"\n[저장 완료] {adapter_path}")

# ── 저장된 어댑터를 다시 로드해 베이스 모델에 결합 (실전 테스트용) ──
reloaded_processor = AutoProcessor.from_pretrained(
    adapter_path,
    min_pixels=CFG.min_pixels,
    max_pixels=CFG.max_pixels,
)
reloaded_model = load_model_with_saved_adapter(CFG, adapter_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 12 : 학습 후 추론 테스트 (선택)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def inference(model, processor, img_path: str, device="cuda") -> dict:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_TEXT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text",  "text": USER_TEXT},
            ],
        },
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
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated  = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_text   = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    clean      = re.sub(r"```(?:json)?|```", "", raw_text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_output": raw_text}


# 검증 셋 첫 번째 샘플로 테스트
sample   = val_rec[0]
sample_path = resolve_image_path(sample, IMAGES_DIR, DATA_ROOT)
img_path = str(sample_path)
result_in_memory = inference(model, processor, img_path)
result_reloaded  = inference(reloaded_model, reloaded_processor, img_path)

print("\n[추론 테스트]")
print(f"  파일  : {sample_path.name}")
print(f"  예측(in-memory) : {json.dumps(result_in_memory, ensure_ascii=False, indent=2)}")
print(f"  예측(reloaded)  : {json.dumps(result_reloaded, ensure_ascii=False, indent=2)}")
print(f"  정답  : {build_target_json(sample)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 13 : (선택) 어댑터 → 베이스 모델 병합 저장 (배포용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QLoRA는 4-bit 모델에 직접 merge 불가
# → 베이스 모델을 float16으로 새로 로드한 뒤 어댑터 병합
#
# merge_path = OUTPUT_DIR / "merged_model"
# base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     CFG.model_id, torch_dtype=torch.float16, device_map="cpu"
# )
# merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
# merged.save_pretrained(merge_path)
# processor.save_pretrained(merge_path)
# print(f"[병합 완료] {merge_path}")
