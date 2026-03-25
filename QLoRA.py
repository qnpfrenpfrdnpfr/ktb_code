import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch

from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from qwen_vl_utils import process_vision_info


# =========================================================
# 0. 설정
# =========================================================
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Hugging Face dataset repo
HF_DATASET_REPO = "ilovelevi/data_3_11"
HF_DATASET_SUBDIR = "dataset_3_10"

# 내려받을 로컬 폴더
LOCAL_DATA_DIR = "./hf_data_3_11"

OUTPUT_DIR = "./outputs/qwen3_vl_8b_qlora_ocr"
SEED = 42
TRAIN_RATIO = 0.8

MAX_LENGTH = 2048
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
SAVE_STEPS = 200
EVAL_STEPS = 200
LOGGING_STEPS = 10

IMAGE_PATCH_SIZE = 16
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

SYSTEM_PROMPT = (
    "너는 명함 OCR 및 정보 추출 전문가다. "
    "이미지를 보고 이 이미지가 명함인지 판단하고, "
    "명함이라면 반드시 아래 JSON 스키마로만 답변하라. "
    "설명 문장 없이 JSON만 출력하라.\n\n"
    "스키마:\n"
    "{\n"
    '  "is_business_card": true,\n'
    '  "name": "",\n'
    '  "company": "",\n'
    '  "job_title": "",\n'
    '  "department": "",\n'
    '  "email": "",\n'
    '  "company_phone": "",\n'
    '  "mobile_phone": ""\n'
    "}\n\n"
    "명함이 아니면 다음처럼 출력하라:\n"
    "{\n"
    '  "is_business_card": false,\n'
    '  "name": "",\n'
    '  "company": "",\n'
    '  "job_title": "",\n'
    '  "department": "",\n'
    '  "email": "",\n'
    '  "company_phone": "",\n'
    '  "mobile_phone": ""\n'
    "}"
)

USER_PROMPT = "이 이미지를 보고 명함 여부를 판단한 뒤, 명함이면 정보를 JSON으로 추출해줘."


# =========================================================
# 1. 공통 유틸
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_phone_fields(phones: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    HF labels.json의 phones 구조 예:
    [
      {"label": "T", "value": "032-4692-7897"},
      {"label": "M", "value": "010-7457-9993"}
    ]
    """
    company_phone = ""
    mobile_phone = ""

    if not phones:
        return company_phone, mobile_phone

    for item in phones:
        label = str(item.get("label", "")).strip().upper()
        value = str(item.get("value", "")).strip()

        if label in {"M", "MOBILE", "CELL", "HP"}:
            if not mobile_phone:
                mobile_phone = value
        else:
            if not company_phone:
                company_phone = value

    # 혹시 M이 없고 번호가 하나만 있으면 회사번호로 둠
    return company_phone, mobile_phone


def choose_best_text(ko: str, en: str, lang_type: str = "") -> str:
    """
    학습 타깃 JSON은 최종 서비스 스키마에 맞추기 위해
    우선 ko를 선호하고, 없으면 en 사용.
    """
    ko = (ko or "").strip()
    en = (en or "").strip()
    lang_type = (lang_type or "").strip().lower()

    if ko:
        return ko
    if en:
        return en
    return ""

def split_rows(rows: List[Dict[str, Any]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        return [], []
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio는 0과 1 사이여야 함: {train_ratio}")

    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    train_size = int(len(shuffled) * train_ratio)
    train_size = max(1, min(len(shuffled) - 1, train_size))

    train_rows = shuffled[:train_size]
    eval_rows = shuffled[train_size:]
    return train_rows, eval_rows


# =========================================================
# 2. Hugging Face dataset 다운로드 + 로드
# =========================================================
def download_hf_dataset_subset() -> str:
    """
    dataset repo의 dataset_3_10 폴더만 내려받음.
    결과 경로 예:
      ./hf_data_3_11/dataset_3_10/...
    """
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    local_path = snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[f"{HF_DATASET_SUBDIR}/*", f"{HF_DATASET_SUBDIR}/**"],
        local_dir=LOCAL_DATA_DIR,
        local_dir_use_symlinks=False,
    )

    return local_path


def load_hf_dataset_rows(base_dir: str) -> List[Dict[str, Any]]:
    """
    내려받은 구조:
      base_dir/
        dataset_3_10/
          images/
          labels/labels.json
    """
    dataset_root = os.path.join(base_dir, HF_DATASET_SUBDIR)
    labels_path = os.path.join(dataset_root, "labels", "labels.json")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json을 찾을 수 없음: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        raw_rows = json.load(f)

    rows = []
    for sample in raw_rows:
        company_phone, mobile_phone = extract_phone_fields(sample.get("phones", []))

        row = {
            "id": sample.get("id", ""),
            "image": os.path.join(dataset_root, sample.get("image", "")),
            "is_business_card": True,  # 이 repo는 명함 데이터셋으로 보이므로 true 처리
            "lang_type": sample.get("lang_type", ""),
            "difficulty_type": sample.get("difficulty_type", ""),
            "effects": sample.get("effects", []),
            "name": choose_best_text(
                sample.get("name_ko", ""),
                sample.get("name_en", ""),
                sample.get("lang_type", ""),
            ),
            "company": choose_best_text(
                sample.get("company_ko", ""),
                sample.get("company_en", ""),
                sample.get("lang_type", ""),
            ),
            "job_title": choose_best_text(
                sample.get("job_title_ko", ""),
                sample.get("job_title_en", ""),
                sample.get("lang_type", ""),
            ),
            "department": choose_best_text(
                sample.get("department_ko", ""),
                sample.get("department_en", ""),
                sample.get("lang_type", ""),
            ),
            "email": sample.get("email", ""),
            "company_phone": company_phone,
            "mobile_phone": mobile_phone,
        }
        rows.append(row)

    return rows


def build_datasets() -> Tuple[Dataset, Dataset]:
    downloaded_base = download_hf_dataset_subset()
    rows = load_hf_dataset_rows(downloaded_base)

    train_rows, eval_rows = split_rows(rows, TRAIN_RATIO, SEED)

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)
    return train_ds, eval_ds


# =========================================================
# 3. 학습용 타깃 JSON
# =========================================================
def build_target_json(sample: Dict[str, Any]) -> str:
    if sample["is_business_card"]:
        obj = {
            "is_business_card": True,
            "name": sample["name"],
            "company": sample["company"],
            "job_title": sample["job_title"],
            "department": sample["department"],
            "email": sample["email"],
            "company_phone": sample["company_phone"],
            "mobile_phone": sample["mobile_phone"],
        }
    else:
        obj = {
            "is_business_card": False,
            "name": "",
            "company": "",
            "job_title": "",
            "department": "",
            "email": "",
            "company_phone": "",
            "mobile_phone": "",
        }

    return json.dumps(obj, ensure_ascii=False)


def make_prompt_messages(image_path: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": USER_PROMPT,
                },
            ],
        },
    ]


def make_full_messages(image_path: str, assistant_text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": USER_PROMPT,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_text,
                }
            ],
        },
    ]


# =========================================================
# 4. 데이터 콜레이터
# =========================================================
@dataclass
class Qwen3VLDataCollator:
    processor: AutoProcessor
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        extra_storage = {}

        for feature in features:
            image_path = feature["image"]
            target_text = build_target_json(feature)

            prompt_messages = make_prompt_messages(image_path)
            full_messages = make_full_messages(image_path, target_text)

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            images, videos = process_vision_info(
                full_messages,
                image_patch_size=IMAGE_PATCH_SIZE,
            )

            full_inputs = self.processor(
                text=[full_text],
                images=images,
                videos=videos,
                do_resize=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            prompt_inputs = self.processor(
                text=[prompt_text],
                images=images,
                videos=videos,
                do_resize=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = full_inputs["input_ids"][0]
            attention_mask = full_inputs["attention_mask"][0]
            prompt_len = prompt_inputs["input_ids"].shape[1]

            labels = input_ids.clone()
            labels[:prompt_len] = -100

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

            for key, value in full_inputs.items():
                if key in ["input_ids", "attention_mask"]:
                    continue
                if key not in extra_storage:
                    extra_storage[key] = []
                extra_storage[key].append(value[0])

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                batch_input_ids, batch_first=True, padding_value=pad_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                batch_attention_mask, batch_first=True, padding_value=0
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                batch_labels, batch_first=True, padding_value=-100
            ),
        }

        for key, values in extra_storage.items():
            try:
                batch[key] = torch.stack(values, dim=0)
            except Exception:
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return batch


# =========================================================
# 5. QLoRA 모델 로드
# =========================================================
def load_model_and_processor():
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # vision + llm 양쪽 선형층 모두 적응
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    return model, processor


# =========================================================
# 6. 메인
# =========================================================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds, eval_ds = build_datasets()
    print(f"Train size: {len(train_ds)}")
    print(f"Eval size : {len(eval_ds)}")

    model, processor = load_model_and_processor()
    data_collator = Qwen3VLDataCollator(
        processor=processor,
        max_length=MAX_LENGTH,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()

    final_save_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.save_model(final_save_dir)
    processor.save_pretrained(final_save_dir)

    print(f"학습 완료: {final_save_dir}")


if __name__ == "__main__":
    main()
