import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    Qwen3VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# =========================================================
# 1. 설정
# =========================================================
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

DATA_ROOT = "/root/dataset_3_10-2"
TRAIN_JSONL = os.path.join(DATA_ROOT, "qwen_field_crop_messages_train.jsonl")
VAL_JSONL = os.path.join(DATA_ROOT, "qwen_field_crop_messages_val.jsonl")
OUTPUT_DIR = os.path.join(DATA_ROOT, "qwen3_vl_4b_field_crop_lora")

# 처음엔 작게 sanity check
MAX_TRAIN_SAMPLES = 50
MAX_VAL_SAMPLES = 10
NUM_EPOCHS = 1

# 전체 학습 시 이렇게 바꾸기
# MAX_TRAIN_SAMPLES = None
# MAX_VAL_SAMPLES = None
# NUM_EPOCHS = 3

SEED = 42

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-5
WARMUP_STEPS = 20
LOGGING_STEPS = 5
SAVE_STEPS = 50


# =========================================================
# 2. seed
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 3. jsonl 로드
# =========================================================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] line {line_idx} parse failed: {e}")
    return rows


# =========================================================
# 4. crop 이미지 경로 절대경로 보정
# =========================================================
def fix_crop_image_paths(rows: List[Dict[str, Any]], data_root: str) -> List[Dict[str, Any]]:
    fixed = []

    for row in rows:
        row = json.loads(json.dumps(row, ensure_ascii=False))

        for msg in row.get("messages", []):
            if msg.get("role") != "user":
                continue

            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img = item.get("image", "")
                    if img and not os.path.isabs(img):
                        item["image"] = os.path.join(data_root, img)

        fixed.append(row)

    return fixed


# =========================================================
# 5. HF Dataset 저장용 직렬화
# =========================================================
def serialize_rows_for_dataset(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized = []

    for idx, row in enumerate(rows):
        serialized.append({
            "sample_id": str(row.get("id", idx)),
            "messages_json": json.dumps(row.get("messages", []), ensure_ascii=False),
        })

    return serialized


# =========================================================
# 6. 전처리 유틸
# =========================================================
def normalize_user_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, list):
        normalized = []
        for item in content:
            if isinstance(item, dict):
                item = dict(item)
                item_type = item.get("type")

                if item_type == "text":
                    item["text"] = str(item.get("text", item.get("content", "")))
                    item.pop("content", None)
                    normalized.append(item)
                elif item_type in {"image", "video"}:
                    normalized.append(item)
                else:
                    normalized.append({
                        "type": "text",
                        "text": json.dumps(item, ensure_ascii=False),
                    })
            elif item is not None:
                normalized.append({"type": "text", "text": str(item)})
        return normalized

    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if isinstance(content, dict):
        item = dict(content)
        item_type = item.get("type")

        if item_type == "text":
            item["text"] = str(item.get("text", item.get("content", "")))
            item.pop("content", None)
            return [item]

        if item_type in {"image", "video"}:
            return [item]

        return [{"type": "text", "text": json.dumps(item, ensure_ascii=False)}]

    if content is None:
        return []

    return [{"type": "text", "text": str(content)}]


def assistant_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", item.get("content", ""))))
            elif isinstance(item, str):
                parts.append(item)
            elif item is not None:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)

    if content is None:
        return ""

    return str(content)


def parse_messages(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_messages = example.get("messages_json", example.get("messages", []))

    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)

    if not isinstance(raw_messages, list):
        raise ValueError(f"messages must be a list, got {type(raw_messages).__name__}")

    messages = []
    for msg in raw_messages:
        if isinstance(msg, dict):
            messages.append(dict(msg))

    if not messages:
        raise ValueError("messages is empty")

    return messages


def ensure_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value

    if dtype is not None:
        return torch.tensor(value, dtype=dtype)

    return torch.as_tensor(value)


def normalize_extra_model_input(key: str, value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value

    if key in {"pixel_values", "pixel_values_videos"}:
        if value.dim() == 3 and value.shape[0] == 1:
            return value[0]
        return value

    if key in {"image_grid_thw", "video_grid_thw"}:
        if value.dim() == 3 and value.shape[0] == 1:
            value = value[0]
        if value.dim() == 1:
            value = value.unsqueeze(0)
        return value

    if value.shape[0] == 1:
        return value[0]

    return value


def normalize_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", item.get("content", ""))))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", content.get("content", "")))
        return json.dumps(content, ensure_ascii=False)

    if content is None:
        return ""

    return str(content)


def normalize_messages_for_training(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []

    for msg in messages:
        role = str(msg.get("role", "")).strip()
        if not role:
            continue

        normalized_msg = {"role": role}
        content = msg.get("content")

        if role == "user":
            normalized_msg["content"] = normalize_user_content(content)
        elif role == "assistant":
            normalized_msg["content"] = [
                {
                    "type": "text",
                    "text": assistant_content_to_text(content),
                }
            ]
        else:
            normalized_msg["content"] = normalize_text_content(content)

        normalized.append(normalized_msg)

    return normalized


# =========================================================
# 7. 전처리
# =========================================================
def preprocess_example(example: Dict[str, Any], processor) -> Dict[str, Any]:
    messages = normalize_messages_for_training(parse_messages(example))

    assistant_idx = next(
        (idx for idx in range(len(messages) - 1, -1, -1) if messages[idx].get("role") == "assistant"),
        None,
    )

    if assistant_idx is None:
        raise ValueError("example must contain at least one assistant message")

    if not any(msg.get("role") == "user" for msg in messages[:assistant_idx]):
        raise ValueError("example must contain at least one user message and one assistant message")

    prompt_messages = messages[:assistant_idx]
    full_messages = messages[:assistant_idx + 1]

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    images, videos = process_vision_info(full_messages)

    full_inputs = processor(
        text=[full_text],
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=False,
    )

    prompt_inputs = processor(
        text=[prompt_text],
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=False,
    )

    input_ids = full_inputs["input_ids"][0]
    attention_mask = full_inputs["attention_mask"][0]
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:prompt_len] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    for k, v in full_inputs.items():
        if k in ["input_ids", "attention_mask"]:
            continue
        out[k] = normalize_extra_model_input(k, v)

    return out


# =========================================================
# 8. data collator
# =========================================================
@dataclass
class QwenVLDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id or 0

        input_ids_list = [ensure_tensor(f["input_ids"], torch.long) for f in features]
        attention_mask_list = [ensure_tensor(f["attention_mask"], torch.long) for f in features]
        labels_list = [ensure_tensor(f["labels"], torch.long) for f in features]

        batch["input_ids"] = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_id,
        )
        batch["attention_mask"] = pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0,
        )
        batch["labels"] = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100,
        )

        extra_keys = [k for k in features[0].keys() if k not in {"input_ids", "attention_mask", "labels"}]
        for k in extra_keys:
            vals = [f[k] for f in features]

            if k in {"pixel_values", "pixel_values_videos"}:
                batch[k] = torch.cat([ensure_tensor(v) for v in vals], dim=0)
                continue

            if k in {"image_grid_thw", "video_grid_thw"}:
                normalized_vals = []
                for v in vals:
                    tensor_v = ensure_tensor(v, torch.long)
                    if tensor_v.dim() == 1:
                        tensor_v = tensor_v.unsqueeze(0)
                    normalized_vals.append(tensor_v)
                batch[k] = torch.cat(normalized_vals, dim=0)
                continue

            try:
                batch[k] = torch.stack([ensure_tensor(v) for v in vals])
            except Exception:
                batch[k] = vals

        return batch


# =========================================================
# 9. 메인
# =========================================================
def main():
    set_seed(SEED)

    print("=== path check ===")
    print("DATA_ROOT:", DATA_ROOT, os.path.exists(DATA_ROOT))
    print("TRAIN_JSONL:", TRAIN_JSONL, os.path.exists(TRAIN_JSONL))
    print("VAL_JSONL:", VAL_JSONL, os.path.exists(VAL_JSONL))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== load jsonl ===")
    train_rows = load_jsonl(TRAIN_JSONL)
    val_rows = load_jsonl(VAL_JSONL)

    if MAX_TRAIN_SAMPLES is not None:
        train_rows = train_rows[:MAX_TRAIN_SAMPLES]
    if MAX_VAL_SAMPLES is not None:
        val_rows = val_rows[:MAX_VAL_SAMPLES]

    print(f"train rows: {len(train_rows)}")
    print(f"val rows  : {len(val_rows)}")

    print("\n=== fix image paths ===")
    train_rows = fix_crop_image_paths(train_rows, DATA_ROOT)
    val_rows = fix_crop_image_paths(val_rows, DATA_ROOT)

    try:
        sample_img = train_rows[0]["messages"][0]["content"][0]["image"]
        print("sample image path:", sample_img)
        print("sample image exists:", os.path.exists(sample_img))
    except Exception as e:
        print("sample image check failed:", e)

    train_dataset = Dataset.from_list(serialize_rows_for_dataset(train_rows))
    val_dataset = Dataset.from_list(serialize_rows_for_dataset(val_rows))

    print("\n=== load processor/model ===")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    print("\n=== apply lora ===")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n=== preprocess dataset ===")
    train_dataset = train_dataset.map(
        lambda x: preprocess_example(x, processor),
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_example(x, processor),
        remove_columns=val_dataset.column_names,
    )

    print("train dataset keys:", train_dataset[0].keys())
    print("len(input_ids):", len(train_dataset[0]["input_ids"]))
    print("len(labels):", len(train_dataset[0]["labels"]))

    data_collator = QwenVLDataCollator(processor=processor)

    print("\n=== training args ===")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=2,
        lr_scheduler_type="cosine",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\n=== start training ===")
    train_result = trainer.train()

    print("\n=== save final adapter ===")
    final_adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)

    print("\ntraining finished.")
    print(train_result)
    print(f"final adapter saved to: {final_adapter_dir}")


if __name__ == "__main__":
    main()
