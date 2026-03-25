import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from datasets import Dataset
from PIL import Image

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)


# =========================================================
# 1. 유틸
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_image_keep_ratio(image: Image.Image, max_side: int = 384) -> Image.Image:
    w, h = image.size
    long_side = max(w, h)

    if long_side <= max_side:
        new_w, new_h = w, h
    else:
        scale = max_side / long_side
        new_w = max(28, int(w * scale))
        new_h = max(28, int(h * scale))

    # Qwen-VL 계열 안정성 위해 28 배수 정렬
    new_w = max(28, (new_w // 28) * 28)
    new_h = max(28, (new_h // 28) * 28)

    return image.resize((new_w, new_h))


def resolve_image_path(image_path: str, image_root: str) -> str:
    if os.path.isabs(image_path):
        return image_path
    return os.path.normpath(os.path.join(image_root, image_path))


# =========================================================
# 2. Data Collator
# assistant 답변 부분만 loss 계산
# =========================================================
@dataclass
class QwenVLDataCollator:
    processor: Any
    image_root: str
    max_side: int = 384

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        mm_token_type_ids_list = []
        pixel_values_list = []
        image_grid_thw_list = []

        for ex in features:
            image_path = resolve_image_path(ex["image_path"], self.image_root)
            messages = ex["messages"]

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

            if len(messages) < 2:
                raise ValueError(f"messages 길이가 2보다 작습니다: {messages}")

            user_text = messages[0]["content"]
            assistant_text = messages[1]["content"]

            image = Image.open(image_path).convert("RGB")
            image = resize_image_keep_ratio(image, self.max_side)

            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": assistant_text,
                },
            ]

            prefix_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]

            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prefix_text = self.processor.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_batch = self.processor(
                text=[full_text],
                images=[image],
                padding=False,
                return_tensors="pt",
            )

            prefix_batch = self.processor(
                text=[prefix_text],
                images=[image],
                padding=False,
                return_tensors="pt",
            )

            input_ids = full_batch["input_ids"][0]
            attention_mask = full_batch["attention_mask"][0]
            prefix_len = prefix_batch["input_ids"].shape[1]

            labels = input_ids.clone()
            labels[:prefix_len] = -100

            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

            if "mm_token_type_ids" not in full_batch:
                raise ValueError("processor output에 mm_token_type_ids가 없습니다.")
            mm_token_type_ids_list.append(full_batch["mm_token_type_ids"][0])

            if "pixel_values" in full_batch:
                pixel_values_list.append(full_batch["pixel_values"][0])

            if "image_grid_thw" in full_batch:
                image_grid_thw_list.append(full_batch["image_grid_thw"][0])

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id가 None입니다.")

        max_len = max(x.size(0) for x in input_ids_list)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_mm_token_type_ids = []

        for input_ids, attention_mask, labels, mm_token_type_ids in zip(
            input_ids_list,
            attention_mask_list,
            labels_list,
            mm_token_type_ids_list,
        ):
            pad_len = max_len - input_ids.size(0)

            if pad_len > 0:
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)]
                )
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros((pad_len,), dtype=attention_mask.dtype)]
                )
                labels = torch.cat(
                    [labels, torch.full((pad_len,), -100, dtype=labels.dtype)]
                )
                mm_token_type_ids = torch.cat(
                    [mm_token_type_ids, torch.zeros((pad_len,), dtype=mm_token_type_ids.dtype)]
                )

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_mm_token_type_ids.append(mm_token_type_ids)

        batch = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
            "mm_token_type_ids": torch.stack(batch_mm_token_type_ids),
        }

        if len(pixel_values_list) > 0:
            batch["pixel_values"] = torch.stack(pixel_values_list)

        if len(image_grid_thw_list) > 0:
            batch["image_grid_thw"] = torch.stack(image_grid_thw_list)

        return batch


# =========================================================
# 3. main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="/root/data/qwen3_vl_visible_only_phone_expanded/train_chat_300_visible_only_phone_expanded.json",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/root/data/qwen3_vl_visible_only_phone_expanded/dataset_3_10",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/output/qwen3_vl_4b_ocr_lora",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=20)

    parser.add_argument("--image_max_side", type=int, default=384)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print("=" * 80)
    print("Train config")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 80)

    # -----------------------------------------------------
    # 데이터 로드
    # -----------------------------------------------------
    with open(args.train_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if args.max_samples is not None and args.max_samples > 0:
        raw_data = raw_data[: args.max_samples]

    dataset = Dataset.from_list(raw_data)

    print(f"dataset size: {len(dataset)}")
    print(f"train_json_path: {args.train_json_path}")
    print(f"image_root: {args.image_root}")

    # 샘플 이미지 경로 확인
    if len(raw_data) > 0:
        sample_img = resolve_image_path(raw_data[0]["image_path"], args.image_root)
        print(f"sample resolved image path: {sample_img}")
        print(f"sample image exists: {os.path.exists(sample_img)}")

    # -----------------------------------------------------
    # processor / model
    # -----------------------------------------------------
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False

    # -----------------------------------------------------
    # LoRA
    # -----------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------------------------------
    # collator
    # -----------------------------------------------------
    data_collator = QwenVLDataCollator(
        processor=processor,
        image_root=args.image_root,
        max_side=args.image_max_side,
    )

    # -----------------------------------------------------
    # train args
    # -----------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=False,
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    # -----------------------------------------------------
    # trainer
    # -----------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # -----------------------------------------------------
    # train
    # -----------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.train()

    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)

    model.save_pretrained(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)

    print(f"학습 완료. adapter 저장 경로: {final_adapter_dir}")


if __name__ == "__main__":
    main()