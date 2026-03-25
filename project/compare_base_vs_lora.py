import os
import json
import random
import argparse
from typing import Dict, Any, List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


def resize_image_keep_ratio(image: Image.Image, max_side: int = 384) -> Image.Image:
    w, h = image.size
    long_side = max(w, h)

    if long_side <= max_side:
        new_w, new_h = w, h
    else:
        scale = max_side / long_side
        new_w = max(28, int(w * scale))
        new_h = max(28, int(h * scale))

    new_w = max(28, (new_w // 28) * 28)
    new_h = max(28, (new_h // 28) * 28)

    return image.resize((new_w, new_h))


def resolve_image_path(image_path: str, image_root: str) -> str:
    if os.path.isabs(image_path):
        return image_path
    return os.path.normpath(os.path.join(image_root, image_path))


def build_prompt() -> str:
    return (
        "Look at this image and extract the business card information using OCR. "
        "Read only the text that is clearly and actually visible in the image. "
        "If any value is missing, blurred, occluded, partially visible, or uncertain, "
        "do not guess and output an empty string instead. "
        "Do not translate, infer, paraphrase, or complete values. "
        "If multiple phone numbers are visible, extract up to two mobile phone numbers "
        "and up to two company phone numbers in reading order from top to bottom. "
        "Return only one JSON object with the following schema: "
        '{"name_ko":"","name_en":"","company_ko":"","company_en":"","job_title_ko":"","job_title_en":"","department_ko":"","department_en":"","email":"","mobile_phone_1":"","mobile_phone_2":"","company_phone_1":"","company_phone_2":""}'
    )


def extract_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start:end + 1]

    return text


def safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(extract_json_text(text))
    except Exception:
        return {"__raw_text__": text}


def generate_one(model, processor, image_path: str, image_max_side: int, max_new_tokens: int = 256) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    image = resize_image_keep_ratio(image, image_max_side)

    prompt = build_prompt()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    parsed = safe_parse_json(decoded)

    return {
        "raw_text": decoded,
        "parsed": parsed,
    }


def get_gt_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    assistant_text = sample["messages"][1]["content"]
    try:
        return json.loads(assistant_text)
    except Exception:
        return {"__raw_text__": assistant_text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="/root/output/qwen3_vl_4b_ocr_lora/final_adapter",
    )
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
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_max_side", type=int, default=384)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    picks = random.sample(data, min(args.num_samples, len(data)))

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    print("Loading base model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    base_model.eval()

    print("Loading adapter model...")
    ft_base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    ft_model = PeftModel.from_pretrained(ft_base_model, args.adapter_path)
    ft_model.eval()

    for i, sample in enumerate(picks, start=1):
        image_path = resolve_image_path(sample["image_path"], args.image_root)
        gt = get_gt_from_sample(sample)

        print("\n" + "=" * 120)
        print(f"[SAMPLE {i}]")
        print("image_path:", image_path)

        base_out = generate_one(
            model=base_model,
            processor=processor,
            image_path=image_path,
            image_max_side=args.image_max_side,
            max_new_tokens=args.max_new_tokens,
        )

        ft_out = generate_one(
            model=ft_model,
            processor=processor,
            image_path=image_path,
            image_max_side=args.image_max_side,
            max_new_tokens=args.max_new_tokens,
        )

        print("\n[GT]")
        print(json.dumps(gt, ensure_ascii=False, indent=2))

        print("\n[BASE MODEL]")
        print(json.dumps(base_out["parsed"], ensure_ascii=False, indent=2))

        print("\n[BASE + LORA ADAPTER]")
        print(json.dumps(ft_out["parsed"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()