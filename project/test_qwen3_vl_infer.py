import os
import json
import argparse

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

    # 가장 단순한 경우: 전체가 json
    if text.startswith("{") and text.endswith("}"):
        return text

    # 중간에 json이 섞여 있으면 첫 { ~ 마지막 } 추출
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start:end + 1]

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="/root/output/test_qwen3_vl_ocr_lora/final_adapter",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
    )
    parser.add_argument("--image_max_side", type=int, default=384)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {args.image_path}")

    print("=" * 80)
    print("Inference config")
    print(f"model_id: {args.model_id}")
    print(f"adapter_path: {args.adapter_path}")
    print(f"image_path: {args.image_path}")
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    image = resize_image_keep_ratio(image, args.image_max_side)

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
            max_new_tokens=args.max_new_tokens,
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

    print("\n[RAW OUTPUT]")
    print(decoded)

    json_text = extract_json_text(decoded)
    print("\n[EXTRACTED JSON TEXT]")
    print(json_text)

    try:
        parsed = json.loads(json_text)
        print("\n[PARSED JSON]")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception as e:
        print("\n[JSON PARSE FAILED]")
        print(e)


if __name__ == "__main__":
    main()