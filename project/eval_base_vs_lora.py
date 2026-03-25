import os
import json
import random
import argparse
import difflib
from typing import Dict, Any, List, Tuple, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


TARGET_FIELDS = [
    "name_ko",
    "name_en",
    "company_ko",
    "company_en",
    "job_title_ko",
    "job_title_en",
    "department_ko",
    "department_en",
    "email",
    "mobile_phone_1",
    "mobile_phone_2",
    "company_phone_1",
    "company_phone_2",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


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


def list_image_files(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"평가 디렉터리를 찾을 수 없습니다: {dir_path}")

    files = []
    for name in sorted(os.listdir(dir_path)):
        path = os.path.join(dir_path, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(path)

    return files


def resolve_existing_dir(dir_path: str, search_roots: Optional[List[str]] = None) -> str:
    raw_path = str(dir_path).strip()
    search_roots = [os.path.abspath(p) for p in (search_roots or [])]

    candidates = []
    seen = set()

    def add_candidate(path: str):
        norm = os.path.normpath(os.path.abspath(path))
        if norm not in seen:
            seen.add(norm)
            candidates.append(norm)

    if os.path.isabs(raw_path):
        add_candidate(raw_path)
        base_name = os.path.basename(os.path.normpath(raw_path))
        if base_name:
            for root in search_roots:
                add_candidate(os.path.join(root, base_name))
    else:
        add_candidate(raw_path)
        for root in search_roots:
            add_candidate(os.path.join(root, raw_path))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"평가 디렉터리를 찾을 수 없습니다: requested={dir_path} | searched={candidates}"
    )


def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def discover_adapter_dirs(search_roots: List[str]) -> List[str]:
    found = []
    seen = set()

    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue

        for current_root, dirnames, filenames in os.walk(root):
            if "adapter_config.json" in filenames:
                norm = os.path.normpath(current_root)
                if norm not in seen:
                    seen.add(norm)
                    found.append(norm)
                dirnames[:] = []

    return sorted(found)


def resolve_adapter_path(adapter_path: str, search_roots: Optional[List[str]] = None) -> str:
    adapter_path = os.path.abspath(adapter_path)
    search_roots = [os.path.abspath(p) for p in (search_roots or [])]

    if os.path.isfile(adapter_path) and os.path.basename(adapter_path) == "adapter_config.json":
        return os.path.dirname(adapter_path)

    direct_config = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(direct_config):
        return adapter_path

    candidate_dirs = []

    if os.path.isdir(adapter_path):
        for name in ["final_adapter", "best_adapter"]:
            candidate = os.path.join(adapter_path, name)
            if os.path.isdir(candidate):
                candidate_dirs.append(candidate)

        for name in sorted(os.listdir(adapter_path)):
            candidate = os.path.join(adapter_path, name)
            if os.path.isdir(candidate) and name.startswith("checkpoint-"):
                candidate_dirs.append(candidate)

    for candidate in candidate_dirs:
        if os.path.isfile(os.path.join(candidate, "adapter_config.json")):
            return candidate

    discovered = discover_adapter_dirs(search_roots)
    if discovered:
        target = adapter_path.replace("\\", "/")
        ranked = sorted(
            discovered,
            key=lambda path: difflib.SequenceMatcher(None, target, path.replace("\\", "/")).ratio(),
            reverse=True,
        )

        best = ranked[0]
        best_score = difflib.SequenceMatcher(None, target, best.replace("\\", "/")).ratio()
        if best_score >= 0.45:
            print(f"[INFO] requested adapter not found, using closest adapter: {best}")
            return best

    searched = [adapter_path] + candidate_dirs + discovered
    raise FileNotFoundError(
        "LoRA adapter 경로를 찾지 못했습니다. "
        f"'adapter_config.json' 이 필요합니다. searched={searched}"
    )


def build_prompt() -> str:
    return (
        "Look at this image and extract the business card information using OCR. "
        "Return only one JSON object in the following format with no explanation, no markdown, and no extra text.\n"
        "{"
        '"name_ko":"","name_en":"","company_ko":"","company_en":"","job_title_ko":"","job_title_en":"","department_ko":"","department_en":"","email":"","mobile_phone_1":"","mobile_phone_2":"","company_phone_1":"","company_phone_2":""'
        "}\n"
        "If a value is missing, output an empty string."
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


def empty_prediction() -> Dict[str, Any]:
    return {field: "" for field in TARGET_FIELDS}


def safe_parse_json(text: str) -> Tuple[Dict[str, Any], bool]:
    try:
        parsed = json.loads(extract_json_text(text))

        # 혹시 모델이 일부 key만 반환해도 평가용으로는 TARGET_FIELDS 기준으로 정렬
        normalized = {}
        for field in TARGET_FIELDS:
            normalized[field] = parsed.get(field, "") if isinstance(parsed, dict) else ""

        return normalized, True
    except Exception:
        return empty_prediction(), False


def generate_one(
    model,
    processor,
    image_path: str,
    image_max_side: int,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
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

    parsed_json, parse_success = safe_parse_json(decoded)

    return {
        "raw_text": decoded,
        "parsed_json": parsed_json,
        "parse_success": parse_success,
    }


def make_parse_summary(
    dataset_name: str,
    total: int,
    base_parse_fail_count: int,
    ft_parse_fail_count: int,
) -> Dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "num_samples": total,
        "base_parse_fail_count": base_parse_fail_count,
        "base_parse_fail_rate": round(base_parse_fail_count / total, 4) if total > 0 else 0.0,
        "base_parse_success_count": total - base_parse_fail_count,
        "base_parse_success_rate": round((total - base_parse_fail_count) / total, 4) if total > 0 else 0.0,
        "ft_parse_fail_count": ft_parse_fail_count,
        "ft_parse_fail_rate": round(ft_parse_fail_count / total, 4) if total > 0 else 0.0,
        "ft_parse_success_count": total - ft_parse_fail_count,
        "ft_parse_success_rate": round((total - ft_parse_fail_count) / total, 4) if total > 0 else 0.0,
    }


def evaluate_labeled_dataset(
    data: List[Dict[str, Any]],
    base_model,
    ft_model,
    processor,
    image_root: str,
    image_max_side: int,
    max_new_tokens: int,
    output_dir: str,
    eval_json_path: str,
):
    results = []
    dataset_name = os.path.splitext(os.path.basename(eval_json_path))[0]

    base_parse_fail_count = 0
    ft_parse_fail_count = 0

    for idx, sample in enumerate(data, start=1):
        image_path = resolve_image_path(sample["image_path"], image_root)
        gt = json.loads(sample["messages"][1]["content"])

        base_out = generate_one(base_model, processor, image_path, image_max_side, max_new_tokens)
        ft_out = generate_one(ft_model, processor, image_path, image_max_side, max_new_tokens)

        if not base_out["parse_success"]:
            base_parse_fail_count += 1
        if not ft_out["parse_success"]:
            ft_parse_fail_count += 1

        results.append({
            "image_path": image_path,
            "ground_truth": gt,

            "base_raw_text": base_out["raw_text"],
            "base_pred": base_out["parsed_json"],
            "base_parse_success": base_out["parse_success"],

            "base_lora_raw_text": ft_out["raw_text"],
            "base_lora_pred": ft_out["parsed_json"],
            "base_lora_parse_success": ft_out["parse_success"],
        })

        print(
            f"[labeled {idx}/{len(data)}] done | "
            f"base_parse_success={base_out['parse_success']} | "
            f"ft_parse_success={ft_out['parse_success']}"
        )

    output_path = os.path.join(output_dir, f"{dataset_name}_predictions.json")
    save_json(results, output_path)
    print(f"saved: {output_path}")

    summary = make_parse_summary(
        dataset_name=dataset_name,
        total=len(results),
        base_parse_fail_count=base_parse_fail_count,
        ft_parse_fail_count=ft_parse_fail_count,
    )

    summary_path = os.path.join(output_dir, f"{dataset_name}_parse_summary.json")
    save_json(summary, summary_path)
    print(f"saved: {summary_path}")

    print("\n=== Parse Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_image_directory(
    dir_path: str,
    base_model,
    ft_model,
    processor,
    image_max_side: int,
    max_new_tokens: int,
    max_samples: Optional[int],
    output_dir: str,
):
    image_paths = list_image_files(dir_path)
    if max_samples is not None and max_samples > 0:
        image_paths = image_paths[:max_samples]

    if not image_paths:
        print(f"[WARN] no images found in: {dir_path}")
        return

    dataset_name = os.path.basename(os.path.normpath(dir_path))
    results = []

    base_parse_fail_count = 0
    ft_parse_fail_count = 0

    for idx, image_path in enumerate(image_paths, start=1):
        base_out = generate_one(base_model, processor, image_path, image_max_side, max_new_tokens)
        ft_out = generate_one(ft_model, processor, image_path, image_max_side, max_new_tokens)

        if not base_out["parse_success"]:
            base_parse_fail_count += 1
        if not ft_out["parse_success"]:
            ft_parse_fail_count += 1

        results.append({
            "image_path": image_path,

            "base_raw_text": base_out["raw_text"],
            "base_pred": base_out["parsed_json"],
            "base_parse_success": base_out["parse_success"],

            "base_lora_raw_text": ft_out["raw_text"],
            "base_lora_pred": ft_out["parsed_json"],
            "base_lora_parse_success": ft_out["parse_success"],
        })

        print(
            f"[{dataset_name} {idx}/{len(image_paths)}] done | "
            f"base_parse_success={base_out['parse_success']} | "
            f"ft_parse_success={ft_out['parse_success']}"
        )

    output_path = os.path.join(output_dir, f"{dataset_name}_predictions.json")
    save_json(results, output_path)
    print(f"saved: {output_path}")

    summary = make_parse_summary(
        dataset_name=dataset_name,
        total=len(results),
        base_parse_fail_count=base_parse_fail_count,
        ft_parse_fail_count=ft_parse_fail_count,
    )

    summary_path = os.path.join(output_dir, f"{dataset_name}_parse_summary.json")
    save_json(summary, summary_path)
    print(f"saved: {summary_path}")

    print("\n=== Parse Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--eval_json_path", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--eval_dirs", type=str, nargs="*", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_max_side", type=int, default=384)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()
    random.seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)
    cwd = os.getcwd()
    dir_search_roots = [
        cwd,
        script_dir,
        workspace_dir,
        os.path.dirname(cwd),
        os.path.dirname(workspace_dir),
    ]

    eval_dirs = args.eval_dirs
    if not args.eval_json_path and not eval_dirs:
        eval_dirs = ["real_data", "down_data"]

    if not args.eval_json_path and not eval_dirs:
        parser.error("either --eval_json_path or --eval_dirs is required")

    if args.eval_json_path and not args.image_root:
        parser.error("--image_root is required when --eval_json_path is used")

    output_dir = args.output_dir or os.path.join(script_dir, "eval_outputs")
    os.makedirs(output_dir, exist_ok=True)
    adapter_search_roots = [
        os.path.join(workspace_dir, "output"),
        os.path.join(script_dir, "output"),
        os.path.join(os.getcwd(), "output"),
        workspace_dir,
        script_dir,
        os.getcwd(),
    ]
    adapter_path = resolve_adapter_path(args.adapter_path, adapter_search_roots)
    resolved_eval_dirs = [resolve_existing_dir(path, dir_search_roots) for path in (eval_dirs or [])]

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

    print("Loading base + adapter model...")
    ft_base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"Resolved adapter path: {adapter_path}")
    if resolved_eval_dirs:
        print(f"Resolved eval dirs: {resolved_eval_dirs}")
    ft_model = PeftModel.from_pretrained(ft_base_model, adapter_path)
    ft_model.eval()

    if args.eval_json_path:
        with open(args.eval_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if args.max_samples is not None and args.max_samples > 0:
            data = data[: args.max_samples]

        evaluate_labeled_dataset(
            data=data,
            base_model=base_model,
            ft_model=ft_model,
            processor=processor,
            image_root=args.image_root,
            image_max_side=args.image_max_side,
            max_new_tokens=args.max_new_tokens,
            output_dir=output_dir,
            eval_json_path=args.eval_json_path,
        )

    for dir_path in resolved_eval_dirs:
        evaluate_image_directory(
            dir_path=dir_path,
            base_model=base_model,
            ft_model=ft_model,
            processor=processor,
            image_max_side=args.image_max_side,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
