import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class EvalConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    use_lora: bool = False  # baseline 측정용 기본값
    adapter_path: Path = Path("/content/drive/MyDrive/qwen25vl_qlora_output/qlora_adapter")
    test_image_path: Path = Path("/content/KakaoTalk_Photo_2026-01-15-14-28-40 008.jpeg")
    run_dataset_eval: bool = False  # False면 단일 테스트 이미지 GT 비교만 실행

    # labels_train.json 우선 사용, 없으면 로컬 labels.json 사용
    label_candidates: tuple[Path, ...] = (
        Path("/content/drive/MyDrive/card_data/labels/labels_train.json"),
        Path("card_data/labels/labels_train.json"),
        Path("/content/drive/MyDrive/labels.json"),
        Path("labels.json"),
    )
    image_root_candidates: tuple[Path, ...] = (
        Path("/content/drive/MyDrive/card_data/images"),
        Path("card_data/images"),
        Path("business_card_dataset/images"),
    )

    # 빠른 실험용: None이면 전체 샘플 평가
    max_samples: int | None = None
    seed: int = 42

    # Qwen2.5-VL 이미지 토큰 범위
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 768 * 28 * 28

    # 생성 파라미터
    max_new_tokens: int = 256


CFG = EvalConfig()

# 평가 대상 필드 (요청한 7개)
TARGET_FIELDS = [
    "name",
    "email",
    "company",
    "mobile_phone",
    "company_phone",
    "job_title",
    "department",
]

# 단일 테스트 이미지 정답(GT)
# 사용자가 제공한 값을 기준으로 구성
TEST_IMAGE_GT = {
    "is_business_card": True,
    "name": "Park Jeonghyun",
    "email": "park.jeonghyun@snu.ac.kr",
    "company_phone": "02 880 1715",
    "mobile_phone": "010 5613 5620",
    "job_title": "Sr.Researcher",
    "department": "Institute of Engineering Research/Engineering Project Management(EMP) Program",
}


# -----------------------------------------------------------------------------
# [비활성] 기존 해상도별 루프 실험
# -----------------------------------------------------------------------------
# scales = [1.0, 0.75, 0.5, 0.35, 0.25]
# for scale in scales:
#     # 이미지 리사이즈 후 단일 추론 실험 로직
#     # 현재는 데이터셋 기반 정량평가로 대체
#     pass


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------
SYSTEM_TEXT = """
You are a careful document understanding system.
You must follow instructions strictly.
Do NOT perform OCR unless explicitly instructed to do so.
If any exclusion condition applies, you must classify the image as NOT a business card.
Output format rules are mandatory and must never be violated.
""".strip()

USER_TEXT = """
You are a document understanding system.

You must follow the steps below IN ORDER.
Failure to follow the order is considered an incorrect response.

────────────────────
[Step 1: Business Card Classification — NO OCR]
────────────────────
Determine whether the input image is a business card.

IMPORTANT:
- At this stage, you MUST NOT perform OCR.
- Use only visual characteristics and surface-level cues such as:
  - Overall layout and size
  - Card-like appearance
  - Typography style (printed vs handwritten)
  - Information density
  - Whether the image visually resembles a professional business card

[Conditions for a Business Card]
- The image visually resembles a small, professionally produced business card.
- Printed text is present (not handwritten).
- The content appears to represent personal or professional identification.
- A personal name MUST be visually identifiable.
- At least TWO of the following must be visually evident:
  - Email-like text pattern
  - Phone-number-like text pattern
  - Company name or job title

────────────────────
[Cases That Are NOT Business Cards]
────────────────────
- Flyers, posters, notices, official announcements, or advertisements
- Receipts, contracts, reports, invoices, or general document pages
- Product photos, menus, or website/app screenshots
- Images with little or no contact-style information
- Images whose purpose is NOT personal or professional identification
- Memo-like or draft-style layouts
- ANY handwritten content (names, numbers, emails, or sketches)

If ANY of the above applies,
you MUST classify the image as NOT a business card
and SKIP OCR entirely.

────────────────────
[Step 2: Conditional OCR — ONLY if Business Card]
────────────────────
ONLY IF the image is classified as a business card:

- Perform OCR on printed text only.
- Extract the following fields:
  - name
  - email
  - company
  - company_phone
  - mobile_phone
  - job_title
  - department

────────────────────
[Field Extraction Rules]
────────────────────

1. Name Extraction Rules
- The `name` field MUST contain ONLY the person's actual name.
- DO NOT include job titles, roles, departments, or honorifics.
- Examples of text that MUST NOT be included in `name`:
  - Sr. Researcher
  - Senior Researcher
  - Researcher
  - Engineer
  - Manager
  - Director
  - Ph.D., PhD, Dr.
- If the name is written in English:
  - Extract ONLY the proper name (e.g., "Park Jeonghyun")
  - Exclude adjacent job titles even if they appear on the same line or in smaller font.
- If multiple words are present:
  - Prefer capitalized name-like tokens
  - Exclude words indicating position or role

2. Phone Number Normalization Rules
- All phone numbers MUST be normalized to include hyphens ("-").
- Do NOT output raw digit-only numbers.
- Format phone numbers according to common Korean phone patterns.

Examples:
- Mobile phone:
  - 01012345678 → 010-1234-5678
- Company / representative phone:
  - 15881456 → 1588-1456
  - 0212359382 → 02-1235-9382
- If the phone number format cannot be confidently determined:
  - Still insert hyphens in a reasonable, standard grouping.

3. Field Presence Rules
- If a field does not exist or cannot be confidently identified:
  - Output an empty string "".
- Do NOT hallucinate or guess missing fields.

────────────────────
[Output Format]
────────────────────

Always output a single JSON object.

You MUST explicitly check EACH item in
[Cases That Are NOT Business Cards]
and report whether it applies.

If the image is NOT a business card, output EXACTLY:

{
  "is_business_card": false,
  "not_business_card_checks": {
    "flyer_or_advertisement": false,
    "receipt_or_document": false,
    "product_or_menu_image": false,
    "insufficient_contact_information": false,
    "not_personal_identification": false,
    "memo_or_draft_style": false,
    "handwritten_content": false
  }
}

If the image IS a business card, output EXACTLY:

{
  "is_business_card": true,
  "name": "",
  "email": "",
  "company": "",
  "company_phone": "",
  "mobile_phone": "",
  "job_title": "",
  "department": ""
}

────────────────────
[Strict Output Rules]
────────────────────
- Output JSON ONLY.
- Do NOT include markdown, comments, explanations, reasoning, confidence, or scores.
- Do NOT perform OCR unless Step 1 classifies the image as a business card.
- Every checklist field MUST be present.
""".strip()


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def pick_first_existing(paths: tuple[Path, ...], label: str) -> Path:
    for p in paths:
        if p.exists():
            print(f"[{label}] {p}")
            return p
    raise FileNotFoundError(f"{label} 경로를 찾지 못했습니다: {paths}")


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    text = str(x).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_email(x: str) -> str:
    return clean_text(x).lower()


def normalize_phone(x: str) -> str:
    # PhoneType 분류/매칭은 숫자만으로 비교
    return re.sub(r"\D", "", clean_text(x))


def normalize_field(field: str, value: Any) -> str:
    s = clean_text(value)
    if field == "email":
        return normalize_email(s)
    if field in {"mobile_phone", "company_phone"}:
        return normalize_phone(s)
    return s


def parse_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        low = v.strip().lower()
        if low in {"true", "1", "yes", "y"}:
            return True
        if low in {"false", "0", "no", "n"}:
            return False
    return None


def parse_json_object(text: str) -> dict | None:
    raw = clean_text(text)
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    if not raw:
        return None

    # 가장 바깥 JSON object 추출 시도
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return None

    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer_score(pred: str, ref: str) -> tuple[float, int, int]:
    pred = clean_text(pred)
    ref = clean_text(ref)

    # ref가 비어있으면 slot 관점 CER만 반영
    if ref == "":
        return (0.0 if pred == "" else 1.0), 0, 0

    dist = levenshtein_distance(pred, ref)
    return dist / len(ref), dist, len(ref)


def pick_value(data: dict, keys: list[str]) -> Any:
    for k in keys:
        if k in data and data[k] is not None:
            val = data[k]
            if isinstance(val, str):
                if val.strip():
                    return val
            else:
                return val
    return ""


def get_gt_object(record: dict) -> dict:
    # 1) messages[1].content dict 우선 (labels_train.json)
    if (
        isinstance(record.get("messages"), list)
        and len(record["messages"]) > 1
        and isinstance(record["messages"][1], dict)
        and isinstance(record["messages"][1].get("content"), dict)
    ):
        return record["messages"][1]["content"]

    # 2) top-level
    return record


def extract_fields(data: dict) -> dict:
    fields = {
        "name": pick_value(data, ["name"]),
        "email": pick_value(data, ["email"]),
        "company": pick_value(data, ["company", "company_name", "company_display", "company_raw"]),
        "mobile_phone": pick_value(data, ["mobile_phone", "mobile", "cell_phone"]),
        "company_phone": pick_value(data, ["company_phone", "office_phone", "tel"]),
        "job_title": pick_value(data, ["job_title", "title", "position", "role_title"]),
        "department": pick_value(data, ["department", "job_function", "role", "duty", "team"]),
    }

    # company/mobile 타입이 명시되지 않은 단일 phone 보정
    phone_fallback = pick_value(data, ["phone"])
    mode = clean_text(data.get("phone_mode", "")).lower()
    if phone_fallback and not fields["mobile_phone"] and not fields["company_phone"]:
        if any(t in mode for t in ["mobile", "cell", "m"]):
            fields["mobile_phone"] = phone_fallback
        elif any(t in mode for t in ["company", "office", "tel", "corp"]):
            fields["company_phone"] = phone_fallback

    # str 강제
    return {k: clean_text(v) for k, v in fields.items()}


def infer_is_business_card(data: dict, fields: dict) -> bool:
    explicit = parse_bool(data.get("is_business_card"))
    if explicit is not None:
        return explicit

    explicit_non = parse_bool(data.get("is_non_business_card"))
    if explicit_non is not None:
        return not explicit_non

    # fallback: 핵심 필드 중 하나라도 있으면 명함으로 간주
    return any(clean_text(v) for v in fields.values())


def resolve_image_path(record: dict, image_roots: list[Path]) -> Path:
    image_name = None
    if isinstance(record.get("images"), list) and record["images"]:
        image_name = record["images"][0]
    elif "file" in record:
        image_name = record["file"]

    if not image_name:
        raise ValueError("이미지 파일명을 찾지 못했습니다. (images[0] 또는 file 필요)")

    for root in image_roots:
        p = root / image_name
        if p.exists():
            return p
    raise FileNotFoundError(f"이미지를 찾지 못했습니다: {image_name} | roots={image_roots}")


# -----------------------------------------------------------------------------
# Model / Inference
# -----------------------------------------------------------------------------
def load_model_and_processor(cfg: EvalConfig):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )

    if cfg.use_lora:
        if cfg.adapter_path.exists():
            model = PeftModel.from_pretrained(model, str(cfg.adapter_path))
            print(f"[LoRA] adapter loaded: {cfg.adapter_path}")
        else:
            print(f"[LoRA] adapter requested but not found. base model only: {cfg.adapter_path}")
    else:
        print("[LoRA] disabled (baseline mode)")

    model.eval()

    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
    )
    return model, processor


def run_single_inference(model, processor, image_path: Path) -> tuple[str, dict | None]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": USER_TEXT},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=CFG.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return output_text, parse_json_object(output_text)


def run_test_image_once(model, processor, cfg: EvalConfig):
    if not cfg.test_image_path.exists():
        print(f"[TEST_IMAGE] not found: {cfg.test_image_path}")
        return

    print(f"[TEST_IMAGE] {cfg.test_image_path}")
    raw_text, pred_obj = run_single_inference(model, processor, cfg.test_image_path)
    print("\n[TEST_IMAGE OUTPUT RAW]")
    print(raw_text)
    print("\n[TEST_IMAGE OUTPUT JSON]")
    print(json.dumps(pred_obj if pred_obj is not None else {"parse_error": True}, ensure_ascii=False, indent=2))


def evaluate_test_image_with_gt(model, processor, cfg: EvalConfig, gt_obj: dict):
    if not cfg.test_image_path.exists():
        raise FileNotFoundError(f"테스트 이미지가 없습니다: {cfg.test_image_path}")

    print("\n================ TEST IMAGE CER ================")
    print(f"[TEST_IMAGE] {cfg.test_image_path}")

    raw_text, pred_obj = run_single_inference(model, processor, cfg.test_image_path)
    json_valid = pred_obj is not None
    if pred_obj is None:
        pred_obj = {}

    print("\n[TEST_IMAGE OUTPUT RAW]")
    print(raw_text)

    gt_fields = extract_fields(gt_obj)
    pred_fields = extract_fields(pred_obj)

    # GT에 명시된 필드만 CER 평가 (company처럼 미기입 필드는 자동 제외)
    eval_fields = [field for field in TARGET_FIELDS if field in gt_obj]
    if not eval_fields:
        eval_fields = TARGET_FIELDS[:]

    field_metrics = {}
    cer_sum = 0.0
    dist_sum = 0
    ref_chars_sum = 0
    hallucinated_slots = 0
    empty_gt_slots = 0

    for field in eval_fields:
        gt_norm = normalize_field(field, gt_fields.get(field, ""))
        pred_norm = normalize_field(field, pred_fields.get(field, ""))
        cer, dist, ref_chars = cer_score(pred_norm, gt_norm)

        field_metrics[field] = {
            "gt": gt_fields.get(field, ""),
            "pred": pred_fields.get(field, ""),
            "cer": cer,
            "edit_distance": dist,
            "ref_chars": ref_chars,
        }

        cer_sum += cer
        dist_sum += dist
        ref_chars_sum += ref_chars

        if gt_norm == "":
            empty_gt_slots += 1
            if pred_norm != "":
                hallucinated_slots += 1

    overall_cer = cer_sum / len(eval_fields) if eval_fields else float("nan")
    overall_cer_char_only = dist_sum / ref_chars_sum if ref_chars_sum > 0 else float("nan")
    hallucination_rate = hallucinated_slots / empty_gt_slots if empty_gt_slots > 0 else float("nan")

    gt_is_card = parse_bool(gt_obj.get("is_business_card"))
    pred_is_card = parse_bool(pred_obj.get("is_business_card"))
    is_business_card_match = (gt_is_card is not None and pred_is_card is not None and gt_is_card == pred_is_card)

    # 단일 샘플 phone type acc (mobile/company)
    phone_type_correct = 0
    phone_type_total_detected = 0
    phone_type_gt_total = 0
    gt_mobile = normalize_phone(gt_fields.get("mobile_phone", ""))
    gt_company = normalize_phone(gt_fields.get("company_phone", ""))
    pred_mobile = normalize_phone(pred_fields.get("mobile_phone", ""))
    pred_company = normalize_phone(pred_fields.get("company_phone", ""))
    for gt_phone, gt_type in [(gt_mobile, "mobile"), (gt_company, "company")]:
        if not gt_phone:
            continue
        phone_type_gt_total += 1
        found_as_mobile = pred_mobile == gt_phone
        found_as_company = pred_company == gt_phone
        if found_as_mobile or found_as_company:
            phone_type_total_detected += 1
            if (gt_type == "mobile" and found_as_mobile) or (gt_type == "company" and found_as_company):
                phone_type_correct += 1
    phone_type_acc = (
        phone_type_correct / phone_type_total_detected if phone_type_total_detected > 0 else float("nan")
    )
    phone_type_coverage = (
        phone_type_total_detected / phone_type_gt_total if phone_type_gt_total > 0 else float("nan")
    )

    test_summary = {
        "test_image_path": str(cfg.test_image_path),
        "json_valid": json_valid,
        "is_business_card_gt": gt_is_card,
        "is_business_card_pred": pred_is_card,
        "is_business_card_match": is_business_card_match,
        "fields_evaluated": eval_fields,
        "cer_per_field": {k: v["cer"] for k, v in field_metrics.items()},
        "overall_cer": overall_cer,
        "overall_cer_char_only": overall_cer_char_only,
        "hallucination_rate": hallucination_rate,
        "phone_type_acc_mobile_company": phone_type_acc,
        "phone_type_coverage": phone_type_coverage,
    }

    print("\n[TEST_IMAGE CER SUMMARY]")
    print(json.dumps(test_summary, ensure_ascii=False, indent=2))

    out = {
        "summary": test_summary,
        "field_details": field_metrics,
        "gt": gt_obj,
        "pred": pred_obj,
    }
    out_path = Path("single_test_image_eval.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Saved] {out_path.resolve()}")


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate_dataset(model=None, processor=None):
    torch.manual_seed(CFG.seed)

    label_file = pick_first_existing(CFG.label_candidates, "LABEL_FILE")
    image_roots = [p for p in CFG.image_root_candidates if p.exists()]
    if not image_roots:
        raise FileNotFoundError(f"IMAGE_ROOT 경로를 찾지 못했습니다: {CFG.image_root_candidates}")
    print(f"[IMAGE_ROOTS] {image_roots}")

    with open(label_file, encoding="utf-8") as f:
        records = json.load(f)

    if CFG.max_samples is not None:
        records = records[: CFG.max_samples]
    print(f"[DATA] samples={len(records)}")

    if model is None or processor is None:
        model, processor = load_model_and_processor(CFG)

    # metric accumulators
    total_samples = 0
    json_valid_count = 0
    skipped_count = 0

    field_cer_sum = {k: 0.0 for k in TARGET_FIELDS}
    field_cer_count = {k: 0 for k in TARGET_FIELDS}

    overall_slot_cer_sum = 0.0
    overall_slot_count = 0
    overall_dist_sum = 0
    overall_ref_char_sum = 0

    # hallucination: GT 빈칸인데 prediction이 채운 비율
    hallucinated_slots = 0
    empty_gt_slots = 0

    # phone type accuracy (mobile/company 구분)
    # 분모: GT phone이 pred mobile/company 중 어디엔가 나타난 경우
    phone_type_correct = 0
    phone_type_total_detected = 0
    phone_type_gt_total = 0

    # non-card F1
    non_tp = 0
    non_fp = 0
    non_fn = 0

    for idx, rec in enumerate(records, start=1):
        try:
            image_path = resolve_image_path(rec, image_roots)
        except Exception as e:
            skipped_count += 1
            print(f"[SKIP] idx={idx} image resolve failed: {e}")
            continue

        gt_obj = get_gt_object(rec)
        gt_fields = extract_fields(gt_obj)
        gt_is_card = infer_is_business_card(gt_obj, gt_fields)

        raw_text, pred_obj = run_single_inference(model, processor, image_path)
        pred_is_valid_json = pred_obj is not None

        total_samples += 1
        if pred_is_valid_json:
            json_valid_count += 1

        if pred_obj is None:
            pred_obj = {}

        pred_fields = extract_fields(pred_obj)
        pred_is_card = infer_is_business_card(pred_obj, pred_fields)

        # per-field CER + overall CER
        for field in TARGET_FIELDS:
            gt_norm = normalize_field(field, gt_fields[field])
            pred_norm = normalize_field(field, pred_fields[field])

            cer, dist, ref_chars = cer_score(pred_norm, gt_norm)

            field_cer_sum[field] += cer
            field_cer_count[field] += 1

            overall_slot_cer_sum += cer
            overall_slot_count += 1
            overall_dist_sum += dist
            overall_ref_char_sum += ref_chars

            if gt_norm == "":
                empty_gt_slots += 1
                if pred_norm != "":
                    hallucinated_slots += 1

        # phone type acc
        gt_mobile = normalize_phone(gt_fields["mobile_phone"])
        gt_company = normalize_phone(gt_fields["company_phone"])
        pred_mobile = normalize_phone(pred_fields["mobile_phone"])
        pred_company = normalize_phone(pred_fields["company_phone"])

        for gt_phone, gt_type in [(gt_mobile, "mobile"), (gt_company, "company")]:
            if not gt_phone:
                continue
            phone_type_gt_total += 1
            found_as_mobile = pred_mobile == gt_phone
            found_as_company = pred_company == gt_phone
            if found_as_mobile or found_as_company:
                phone_type_total_detected += 1
                if (gt_type == "mobile" and found_as_mobile) or (gt_type == "company" and found_as_company):
                    phone_type_correct += 1

        # non-card F1
        gt_non = not gt_is_card
        pred_non = not pred_is_card
        if gt_non and pred_non:
            non_tp += 1
        elif (not gt_non) and pred_non:
            non_fp += 1
        elif gt_non and (not pred_non):
            non_fn += 1

    # summarize
    if total_samples == 0:
        raise RuntimeError("평가 가능한 샘플이 없습니다.")

    field_cer = {
        k: (field_cer_sum[k] / field_cer_count[k] if field_cer_count[k] > 0 else float("nan"))
        for k in TARGET_FIELDS
    }
    overall_cer = overall_slot_cer_sum / overall_slot_count if overall_slot_count > 0 else float("nan")
    overall_cer_char_only = (
        overall_dist_sum / overall_ref_char_sum if overall_ref_char_sum > 0 else float("nan")
    )

    json_valid_rate = json_valid_count / total_samples
    hallucination_rate = hallucinated_slots / empty_gt_slots if empty_gt_slots > 0 else float("nan")

    # phone type acc
    phone_type_acc = (
        phone_type_correct / phone_type_total_detected if phone_type_total_detected > 0 else float("nan")
    )
    phone_type_coverage = (
        phone_type_total_detected / phone_type_gt_total if phone_type_gt_total > 0 else float("nan")
    )

    # non-card F1
    non_precision = non_tp / (non_tp + non_fp) if (non_tp + non_fp) > 0 else 0.0
    non_recall = non_tp / (non_tp + non_fn) if (non_tp + non_fn) > 0 else 0.0
    noncard_f1 = (
        2 * non_precision * non_recall / (non_precision + non_recall)
        if (non_precision + non_recall) > 0
        else 0.0
    )

    summary = {
        "samples_evaluated": total_samples,
        "samples_skipped": skipped_count,
        "fields": TARGET_FIELDS,
        "cer_per_field": field_cer,
        "overall_cer": overall_cer,
        "overall_cer_char_only": overall_cer_char_only,
        "phone_type_acc_mobile_company": phone_type_acc,
        "phone_type_coverage": phone_type_coverage,
        "noncard_f1": noncard_f1,
        "noncard_precision": non_precision,
        "noncard_recall": non_recall,
        "json_valid_rate": json_valid_rate,
        "hallucination_rate": hallucination_rate,
    }

    print("\n================ EVAL SUMMARY ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_path = Path("inference_eval_summary.json")
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Saved] {out_path.resolve()}")


if __name__ == "__main__":
    model, processor = load_model_and_processor(CFG)
    evaluate_test_image_with_gt(model, processor, CFG, TEST_IMAGE_GT)
    if CFG.run_dataset_eval:
        evaluate_dataset(model, processor)
