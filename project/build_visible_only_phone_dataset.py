import os
import re
import json
import random
from typing import List, Dict, Any, Tuple

from PIL import Image
import matplotlib.pyplot as plt


# =========================================================
# 0. 설정
# =========================================================
DATA_ROOT = "/root/data/qwen3_vl_visible_only_phone_expanded/dataset_3_10"
ANN_DIR = os.path.join(DATA_ROOT, "annotations")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")

OUTPUT_DIR = "/root/data/qwen3_vl_visible_only_phone_expanded"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
SUBSET_SIZE = 300

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

random.seed(SEED)


# =========================================================
# 1. 유틸
# =========================================================
def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_email(x: Any) -> str:
    return normalize_text(x).lower()


def normalize_phone(x: Any) -> str:
    x = normalize_text(x)
    if not x:
        return ""
    x = x.replace(".", "-")
    x = re.sub(r"[()]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def save_json(data: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_visible_texts(word_boxes: List[Dict[str, Any]]) -> List[str]:
    texts = []
    for wb in word_boxes:
        t = normalize_text(wb.get("text", ""))
        if t:
            texts.append(t)
    return texts


def is_value_visible(value: str, visible_texts: List[str]) -> bool:
    value = normalize_text(value)
    if not value:
        return False

    if value in visible_texts:
        return True

    value_ns = re.sub(r"\s+", "", value)
    visible_ns = [re.sub(r"\s+", "", t) for t in visible_texts]

    if value_ns in visible_ns:
        return True

    for t in visible_texts:
        t_ns = re.sub(r"\s+", "", t)
        if value_ns and (value_ns in t_ns or t_ns in value_ns):
            return True

    return False


def make_relative_image_path(abs_image_path: str) -> str:
    """
    학습 JSON에는 절대경로 대신 상대경로 저장:
    images/card_000001.png
    """
    filename = os.path.basename(abs_image_path)
    return os.path.join("images", filename)


# =========================================================
# 2. phones 파싱
# =========================================================
def extract_phone_lists(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    mobile_phones = []
    company_phones = []

    phones = row.get("phones", [])
    if not isinstance(phones, list):
        return mobile_phones, company_phones

    for p in phones:
        if not isinstance(p, dict):
            continue

        label = str(p.get("label", "")).strip().lower()
        value = normalize_phone(p.get("value", ""))

        if not value:
            continue

        if label in ["m", "mobile", "cell", "hp", "hand", "handphone"]:
            mobile_phones.append(value)
        elif label in ["t", "tel", "telephone", "office", "office:", "work"]:
            company_phones.append(value)
        else:
            digits = re.sub(r"\D", "", value)
            if digits.startswith("010") or digits.startswith("8210"):
                mobile_phones.append(value)
            else:
                company_phones.append(value)

    mobile_phones = list(dict.fromkeys(mobile_phones))
    company_phones = list(dict.fromkeys(company_phones))

    return mobile_phones, company_phones


def pad_phone_fields(phone_list: List[str], max_count: int = 2) -> List[str]:
    phone_list = phone_list[:max_count]
    while len(phone_list) < max_count:
        phone_list.append("")
    return phone_list


# =========================================================
# 3. visible-only 정제
# =========================================================
def sanitize_by_lang_type(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    lang_type = normalize_text(row.get("lang_type", "")).lower()

    if lang_type == "english_only":
        row["name_ko"] = ""
        row["company_ko"] = ""
        row["job_title_ko"] = ""
        row["department_ko"] = ""

    elif lang_type == "korean_only":
        row["name_en"] = ""
        row["company_en"] = ""
        row["job_title_en"] = ""
        row["department_en"] = ""

    return row


def sanitize_by_visible_text(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    visible_texts = get_visible_texts(row.get("word_boxes", []))

    for field in [
        "name_ko", "name_en",
        "company_ko", "company_en",
        "job_title_ko", "job_title_en",
        "department_ko", "department_en",
        "email",
        "mobile_phone_1", "mobile_phone_2",
        "company_phone_1", "company_phone_2",
    ]:
        value = normalize_text(row.get(field, ""))
        if value and not is_value_visible(value, visible_texts):
            row[field] = ""

    return row


# =========================================================
# 4. annotations -> visible-only expanded dataset
# =========================================================
def build_visible_only_dataset(ann_dir: str) -> List[Dict[str, Any]]:
    ann_files = sorted([
        os.path.join(ann_dir, f)
        for f in os.listdir(ann_dir)
        if f.endswith(".json")
    ])

    rows = []

    for ann_path in ann_files:
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        image_rel = ann.get("image", "")
        abs_image_path = os.path.join(DATA_ROOT, image_rel)

        if not image_rel or not os.path.exists(abs_image_path):
            continue

        mobile_list, company_list = extract_phone_lists(ann)
        mobile_list = pad_phone_fields(mobile_list, max_count=2)
        company_list = pad_phone_fields(company_list, max_count=2)

        row = {
            "id": normalize_text(ann.get("id", "")),
            "image": image_rel,
            "image_path": make_relative_image_path(abs_image_path),
            "lang_type": normalize_text(ann.get("lang_type", "")),
            "difficulty_type": normalize_text(ann.get("difficulty_type", "")),
            "effects": ann.get("effects", []),

            "name_ko": normalize_text(ann.get("name_ko", "")),
            "name_en": normalize_text(ann.get("name_en", "")),
            "company_ko": normalize_text(ann.get("company_ko", "")),
            "company_en": normalize_text(ann.get("company_en", "")),
            "job_title_ko": normalize_text(ann.get("job_title_ko", "")),
            "job_title_en": normalize_text(ann.get("job_title_en", "")),
            "department_ko": normalize_text(ann.get("department_ko", "")),
            "department_en": normalize_text(ann.get("department_en", "")),
            "email": normalize_email(ann.get("email", "")),

            "mobile_phone_1": mobile_list[0],
            "mobile_phone_2": mobile_list[1],
            "company_phone_1": company_list[0],
            "company_phone_2": company_list[1],

            "word_boxes": ann.get("word_boxes", []),
            "char_boxes": ann.get("char_boxes", []),
        }

        row = sanitize_by_lang_type(row)
        row = sanitize_by_visible_text(row)

        rows.append(row)

    return rows


# =========================================================
# 5. 검증 / 미리보기
# =========================================================
def validate_dataset(rows: List[Dict[str, Any]]):
    print("===== VISIBLE-ONLY EXPANDED DATASET VALIDATION =====")
    print("total rows:", len(rows))

    empty_count = {field: 0 for field in TARGET_FIELDS}

    for row in rows:
        for field in TARGET_FIELDS:
            if normalize_text(row.get(field, "")) == "":
                empty_count[field] += 1

    print("\n[field empty counts]")
    for field in TARGET_FIELDS:
        print(f"{field}: {empty_count[field]} / {len(rows)}")


def preview_random_samples(rows: List[Dict[str, Any]], n: int = 5):
    picks = random.sample(rows, min(n, len(rows)))

    for i, row in enumerate(picks, start=1):
        print("\n" + "=" * 100)
        print(f"SAMPLE {i}")
        print("=" * 100)
        print("id:", row["id"])
        print("image_path:", row["image_path"])
        print("lang_type:", row["lang_type"])

        for field in TARGET_FIELDS:
            print(f"{field}: {row.get(field, '')}")

        abs_img_path = os.path.join(DATA_ROOT, row["image_path"])
        img = Image.open(abs_img_path).convert("RGB")
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(abs_img_path))
        plt.show()


def preview_specific_id(rows: List[Dict[str, Any]], target_id: str):
    found = [r for r in rows if r.get("id", "") == target_id]
    print("found count:", len(found))

    for row in found:
        print("\n" + "=" * 100)
        print("id:", row["id"])
        print("image_path:", row["image_path"])
        print("lang_type:", row["lang_type"])
        for field in TARGET_FIELDS:
            print(f"{field}: {row.get(field, '')}")

        abs_img_path = os.path.join(DATA_ROOT, row["image_path"])
        img = Image.open(abs_img_path).convert("RGB")
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(abs_img_path))
        plt.show()


# =========================================================
# 6. 학습용 chat format
# =========================================================
def convert_to_chat_format(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset = []

    for row in rows:
        assistant_json = {
            "name_ko": row["name_ko"],
            "name_en": row["name_en"],
            "company_ko": row["company_ko"],
            "company_en": row["company_en"],
            "job_title_ko": row["job_title_ko"],
            "job_title_en": row["job_title_en"],
            "department_ko": row["department_ko"],
            "department_en": row["department_en"],
            "email": row["email"],
            "mobile_phone_1": row["mobile_phone_1"],
            "mobile_phone_2": row["mobile_phone_2"],
            "company_phone_1": row["company_phone_1"],
            "company_phone_2": row["company_phone_2"],
        }

        dataset.append({
            "image_path": row["image_path"],
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Look at this image and extract the business card information using OCR. "
                        "Read only the text that is clearly and actually visible in the image. "
                        "If any value is missing, blurred, occluded, partially visible, or uncertain, "
                        "do not guess and output an empty string instead. "
                        "Do not translate, infer, paraphrase, or complete values. "
                        "If multiple phone numbers are visible, extract up to two mobile phone numbers and up to two company phone numbers in reading order from top to bottom. "
                        "Return only one JSON object with the following schema: "
                        '{"name_ko":"","name_en":"","company_ko":"","company_en":"","job_title_ko":"","job_title_en":"","department_ko":"","department_en":"","email":"","mobile_phone_1":"","mobile_phone_2":"","company_phone_1":"","company_phone_2":""}'
                    )
                },
                {
                    "role": "assistant",
                    "content": json.dumps(assistant_json, ensure_ascii=False)
                }
            ]
        })

    return dataset


# =========================================================
# 7. subset
# =========================================================
def sample_subset(rows: List[Dict[str, Any]], subset_size: int = 300, seed: int = 42):
    random.seed(seed)
    if len(rows) <= subset_size:
        return rows[:]
    return random.sample(rows, subset_size)


# =========================================================
# 8. 실행
# =========================================================
def main():
    rows = build_visible_only_dataset(ANN_DIR)

    validate_dataset(rows)

    all_rows_path = os.path.join(OUTPUT_DIR, "dataset_visible_only_phone_expanded_all.json")
    save_json(rows, all_rows_path)
    print(f"\n[저장 완료] {all_rows_path}")

    subset_300 = sample_subset(rows, subset_size=SUBSET_SIZE, seed=SEED)
    subset_300_path = os.path.join(OUTPUT_DIR, "dataset_visible_only_phone_expanded_subset_300.json")
    save_json(subset_300, subset_300_path)
    print(f"[저장 완료] {subset_300_path}")

    train_chat_300 = convert_to_chat_format(subset_300)
    train_chat_300_path = os.path.join(OUTPUT_DIR, "train_chat_300_visible_only_phone_expanded.json")
    save_json(train_chat_300, train_chat_300_path)
    print(f"[저장 완료] {train_chat_300_path}")

    preview_random_samples(rows, n=3)
    preview_specific_id(rows, "card_000228")


if __name__ == "__main__":
    main()