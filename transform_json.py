import json
import re

INPUT_PATH = "/Users/jisu/Documents/coding_test/business_card_dataset/labels/labels.json"
OUTPUT_PATH = "/Users/jisu/Documents/coding_test/business_card_dataset/labels/llama_factory_dataset.json"


def clean_phone(phone):
    if not phone:
        return ""
    return re.sub(r"[^0-9\-+ ]", "", phone)


def convert_record(record):

    phone_mode = record.get("phone_mode", "")

    raw_company_phone = record.get("company_phone", "")
    raw_mobile_phone = record.get("mobile_phone", "")
    raw_phone = record.get("phone", "")  # backward compat

    if phone_mode == "mobile_only":
        company_phone = clean_phone(raw_company_phone) if raw_company_phone else ""
        mobile_phone = clean_phone(raw_mobile_phone or raw_phone)
    elif phone_mode == "company_only":
        company_phone = clean_phone(raw_company_phone or raw_phone)
        mobile_phone = clean_phone(raw_mobile_phone) if raw_mobile_phone else ""
    elif phone_mode == "both":
        company_phone = clean_phone(raw_company_phone)
        mobile_phone = clean_phone(raw_mobile_phone)
    else:
        # Legacy/unknown schema: keep previous behavior.
        company_phone = clean_phone(raw_company_phone or raw_phone)
        mobile_phone = clean_phone(raw_mobile_phone)

    assistant_output = {
        "is_business_card": True,
        "name": record.get("name", ""),
        "company": record.get("company_display", ""),
        "job_title": record.get("title", ""),
        "department": "",
        "email": record.get("email", ""),
        "company_phone": company_phone,
        "mobile_phone": mobile_phone,
    }

    new_item = {
        "messages": [
            {
                "role": "user",
                "content": "<image>\n이 이미지가 명함인지 판단하고 명함이라면 정보를 JSON으로 추출해줘."
            },
            {
                "role": "assistant",
                "content": assistant_output
            }
        ],
        "images": [record.get("file")],

        # augmentation 그대로 복사
        "augmentation": record.get("augmentation", {})
    }

    return new_item


def main():

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = [convert_record(record) for record in data]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
