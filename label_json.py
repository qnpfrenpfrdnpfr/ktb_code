import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT = "/Users/jisu/Documents/coding_test/LLaMA-Factory/data/business_card_dataset/labels/labels_train.json"
KEEP_FIELDS = ["name", "job_title", "department", "email", "phone_digits", "company_phone"]


def _as_text(value):
    if value is None:
        return ""
    return str(value).strip()


def _digits_only(value):
    return re.sub(r"\D", "", _as_text(value))


def _format_phone_like_kor(digits):
    if len(digits) == 11 and digits.startswith("010"):
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    if len(digits) == 10 and digits.startswith("02"):
        return f"02-{digits[2:6]}-{digits[6:]}"
    if len(digits) == 9 and digits.startswith("02"):
        return f"02-{digits[2:5]}-{digits[5:]}"
    if len(digits) == 11 and digits.startswith("0"):
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    if len(digits) == 10 and digits.startswith("0"):
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:]}"
    return digits


def _extract_source(record):
    if not isinstance(record, dict):
        return {}

    messages = record.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and isinstance(msg.get("content"), dict)
            ):
                return msg["content"]

    return record


def filter_record(record):
    src = _extract_source(record)

    phone_kind = _as_text(src.get("phone_kind")).lower()
    company_phone_src = _as_text(src.get("company_phone"))
    mobile_phone_src = _as_text(src.get("mobile_phone"))
    phone_digits_src = _as_text(src.get("phone_digits") or src.get("phon_digits"))

    digits_fallback = _digits_only(phone_digits_src or mobile_phone_src or company_phone_src)

    if phone_kind == "company":
        company_phone = company_phone_src or _format_phone_like_kor(digits_fallback)
        phone_digits = ""
    elif phone_kind == "mobile":
        company_phone = ""
        phone_digits = _digits_only(phone_digits_src or mobile_phone_src or company_phone_src)
    else:
        # 축약 라벨(phone_kind 없음) 대응:
        # company_phone가 있으면 회사번호로 확정, 없으면 phone_digits만 유지
        if company_phone_src:
            company_phone = company_phone_src
            phone_digits = ""
        else:
            company_phone = ""
            phone_digits = _digits_only(phone_digits_src or mobile_phone_src)

    return {
        "name": _as_text(src.get("name")),
        "job_title": _as_text(src.get("job_title") or src.get("title")),
        "department": _as_text(src.get("department")),
        "email": _as_text(src.get("email")),
        "phone_digits": phone_digits,
        "company_phone": company_phone,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="input labels json path")
    parser.add_argument("--output", default=None, help="output json path (default: overwrite input)")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    filtered = [filter_record(rec) for rec in data]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"done: {output_path} ({len(filtered)} records)")
    print(f"fields: {', '.join(KEEP_FIELDS)}")


if __name__ == "__main__":
    main()
