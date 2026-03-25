import json
import re


INPUT_FILE = "LLaMA-Factory/data/business_card_dataset/labels/labels_train.json"
OUTPUT_FILE = "LLaMA-Factory/data/business_card_dataset/labels/labels_train.json"


def normalize_phone(phone):
    if not phone:
        return ""

    digits = re.sub(r"\D", "", str(phone))
    if not digits:
        return ""

    # +82 / 0082 국제번호를 국내 표기로 변환
    if digits.startswith("0082"):
        digits = digits[4:]
        if not digits.startswith("0"):
            digits = "0" + digits
    elif digits.startswith("82"):
        digits = digits[2:]
        if not digits.startswith("0"):
            digits = "0" + digits

    # 서울 지역번호(02)
    if digits.startswith("02"):
        if len(digits) == 10:
            return f"02-{digits[2:6]}-{digits[6:]}"
        if len(digits) == 9:
            return f"02-{digits[2:5]}-{digits[5:]}"

    # 휴대폰
    if digits.startswith(("010", "011", "016", "017", "018", "019")):
        if len(digits) == 11:
            return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
        if len(digits) == 10:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"

    if len(digits) == 11:
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"

    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"

    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:]}"

    return digits


def normalize_email(email):
    if not email:
        return ""

    return email.strip().lower()


def normalize_company(company):
    if not company:
        return ""

    company = str(company).strip()

    patterns = [
        r"\(\s*주\s*\)",
        r"㈜",
        r"\(\s*유\s*\)",
        r"주식회사",
        r"유한회사",
        r"Co\.,?\s*Ltd\.?",
        r"Ltd\.?",
        r"Corp\.?",
        r"Corporation",
        r"Inc\.?",
        r"LLC",
        r"Group",
        r"Holdings?",
        r"그룹",
        r"홀딩스",
    ]

    for p in patterns:
        company = re.sub(p, " ", company, flags=re.IGNORECASE)

    company = re.sub(r"[()\[\]]", " ", company)
    company = re.sub(r"\s+", " ", company)

    return company.strip(" .,-")


def normalize_text(text):
    if not text:
        return ""

    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_card_fields(card):

    if not isinstance(card, dict):
        return card

    card["name"] = normalize_text(card.get("name", ""))
    card["company"] = normalize_company(card.get("company", ""))
    card["job_title"] = normalize_text(card.get("job_title", ""))
    card["department"] = normalize_text(card.get("department", ""))
    card["email"] = normalize_email(card.get("email", ""))
    card["company_phone"] = normalize_phone(card.get("company_phone", ""))
    card["mobile_phone"] = normalize_phone(card.get("mobile_phone", ""))
    return card


def normalize_record(rec):
    # 실제 라벨 값이 들어있는 assistant content를 정규화
    for msg in rec.get("messages", []):
        if msg.get("role") == "assistant":
            msg["content"] = normalize_card_fields(msg.get("content", {}))

    # 상위 키가 있을 경우에도 동일 규칙 적용
    rec["name"] = normalize_text(rec.get("name", ""))
    rec["company"] = normalize_company(rec.get("company", ""))
    rec["job_title"] = normalize_text(rec.get("job_title", ""))
    rec["department"] = normalize_text(rec.get("department", ""))
    rec["email"] = normalize_email(rec.get("email", ""))
    rec["company_phone"] = normalize_phone(rec.get("company_phone", ""))
    rec["mobile_phone"] = normalize_phone(rec.get("mobile_phone", ""))

    return rec


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = []

    for rec in data:
        normalized.append(normalize_record(rec))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"완료: {OUTPUT_FILE} 생성 ({len(normalized)} records)")


if __name__ == "__main__":
    main()
