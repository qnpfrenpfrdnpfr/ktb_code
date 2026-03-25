"""
명함 데이터 생성기 (test.py 기반 개선판)

요구사항 반영:
1) 영어/한글/혼합 명함 비율 1:1:1
2) T/Tel/TEL 접두어 -> company_phone, M/Mobile 접두어 -> mobile_phone
3) 라벨에는 접두어를 제거한 "번호 문자열만" 저장
"""

import argparse
import json
import os
import random
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
CARD_W, CARD_H = 856, 540

BG_COLORS = [
    (255, 255, 255), (247, 247, 247), (240, 248, 255), (255, 250, 240),
    (28, 28, 28), (20, 38, 72), (64, 18, 18),
]
TEXT_COLORS_LIGHT = [(20, 20, 20), (35, 35, 70), (75, 20, 20)]
TEXT_COLORS_DARK = [(235, 235, 235), (200, 220, 255), (255, 220, 200)]

COMPANY_PREFIXES = ["T", "T.", "Tel", "Tel.", "TEL", "Phone"]
MOBILE_PREFIXES = ["M", "M.", "Mobile", "Mobile.", "HP", "H.P."]

SEPARATORS = ["-", ".", " "]

KOR_COMPANIES = [
    "삼성전자", "네이버", "카카오", "현대자동차", "LG CNS", "SK텔레콤",
    "배달의민족", "우아한형제들", "성심당", "한빛테크", "미래소프트",
]
ENG_COMPANIES = [
    "Samsung Electronics", "NAVER Corp.", "Kakao Corp.", "Hyundai Motor Company",
    "LG CNS", "SK Telecom", "Baemin", "Woowa Brothers", "BluePeak Tech", "NovaWorks",
]

KOR_SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
KOR_GIVEN = ["민준", "서연", "지호", "수빈", "예진", "현우", "도현", "지수"]
ENG_FIRST = ["James", "Emily", "Daniel", "Sophia", "Ryan", "Olivia", "Noah", "Emma"]
ENG_LAST = ["Kim", "Lee", "Park", "Choi", "Smith", "Brown", "Johnson"]

KOR_TITLES = ["대표", "이사", "부장", "차장", "과장", "대리", "연구원", "팀장"]
ENG_TITLES = ["CEO", "Director", "Manager", "Engineer", "Researcher", "Team Lead"]
DEPARTMENTS = ["플랫폼팀", "개발팀", "연구소", "AI팀", "R&D", "Business Development", "Security Team", ""]
DOMAINS = ["gmail.com", "company.co.kr", "corp.com", "naver.com", "biz.kr"]

KOR_ADDRESSES = [
    "서울특별시 강남구 테헤란로 123",
    "경기도 성남시 분당구 판교역로 235",
    "부산광역시 해운대구 센텀서로 45",
]
ENG_ADDRESSES = [
    "123 Teheran-ro, Gangnam-gu, Seoul",
    "235 Pangyoyeok-ro, Bundang-gu, Seongnam-si",
    "45 Centumseo-ro, Haeundae-gu, Busan",
]
MIX_ADDRESSES = [
    "서울특별시 강남구 Teheran-ro 123",
    "Busan 해운대구 Centumseo-ro 45",
    "Seongnam 분당구 판교역로 235",
]


# ─────────────────────────────────────────────
# 1. 폰트
# ─────────────────────────────────────────────
@lru_cache(maxsize=256)
def _load_font(path: str, size: int):
    return ImageFont.truetype(path, size)


def pick_font(size: int):
    candidates = [
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return _load_font(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ─────────────────────────────────────────────
# 2. 유틸
# ─────────────────────────────────────────────
def has_korean(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


def rand_kor_name() -> str:
    return random.choice(KOR_SURNAMES) + random.choice(KOR_GIVEN)


def rand_eng_name() -> str:
    return f"{random.choice(ENG_FIRST)} {random.choice(ENG_LAST)}"


def rand_email(name: str) -> str:
    seed = re.sub(r"[^a-zA-Z]", "", name).lower() or "contact"
    return f"{seed[:12]}{random.randint(1,99)}@{random.choice(DOMAINS)}"


def build_balanced_types(n_samples: int) -> list[str]:
    base = ["KOR", "ENG", "MIX"]
    q, r = divmod(n_samples, 3)
    out = base * q + base[:r]
    random.shuffle(out)
    return out


# ─────────────────────────────────────────────
# 3. 전화번호 생성/분리
# ─────────────────────────────────────────────
def gen_company_number() -> str:
    area = random.choice(["02", "031", "032", "051", "053", "062"])
    sep = random.choice(SEPARATORS)
    if area == "02":
        mid = str(random.randint(100, 9999))
        last = str(random.randint(1000, 9999))
    else:
        mid = str(random.randint(100, 9999))
        last = str(random.randint(1000, 9999))
    return f"{area}{sep}{mid}{sep}{last}"


def gen_mobile_number() -> str:
    sep = random.choice(SEPARATORS)
    return f"010{sep}{random.randint(1000, 9999)}{sep}{random.randint(1000, 9999)}"


def add_prefix(number: str, kind: str) -> str:
    if kind == "company":
        pref = random.choice(COMPANY_PREFIXES)
    else:
        pref = random.choice(MOBILE_PREFIXES)
    joiner = random.choice([" ", ". ", ": ", " : "])
    return f"{pref}{joiner}{number}"


def extract_phone_from_line(line: str) -> tuple[str, str]:
    """
    line에서 접두어를 보고 (kind, number_only_text) 반환.
    kind: company | mobile
    number_only_text: 접두어 제거한 번호 문자열 (정규화 최소화)
    """
    s = line.strip()
    m = re.match(r"^\s*([A-Za-z\.]+)\s*[:.]?\s*(.+)$", s)
    if m:
        pref = m.group(1).lower().rstrip(".")
        rest = m.group(2).strip()
    else:
        pref = ""
        rest = s

    if pref in {"t", "tel", "phone"}:
        kind = "company"
    elif pref in {"m", "mobile", "hp", "h.p"}:
        kind = "mobile"
    else:
        digits = re.sub(r"\D", "", rest)
        kind = "mobile" if digits.startswith(("010", "011", "016", "017", "018", "019")) else "company"

    # 번호에 해당하는 문자만 남김 (접두어/기타 텍스트 제거)
    keep = re.search(r"[0-9][0-9\-\.\s()]*", rest)
    number = keep.group(0).strip() if keep else rest
    number = re.sub(r"\s+", " ", number).strip()

    return kind, number


# ─────────────────────────────────────────────
# 4. 샘플 생성
# ─────────────────────────────────────────────
def sample_card_fields(card_type: str) -> dict:
    if card_type == "KOR":
        company = random.choice(KOR_COMPANIES)
        name = rand_kor_name()
        title = random.choice(KOR_TITLES)
        address = random.choice(KOR_ADDRESSES)
    elif card_type == "ENG":
        company = random.choice(ENG_COMPANIES)
        name = rand_eng_name()
        title = random.choice(ENG_TITLES)
        address = random.choice(ENG_ADDRESSES)
    else:
        company = random.choice(KOR_COMPANIES + ENG_COMPANIES)
        name = random.choice([rand_kor_name(), rand_eng_name()])
        title = random.choice(KOR_TITLES + ENG_TITLES)
        address = random.choice(MIX_ADDRESSES)

    dept = random.choice(DEPARTMENTS)
    email = rand_email(name)

    mode = random.choice(["company_only", "mobile_only", "both"])
    lines = []
    if mode in {"company_only", "both"}:
        lines.append(add_prefix(gen_company_number(), "company"))
    if mode in {"mobile_only", "both"}:
        lines.append(add_prefix(gen_mobile_number(), "mobile"))

    company_phone = ""
    mobile_phone = ""
    for ln in lines:
        kind, number = extract_phone_from_line(ln)
        if kind == "company" and not company_phone:
            company_phone = number
        elif kind == "mobile" and not mobile_phone:
            mobile_phone = number

    return {
        "card_type": card_type,
        "company": company,
        "name": name,
        "job_title": title,
        "department": dept,
        "email": email,
        "address": address,
        "phone_lines_rendered": lines,
        "company_phone": company_phone,
        "mobile_phone": mobile_phone,
    }


# ─────────────────────────────────────────────
# 5. 이미지 생성/증강
# ─────────────────────────────────────────────
def draw_business_card(fields: dict) -> Image.Image:
    bg = random.choice(BG_COLORS)
    img = Image.new("RGB", (CARD_W, CARD_H), bg)
    draw = ImageDraw.Draw(img)

    dark_bg = sum(bg) < 380
    tc = random.choice(TEXT_COLORS_DARK if dark_bg else TEXT_COLORS_LIGHT)
    accent = (random.randint(0, 200), random.randint(0, 120), random.randint(100, 255))

    f_company = pick_font(random.randint(28, 36))
    f_name = pick_font(random.randint(30, 40))
    f_title = pick_font(random.randint(20, 26))
    f_info = pick_font(random.randint(18, 24))

    draw.text((50, 50), fields["company"], font=f_company, fill=accent)
    draw.line([(50, 105), (CARD_W - 50, 105)], fill=tc, width=1)
    draw.text((50, 125), fields["name"], font=f_name, fill=tc)
    draw.text((50, 175), fields["job_title"], font=f_title, fill=tc)
    draw.text((50, 210), fields["department"], font=f_info, fill=tc)

    y = 260
    for line in fields["phone_lines_rendered"]:
        draw.text((50, y), line, font=f_info, fill=tc)
        y += 34
    draw.text((50, y), fields["email"], font=f_info, fill=tc)
    y += 34
    draw.text((50, y), fields["address"], font=f_info, fill=tc)
    return img


def random_augment_config() -> dict:
    return {
        "blur": random.choice([0, 0.5, 1.0, 1.5, 2.0]),
        "noise": random.choice([0, 3, 7, 12, 20]),
        "shadow": random.random() < 0.35,
        "rotation": random.choice([0, 0, random.uniform(-8, 8)]),
        "perspective": random.random() < 0.25,
        "jpeg_quality": random.choice([100, 95, 85, 75, 60]),
        "distance": random.random() < 0.30,
    }


def augment_image(img: Image.Image, config: dict) -> Image.Image:
    arr = np.array(img).astype(np.float32)

    if config["noise"] > 0:
        noise = np.random.normal(0, config["noise"], arr.shape)
        arr = np.clip(arr + noise, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))

    if config["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=config["blur"]))

    if config["shadow"]:
        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        side = random.choice(["left", "right", "top", "bottom"])
        alpha_max = random.randint(60, 130)
        w, h = img.size
        steps = 80
        for i in range(steps):
            a = int(alpha_max * (1 - i / steps))
            if side == "left":
                sd.rectangle([(0, 0), (w * i // steps, h)], fill=(0, 0, 0, a))
            elif side == "right":
                sd.rectangle([(w - w * i // steps, 0), (w, h)], fill=(0, 0, 0, a))
            elif side == "top":
                sd.rectangle([(0, 0), (w, h * i // steps)], fill=(0, 0, 0, a))
            else:
                sd.rectangle([(0, h - h * i // steps), (w, h)], fill=(0, 0, 0, a))
        img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")

    if config["perspective"]:
        arr_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = arr_cv.shape[:2]
        jitter = lambda: random.randint(-30, 30)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[jitter(), jitter()], [w + jitter(), jitter()], [w + jitter(), h + jitter()], [jitter(), h + jitter()]])
        M = cv2.getPerspectiveTransform(src, dst)
        arr_cv = cv2.warpPerspective(arr_cv, M, (w, h), borderValue=(255, 255, 255))
        img = Image.fromarray(cv2.cvtColor(arr_cv, cv2.COLOR_BGR2RGB))

    if abs(config["rotation"]) > 0.1:
        img = img.rotate(config["rotation"], expand=False, fillcolor=(255, 255, 255))

    if config["distance"]:
        arr = np.array(img)
        h, w = arr.shape[:2]
        scale = random.uniform(0.62, 0.9)
        nw, nh = int(w * scale), int(h * scale)
        small = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full_like(arr, 245)
        ox = random.randint(0, w - nw)
        oy = random.randint(0, h - nh)
        canvas[oy:oy + nh, ox:ox + nw] = small
        img = Image.fromarray(canvas)

    if config["jpeg_quality"] < 95:
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=config["jpeg_quality"])
        buf.seek(0)
        img = Image.open(buf).copy()

    return img


# ─────────────────────────────────────────────
# 6. 데이터셋 생성
# ─────────────────────────────────────────────
def generate_dataset(output_dir: Path, n_samples: int = 600, augment: bool = True):
    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    card_types = build_balanced_types(n_samples)
    labels = []

    for idx, card_type in enumerate(card_types):
        fields = sample_card_fields(card_type)
        img = draw_business_card(fields)
        aug_cfg = {}
        if augment:
            aug_cfg = random_augment_config()
            img = augment_image(img, aug_cfg)

        filename = f"card_{idx:05d}.jpg"
        img.save(image_dir / filename, "JPEG", quality=95)

        labels.append({
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\n이 이미지가 명함인지 판단하고 명함이라면 정보를 JSON으로 추출해줘.",
                },
                {
                    "role": "assistant",
                    "content": {
                        "is_business_card": True,
                        "name": fields["name"],
                        "company": fields["company"],
                        "job_title": fields["job_title"],
                        "department": fields["department"],
                        "email": fields["email"],
                        "company_phone": fields["company_phone"],
                        "mobile_phone": fields["mobile_phone"],
                    },
                },
            ],
            "images": [filename],
            "augmentation": aug_cfg,
            # 상위 키는 기존 호환용으로 비워둠
            "name": "",
            "company": "",
            "job_title": "",
            "department": "",
            "email": "",
            "company_phone": "",
            "mobile_phone": "",
        })

    with open(label_dir / "labels_train.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"[DONE] images={image_dir} labels={label_dir / 'labels_train.json'}")
    print(f"[STATS] KOR={card_types.count('KOR')} ENG={card_types.count('ENG')} MIX={card_types.count('MIX')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="card_data", help="output root dir")
    parser.add_argument("--n", type=int, default=600, help="number of samples")
    parser.add_argument("--no-augment", action="store_true", help="disable visual augmentation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out).expanduser().resolve()
    generate_dataset(out_dir, n_samples=args.n, augment=not args.no_augment)


if __name__ == "__main__":
    main()
