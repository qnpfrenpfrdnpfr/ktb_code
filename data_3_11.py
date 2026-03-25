import os
import re
import io
import json
import math
import random
import zipfile
import requests
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# =========================================================
# 설정
# =========================================================

OUTPUT_DIR = Path("./dataset_3_10")
IMAGE_DIR = OUTPUT_DIR / "images"
ANNOTATION_DIR = OUTPUT_DIR / "annotations"
LABEL_PATH = OUTPUT_DIR / "labels.jsonl"

NUM_SAMPLES = 1000
SEED = 42

DART_API_KEY = os.environ.get("DART_API_KEY", "").strip()

LANG_TYPE_WEIGHTS = {
    "english_only": 0.25,
    "korean_only": 0.25,
    "mixed": 0.50,
}

DIFFICULTY_WEIGHTS = {
    "clean": 0.25,
    "hard": 0.45,
    "confusing": 0.15,
    "challenging": 0.15,
}

HARD_EFFECT_WEIGHTS = {
    "blur": 0.24,              # blur를 조금 더 높임
    "rotation": 0.10,
    "perspective": 0.10,
    "shadow": 0.09,
    "occlusion": 0.08,
    "low_light": 0.08,
    "glare": 0.08,
    "background_clutter": 0.08,
    "handheld": 0.08,
    "noise": 0.07,
}

CARD_WIDTH = 1000
CARD_HEIGHT = 600
CANVAS_W = 1400
CANVAS_H = 1000

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

# =========================================================
# 기본 사전
# =========================================================

KOREAN_LAST_NAMES = [
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "전", "홍"
]

KOREAN_FIRST_NAMES = [
    "민준", "서준", "도윤", "예준", "시우", "하준", "지호", "주원", "지후", "준우",
    "서연", "서윤", "지우", "하은", "하윤", "민서", "지민", "채원", "수아", "유진",
    "지수", "정민", "가영", "현우", "지훈", "수빈", "예린", "다은", "하람", "유나",
    "지안", "민지", "소연", "준혁", "태현", "시연", "지현", "은지", "도연", "지은"
]

ENGLISH_FIRST_NAMES = [
    "Jisu", "Minjun", "Seojun", "Doyoon", "Yejun", "Siwoo", "Hajun", "Jiho",
    "Jiwon", "Jimin", "Sujin", "Soojin", "Eunji", "Yuna", "Minji", "Hyunwoo",
    "Daniel", "Sophia", "Olivia", "James", "Emily", "William", "Grace", "Chloe",
    "Noah", "Mason", "Liam", "Ella", "Ava", "Lucas", "Henry", "Leo"
]

ENGLISH_LAST_NAMES = [
    "Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon", "Jang", "Lim",
    "Han", "Oh", "Seo", "Shin", "Kwon", "Hwang", "Ahn", "Song", "Jeon", "Hong",
    "Smith", "Johnson", "Brown", "Taylor", "Anderson", "Thomas", "Martin"
]

JOB_TITLE_KO = [
    "사원", "주임", "대리", "과장", "차장", "부장", "이사", "상무", "전무", "대표",
    "매니저", "책임연구원", "선임연구원", "연구원", "팀장", "실장", "센터장", "파트장"
]

JOB_TITLE_EN = [
    "Staff", "Associate", "Assistant Manager", "Manager", "Senior Manager",
    "Deputy General Manager", "General Manager", "Director", "Executive Director",
    "Vice President", "CEO", "Lead Researcher", "Researcher", "Team Lead", "Head"
]

DEPARTMENT_KO = [
    "플랫폼팀", "개발팀", "AI연구소", "디자인팀", "마케팅팀", "인사팀", "전략기획팀",
    "영업팀", "고객성공팀", "데이터팀", "보안팀", "R&D센터", "사업개발팀", "서비스기획팀"
]

DEPARTMENT_EN = [
    "Platform Team", "Development Team", "AI Lab", "Design Team", "Marketing Team",
    "HR Team", "Strategy Planning Team", "Sales Team", "Customer Success Team",
    "Data Team", "Security Team", "R&D Center", "Business Development Team",
    "Service Planning Team"
]

SLOGAN_KO = [
    "미래를 연결합니다",
    "기술로 가치를 만듭니다",
    "사람과 비즈니스를 잇다",
    "신뢰를 디자인합니다",
    "더 나은 내일을 위한 혁신"
]

SLOGAN_EN = [
    "Connecting the Future",
    "Technology Creates Value",
    "People, Business, Beyond",
    "Designed for Trust",
    "Innovation for Tomorrow"
]

ADDRESS_KO = [
    "서울특별시 강남구 테헤란로 123",
    "서울특별시 서초구 반포대로 45",
    "경기도 성남시 분당구 판교역로 235",
    "부산광역시 해운대구 센텀중앙로 97",
    "대전광역시 유성구 대학로 99"
]

ADDRESS_EN = [
    "123 Teheran-ro, Gangnam-gu, Seoul",
    "45 Banpo-daero, Seocho-gu, Seoul",
    "235 Pangyoyeok-ro, Bundang-gu, Seongnam-si",
    "97 Centum Jungang-ro, Haeundae-gu, Busan",
    "99 Daehak-ro, Yuseong-gu, Daejeon"
]

PHONE_LABELS_MIXED = [
    ["T", "M"],
    ["Office", "Mobile"],
    ["Direct", "Fax"],
    ["Tel", "HP"],
    ["T", "Direct", "Fax"],
    ["Office", "Mobile", "Fax"],
]

FALLBACK_COMPANY_EN = [
    "Asterix Labs", "Bluewave Systems", "Novalink Partners", "Vertex AI",
    "Northline Tech", "Pixelbridge Studio", "Brighton Dynamics", "Nexton Works",
    "Lunaris Data", "Primecore Solutions", "Horizon Cloud", "Altair Robotics"
]

FALLBACK_COMPANY_KO = [
    "한빛테크", "넥스트에이아이", "미래데이터", "블루웨이브", "프라임솔루션",
    "에이펙스랩", "하이브릿지", "루나리스", "온리원시스템", "비전네트웍스"
]

# =========================================================
# 유틸
# =========================================================

def ensure_dirs():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

def weighted_choice(weight_dict: Dict[str, float]) -> str:
    keys = list(weight_dict.keys())
    weights = list(weight_dict.values())
    return random.choices(keys, weights=weights, k=1)[0]

def maybe(prob: float) -> bool:
    return random.random() < prob

def safe_str(x) -> str:
    return "" if x is None else str(x).strip()

def normalize_company_name(name: str) -> str:
    name = safe_str(name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

def load_font(size: int, bold_preferred: bool = False):
    candidates = FONT_CANDIDATES[:]
    if bold_preferred:
        candidates = sorted(candidates, key=lambda x: 0 if ("Bold" in x or "bold" in x) else 1)

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()

def pick_text_color(bg_rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    brightness = sum(bg_rgb) / 3
    return (20, 20, 20) if brightness > 140 else (245, 245, 245)

def random_color(bright=False, dark=False):
    if bright:
        return tuple(random.randint(170, 255) for _ in range(3))
    if dark:
        return tuple(random.randint(0, 100) for _ in range(3))
    return tuple(random.randint(40, 230) for _ in range(3))

def rect_from_quad(quad: List[List[float]]) -> List[float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [min(xs), min(ys), max(xs), max(ys)]

def quad_from_rect(rect: List[float]) -> List[List[float]]:
    x1, y1, x2, y2 = rect
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def shift_quad(quad: List[List[float]], dx: float, dy: float) -> List[List[float]]:
    return [[x + dx, y + dy] for x, y in quad]

def rotate_point(x: float, y: float, cx: float, cy: float, rad: float) -> Tuple[float, float]:
    tx = x - cx
    ty = y - cy
    rx = tx * math.cos(rad) - ty * math.sin(rad)
    ry = tx * math.sin(rad) + ty * math.cos(rad)
    return rx + cx, ry + cy

def rotate_quad_expand(
    quad: List[List[float]],
    old_w: int,
    old_h: int,
    angle_deg: float,
    new_w: int,
    new_h: int
) -> List[List[float]]:
    rad = math.radians(angle_deg)
    old_cx, old_cy = old_w / 2.0, old_h / 2.0
    new_cx, new_cy = new_w / 2.0, new_h / 2.0

    corners = [[0, 0], [old_w, 0], [old_w, old_h], [0, old_h]]
    rot_corners = [rotate_point(x, y, old_cx, old_cy, rad) for x, y in corners]
    min_x = min(p[0] for p in rot_corners)
    min_y = min(p[1] for p in rot_corners)

    out = []
    for x, y in quad:
        rx, ry = rotate_point(x, y, old_cx, old_cy, rad)
        rx -= min_x
        ry -= min_y
        out.append([rx, ry])

    return out

def clip_quad(quad: List[List[float]], w: int, h: int) -> List[List[float]]:
    out = []
    for x, y in quad:
        out.append([max(0, min(w - 1, x)), max(0, min(h - 1, y))])
    return out

def update_rect_from_quad(box_item: Dict):
    box_item["bbox"] = rect_from_quad(box_item["quad"])

def apply_shift_to_boxes(boxes: List[Dict], dx: float, dy: float):
    for b in boxes:
        b["quad"] = shift_quad(b["quad"], dx, dy)
        update_rect_from_quad(b)

def apply_scale_to_boxes(boxes: List[Dict], sx: float, sy: float):
    for b in boxes:
        b["quad"] = [[x * sx, y * sy] for x, y in b["quad"]]
        update_rect_from_quad(b)

def apply_rotation_to_boxes(boxes: List[Dict], old_w: int, old_h: int, angle_deg: float, new_w: int, new_h: int):
    for b in boxes:
        b["quad"] = rotate_quad_expand(b["quad"], old_w, old_h, angle_deg, new_w, new_h)
        b["quad"] = clip_quad(b["quad"], new_w, new_h)
        update_rect_from_quad(b)

def draw_multiline_with_boxes(
    draw,
    x: int,
    y: int,
    text: str,
    font,
    fill,
    max_width: int,
    field_name: str,
    word_boxes: List[Dict],
    char_boxes: List[Dict],
    line_spacing: int = 8,
):
    lines = []
    current = ""

    for part in text.split(" "):
        candidate = part if not current else current + " " + part
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)

    cy = y
    for line in lines:
        draw_text_with_boxes(draw, x, cy, line, font, fill, field_name, word_boxes, char_boxes)
        line_bbox = draw.textbbox((x, cy), line, font=font)
        cy += (line_bbox[3] - line_bbox[1]) + line_spacing
    return cy

def draw_text_with_boxes(
    draw,
    x: int,
    y: int,
    text: str,
    font,
    fill,
    field_name: str,
    word_boxes: List[Dict],
    char_boxes: List[Dict],
):
    if not text:
        return

    draw.text((x, y), text, font=font, fill=fill)

    # char-level
    cursor_x = x
    for ch in text:
        ch_bbox = draw.textbbox((cursor_x, y), ch, font=font)
        quad = quad_from_rect([ch_bbox[0], ch_bbox[1], ch_bbox[2], ch_bbox[3]])
        char_boxes.append({
            "char": ch,
            "bbox": [ch_bbox[0], ch_bbox[1], ch_bbox[2], ch_bbox[3]],
            "quad": quad,
            "field": field_name,
            "occluded": False,
        })
        cursor_x = ch_bbox[2]

    # word-level
    for m in re.finditer(r"\S+", text):
        word = m.group()
        prefix = text[:m.start()]
        word_prefix_bbox = draw.textbbox((x, y), prefix, font=font)
        word_bbox = draw.textbbox((x + (word_prefix_bbox[2] - word_prefix_bbox[0]), y), word, font=font)
        quad = quad_from_rect([word_bbox[0], word_bbox[1], word_bbox[2], word_bbox[3]])
        word_boxes.append({
            "text": word,
            "bbox": [word_bbox[0], word_bbox[1], word_bbox[2], word_bbox[3]],
            "quad": quad,
            "field": field_name,
            "occluded": False,
        })

# =========================================================
# DART 회사명
# =========================================================

def fetch_companies_from_dart(api_key: str, limit: int = 5000) -> List[Dict[str, str]]:
    if not api_key:
        print("[WARN] DART_API_KEY가 없어 fallback 회사명을 사용합니다.")
        return []

    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    params = {"crtfc_key": api_key}

    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] DART 요청 실패: {e}")
        return []

    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        xml_name = zf.namelist()[0]
        xml_bytes = zf.read(xml_name)
        xml_text = xml_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] DART ZIP 파싱 실패: {e}")
        return []

    rows = re.findall(
        r"<list>.*?<corp_code>(.*?)</corp_code>.*?<corp_name>(.*?)</corp_name>.*?"
        r"<corp_eng_name>(.*?)</corp_eng_name>.*?<stock_code>(.*?)</stock_code>.*?"
        r"<modify_date>(.*?)</modify_date>.*?</list>",
        xml_text,
        flags=re.DOTALL
    )

    companies = []
    for corp_code, corp_name, corp_eng_name, stock_code, modify_date in rows:
        ko = normalize_company_name(corp_name)
        en = normalize_company_name(corp_eng_name)
        if not ko:
            continue

        companies.append({
            "corp_code": corp_code.strip(),
            "company_ko": ko,
            "company_en": en,
            "stock_code": stock_code.strip(),
            "modify_date": modify_date.strip(),
        })

    uniq = {}
    for c in companies:
        key = (c["company_ko"], c["company_en"])
        if key not in uniq:
            uniq[key] = c

    companies = list(uniq.values())
    random.shuffle(companies)
    return companies[:limit]

# =========================================================
# 프로필 생성
# =========================================================

def generate_korean_name() -> str:
    return random.choice(KOREAN_LAST_NAMES) + random.choice(KOREAN_FIRST_NAMES)

def generate_english_name() -> str:
    return f"{random.choice(ENGLISH_FIRST_NAMES)} {random.choice(ENGLISH_LAST_NAMES)}"

def generate_name_pair() -> Tuple[str, str]:
    return generate_korean_name(), generate_english_name()

def generate_phone_number(kind: str = "office") -> str:
    if kind == "mobile":
        return f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"

    area_codes = ["02", "031", "032", "033", "041", "042", "051", "052", "053", "054", "055", "061", "062", "063", "064"]
    area = random.choice(area_codes)
    mid = random.randint(100, 9999)
    last = random.randint(1000, 9999)
    return f"{area}-{mid}-{last}"

def generate_international_phone() -> str:
    country_codes = ["+82", "+1", "+81", "+44", "+65"]
    cc = random.choice(country_codes)
    if cc == "+82":
        return f"{cc} 10-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
    return f"{cc} {random.randint(100,999)} {random.randint(100,999)} {random.randint(1000,9999)}"

def generate_email(name_en: str, company_domain_seed: str) -> str:
    local = name_en.lower().replace(" ", ".")
    domain_seed = re.sub(r"[^a-zA-Z0-9]", "", company_domain_seed.lower())
    if not domain_seed:
        domain_seed = "company"
    suffix = random.choice(["com", "co.kr", "kr", "ai", "io"])
    return f"{local}@{domain_seed}.{suffix}"

def romanize_company_simple(ko: str) -> str:
    cleaned = re.sub(r"[^가-힣A-Za-z0-9 ]", "", ko).strip()
    return cleaned.upper() if cleaned else random.choice(FALLBACK_COMPANY_EN)

def get_company_pair(companies: List[Dict[str, str]]) -> Tuple[str, str]:
    if companies:
        c = random.choice(companies)
        ko = c["company_ko"]
        en = c["company_en"] if c["company_en"] else romanize_company_simple(ko)
        return ko, en
    return random.choice(FALLBACK_COMPANY_KO), random.choice(FALLBACK_COMPANY_EN)

def generate_profile(lang_type: str, companies: List[Dict[str, str]], difficulty_type: str) -> Dict:
    company_ko, company_en = get_company_pair(companies)
    name_ko, name_en = generate_name_pair()

    title_ko = random.choice(JOB_TITLE_KO)
    title_en = random.choice(JOB_TITLE_EN)
    dept_ko = random.choice(DEPARTMENT_KO)
    dept_en = random.choice(DEPARTMENT_EN)

    address_ko = random.choice(ADDRESS_KO)
    address_en = random.choice(ADDRESS_EN)
    slogan_ko = random.choice(SLOGAN_KO)
    slogan_en = random.choice(SLOGAN_EN)

    phones = []
    phone_count = random.randint(2, 4) if difficulty_type == "confusing" else random.randint(1, 3)
    label_set = random.choice(PHONE_LABELS_MIXED)

    for i in range(phone_count):
        label = label_set[i % len(label_set)]
        if label in ["M", "Mobile", "HP"]:
            value = generate_phone_number("mobile")
        else:
            value = generate_phone_number("office")
        phones.append({"label": label, "value": value})

    if difficulty_type == "challenging" and maybe(0.35):
        phones.append({"label": "Global", "value": generate_international_phone()})

    email = generate_email(name_en, company_en)

    profile = {
        "company_ko": company_ko,
        "company_en": company_en,
        "name_ko": name_ko,
        "name_en": name_en,
        "job_title_ko": title_ko,
        "job_title_en": title_en,
        "department_ko": dept_ko,
        "department_en": dept_en,
        "address_ko": address_ko,
        "address_en": address_en,
        "slogan_ko": slogan_ko,
        "slogan_en": slogan_en,
        "phones": phones,
        "email": email,
    }

    if lang_type == "english_only":
        profile["render_name"] = name_en
        profile["render_company"] = company_en
        profile["render_title"] = title_en
        profile["render_department"] = dept_en
        profile["render_address"] = address_en
        profile["render_slogan"] = slogan_en

    elif lang_type == "korean_only":
        profile["render_name"] = name_ko
        profile["render_company"] = company_ko
        profile["render_title"] = title_ko
        profile["render_department"] = dept_ko
        profile["render_address"] = address_ko
        profile["render_slogan"] = slogan_ko

    else:
        profile["render_name"] = f"{name_ko} | {name_en}" if maybe(0.7) else name_ko
        profile["render_company"] = f"{company_ko} | {company_en}" if maybe(0.7) else company_en
        profile["render_title"] = f"{title_ko} / {title_en}" if maybe(0.7) else title_en
        profile["render_department"] = f"{dept_ko} | {dept_en}" if maybe(0.7) else dept_ko
        profile["render_address"] = f"{address_ko}\n{address_en}" if maybe(0.5) else address_ko
        profile["render_slogan"] = f"{slogan_ko} / {slogan_en}" if maybe(0.5) else slogan_en

    profile["challenging_flags"] = {
        "big_logo_small_text": False,
        "name_only_emphasis": False,
        "abbr_title": False,
        "vertical_card": False,
        "bilingual": lang_type == "mixed",
        "double_sided_like": False,
    }

    if difficulty_type == "challenging":
        if maybe(0.35):
            profile["challenging_flags"]["big_logo_small_text"] = True
        if maybe(0.30):
            profile["challenging_flags"]["name_only_emphasis"] = True
        if maybe(0.25):
            profile["challenging_flags"]["abbr_title"] = True
            profile["render_title"] = random.choice(["Mgr.", "Dir.", "VP", "Lead", "PM", "R&D"])
        if maybe(0.20):
            profile["challenging_flags"]["vertical_card"] = True
        if maybe(0.20):
            profile["challenging_flags"]["double_sided_like"] = True

    return profile

# =========================================================
# 카드 렌더링 + bbox 수집
# =========================================================

def create_base_card(profile: Dict, difficulty_type: str):
    vertical = profile["challenging_flags"].get("vertical_card", False)
    width, height = (700, 1100) if vertical else (CARD_WIDTH, CARD_HEIGHT)

    bg = random_color(bright=True)
    fg = pick_text_color(bg)
    accent = random_color(dark=(sum(bg) > 400))

    card = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(card)

    word_boxes = []
    char_boxes = []

    if profile["challenging_flags"].get("big_logo_small_text", False):
        draw.ellipse((30, 30, 180, 180), fill=accent)
        logo_font = load_font(60, True)
        logo_letter = profile["render_company"][:1] if profile["render_company"] else "C"
        draw_text_with_boxes(draw, 60, 80, logo_letter, logo_font, fg, "logo_text", word_boxes, char_boxes)
    else:
        draw.rectangle((0, 0, width, 18), fill=accent)
        if maybe(0.4):
            draw.rectangle((0, height - 14, width, height), fill=accent)

    company_font = load_font(38, True)
    name_font = load_font(46, True)
    title_font = load_font(28)
    body_font = load_font(26)
    small_font = load_font(20)

    margin_x = 50
    y = 55
    max_w = width - 100

    if not profile["challenging_flags"].get("name_only_emphasis", False):
        y = draw_multiline_with_boxes(
            draw, margin_x, y, profile["render_company"], company_font, fg, max_w,
            "company", word_boxes, char_boxes
        )
        y += 15

    name_font_size = 62 if profile["challenging_flags"].get("name_only_emphasis", False) else 46
    name_font = load_font(name_font_size, True)
    y = draw_multiline_with_boxes(
        draw, margin_x, y, profile["render_name"], name_font, fg, max_w,
        "name", word_boxes, char_boxes
    )
    y += 10

    if maybe(0.85):
        y = draw_multiline_with_boxes(
            draw, margin_x, y, profile["render_title"], title_font, fg, max_w,
            "job_title", word_boxes, char_boxes
        )
    if maybe(0.8):
        y = draw_multiline_with_boxes(
            draw, margin_x, y + 5, profile["render_department"], title_font, fg, max_w,
            "department", word_boxes, char_boxes
        )
    y += 15

    if maybe(0.8):
        draw.line((margin_x, y, width - margin_x, y), fill=fg, width=2)
        y += 18

    for idx, ph in enumerate(profile["phones"]):
        line = f"{ph['label']}: {ph['value']}"
        y0 = y
        draw_text_with_boxes(draw, margin_x, y0, line, body_font, fg, f"phone_{idx}", word_boxes, char_boxes)
        y += 36

    email_font = small_font if difficulty_type == "confusing" and maybe(0.7) else body_font
    draw_text_with_boxes(draw, margin_x, y + 8, profile["email"], email_font, fg, "email", word_boxes, char_boxes)
    y += 52

    if difficulty_type == "confusing" and maybe(0.7):
        mixed_line = f"{profile['render_address']}   |   {profile['render_slogan']}"
        draw_multiline_with_boxes(
            draw, margin_x, y, mixed_line, small_font, fg, max_w,
            "address_or_slogan", word_boxes, char_boxes
        )
    else:
        if maybe(0.75):
            y = draw_multiline_with_boxes(
                draw, margin_x, y, profile["render_address"], small_font, fg, max_w,
                "address", word_boxes, char_boxes
            )
            y += 10
        if maybe(0.75):
            draw_multiline_with_boxes(
                draw, margin_x, y, profile["render_slogan"], small_font, fg, max_w,
                "slogan", word_boxes, char_boxes
            )

    return card, word_boxes, char_boxes

# =========================================================
# 배경 / 카드 배치
# =========================================================

def add_background_clutter(canvas: Image.Image):
    draw = ImageDraw.Draw(canvas)
    for _ in range(random.randint(20, 50)):
        shape_type = random.choice(["line", "rect", "ellipse"])
        color = tuple(random.randint(100, 220) for _ in range(3))
        if shape_type == "line":
            x1, y1 = random.randint(0, canvas.width), random.randint(0, canvas.height)
            x2, y2 = random.randint(0, canvas.width), random.randint(0, canvas.height)
            draw.line((x1, y1, x2, y2), fill=color, width=random.randint(1, 3))
        elif shape_type == "rect":
            x1, y1 = random.randint(0, canvas.width - 1), random.randint(0, canvas.height - 1)
            x2 = min(canvas.width, x1 + random.randint(20, 160))
            y2 = min(canvas.height, y1 + random.randint(20, 120))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        else:
            x1, y1 = random.randint(0, canvas.width - 1), random.randint(0, canvas.height - 1)
            x2 = min(canvas.width, x1 + random.randint(20, 160))
            y2 = min(canvas.height, y1 + random.randint(20, 120))
            draw.ellipse((x1, y1, x2, y2), outline=color, width=2)

def paste_card_on_canvas(card: Image.Image, word_boxes: List[Dict], char_boxes: List[Dict], difficulty_type: str, applied_effects: List[str]):
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), random_color(bright=True))

    if "background_clutter" in applied_effects or difficulty_type in ["hard", "challenging"]:
        if maybe(0.7):
            add_background_clutter(canvas)

    target_w = max(1, CANVAS_W - 240)
    target_h = max(1, CANVAS_H - 200)
    if card.width > target_w or card.height > target_h:
        old_w, old_h = card.size
        scale = min(target_w / old_w, target_h / old_h)
        new_w = max(1, int(old_w * scale))
        new_h = max(1, int(old_h * scale))
        card = card.resize((new_w, new_h), Image.Resampling.BICUBIC)

        sx = new_w / old_w
        sy = new_h / old_h
        apply_scale_to_boxes(word_boxes, sx, sy)
        apply_scale_to_boxes(char_boxes, sx, sy)

    def sample_axis(canvas_len: int, card_len: int, margin: int) -> int:
        free = canvas_len - card_len
        if free <= 0:
            return 0
        margin = min(margin, free // 2)
        low = margin
        high = free - margin
        if high < low:
            return free // 2
        return random.randint(low, high)

    shadow = Image.new("RGBA", card.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle((10, 10, card.width - 5, card.height - 5), radius=18, fill=(0, 0, 0, 90))

    x = sample_axis(CANVAS_W, card.width, 120)
    y = sample_axis(CANVAS_H, card.height, 100)

    base = canvas.convert("RGBA")
    base.alpha_composite(shadow, (x + 18, y + 18))
    base.alpha_composite(card.convert("RGBA"), (x, y))

    apply_shift_to_boxes(word_boxes, x, y)
    apply_shift_to_boxes(char_boxes, x, y)

    return base.convert("RGB")

# =========================================================
# 효과 적용
# =========================================================

def choose_effects(difficulty_type: str) -> List[str]:
    if difficulty_type == "clean":
        return []

    if difficulty_type == "hard":
        n = random.randint(1, 3)
    elif difficulty_type == "confusing":
        n = random.randint(1, 2)
    else:
        n = random.randint(2, 4)

    keys = list(HARD_EFFECT_WEIGHTS.keys())
    weights = list(HARD_EFFECT_WEIGHTS.values())

    effects = []
    while len(effects) < n:
        e = random.choices(keys, weights=weights, k=1)[0]
        if e not in effects:
            effects.append(e)
    return effects

def apply_blur(img: Image.Image) -> Image.Image:
    radius = random.uniform(1.2, 3.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_low_light(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.45, 0.8))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.0))
    return img

def apply_noise(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.int16)
    noise = np.random.normal(0, random.uniform(8, 22), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_shadow(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    x0 = random.randint(0, w // 2)
    y0 = random.randint(0, h // 2)
    x1 = random.randint(w // 2, w)
    y1 = random.randint(h // 2, h)

    mask = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    line = ((xx - x0) * (y1 - y0) - (yy - y0) * (x1 - x0))
    mask[line > 0] = 1.0

    alpha = random.uniform(0.18, 0.35)
    arr *= (1.0 - alpha * mask[..., None])
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_glare(img: Image.Image) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for _ in range(random.randint(1, 3)):
        x1 = random.randint(0, img.width)
        y1 = random.randint(0, img.height)
        x2 = min(img.width, x1 + random.randint(120, 420))
        y2 = min(img.height, y1 + random.randint(30, 140))
        draw.ellipse((x1, y1, x2, y2), fill=(255, 255, 255, random.randint(35, 90)))

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

def mark_occluded_boxes(boxes: List[Dict], occ_rects: List[List[int]]):
    for b in boxes:
        bx1, by1, bx2, by2 = b["bbox"]
        for ox1, oy1, ox2, oy2 in occ_rects:
            inter_x1 = max(bx1, ox1)
            inter_y1 = max(by1, oy1)
            inter_x2 = min(bx2, ox2)
            inter_y2 = min(by2, oy2)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                b["occluded"] = True
                break

def apply_occlusion(img: Image.Image, word_boxes: List[Dict], char_boxes: List[Dict]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    occ_rects = []

    for _ in range(random.randint(1, 2)):
        x1 = random.randint(0, img.width - 120)
        y1 = random.randint(0, img.height - 80)
        x2 = x1 + random.randint(80, 220)
        y2 = y1 + random.randint(40, 160)
        color = random.choice([
            (180, 140, 120),
            (160, 120, 100),
            (220, 220, 220),
            (80, 80, 80),
        ])
        draw.rounded_rectangle((x1, y1, x2, y2), radius=20, fill=color)
        occ_rects.append([x1, y1, x2, y2])

    mark_occluded_boxes(word_boxes, occ_rects)
    mark_occluded_boxes(char_boxes, occ_rects)
    return out

def apply_rotation(img: Image.Image, word_boxes: List[Dict], char_boxes: List[Dict]) -> Image.Image:
    angle = random.uniform(-18, 18)
    old_w, old_h = img.size
    out = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(245, 245, 245))
    new_w, new_h = out.size

    apply_rotation_to_boxes(word_boxes, old_w, old_h, angle, new_w, new_h)
    apply_rotation_to_boxes(char_boxes, old_w, old_h, angle, new_w, new_h)

    return out

def apply_perspective_like(img: Image.Image, word_boxes: List[Dict], char_boxes: List[Dict]) -> Image.Image:
    w, h = img.size
    left_pad = random.randint(0, 90)
    right_pad = random.randint(0, 90)
    top_pad = random.randint(0, 70)
    bottom_pad = random.randint(0, 70)

    new_w = w + left_pad + right_pad
    new_h = h + top_pad + bottom_pad
    out = Image.new("RGB", (new_w, new_h), (245, 245, 245))

    top_scale = random.uniform(0.86, 1.0)
    bottom_scale = random.uniform(0.86, 1.0)
    top_w = int(w * top_scale)
    bottom_w = int(w * bottom_scale)

    src = np.array(img)
    out_arr = np.full((new_h, new_w, 3), 245, dtype=np.uint8)

    row_params = []
    for y in range(h):
        ratio = y / max(h - 1, 1)
        row_w = int(top_w * (1 - ratio) + bottom_w * ratio)
        x_offset = left_pad + (w - row_w) // 2 + random.randint(-4, 4)
        x_offset = max(0, min(new_w - row_w, x_offset))

        row = Image.fromarray(src[y:y+1, :, :]).resize((row_w, 1))
        out_arr[top_pad + y, x_offset:x_offset + row_w] = np.array(row)[0]
        row_params.append((row_w, x_offset))

    for box_list in [word_boxes, char_boxes]:
        for b in box_list:
            new_quad = []
            for x, y in b["quad"]:
                y_int = int(max(0, min(h - 1, round(y))))
                row_w, x_offset = row_params[y_int]
                x_scaled = (x / max(w, 1)) * row_w
                nx = x_offset + x_scaled
                ny = top_pad + y
                new_quad.append([nx, ny])
            b["quad"] = clip_quad(new_quad, new_w, new_h)
            update_rect_from_quad(b)

    return Image.fromarray(out_arr)

def finalize_size(img: Image.Image, word_boxes: List[Dict], char_boxes: List[Dict]) -> Image.Image:
    if maybe(0.5):
        down = random.uniform(0.6, 0.9)
        w, h = img.size
        sw = max(200, int(w * down))
        sh = max(150, int(h * down))
        small = img.resize((sw, sh), Image.Resampling.BILINEAR)
        out = small.resize((w, h), Image.Resampling.BILINEAR)

        sx = sw / w
        sy = sh / h
        for box_list in [word_boxes, char_boxes]:
            for b in box_list:
                scaled = [[px * sx, py * sy] for px, py in b["quad"]]
                restored = [[px / sx, py / sy] for px, py in scaled]
                b["quad"] = clip_quad(restored, w, h)
                update_rect_from_quad(b)

        return out
    return img

def apply_handheld_crop(img: Image.Image, word_boxes: List[Dict], char_boxes: List[Dict]) -> Image.Image:
    scale = random.uniform(1.03, 1.15)
    ow, oh = img.size
    nw, nh = int(ow * scale), int(oh * scale)
    zoomed = img.resize((nw, nh), Image.Resampling.BICUBIC)

    for box_list in [word_boxes, char_boxes]:
        for b in box_list:
            b["quad"] = [[x * scale, y * scale] for x, y in b["quad"]]
            update_rect_from_quad(b)

    crop_w = min(CANVAS_W, nw)
    crop_h = min(CANVAS_H, nh)
    x = random.randint(0, max(0, nw - crop_w))
    y = random.randint(0, max(0, nh - crop_h))

    cropped = zoomed.crop((x, y, x + crop_w, y + crop_h))
    apply_shift_to_boxes(word_boxes, -x, -y)
    apply_shift_to_boxes(char_boxes, -x, -y)

    for box_list in [word_boxes, char_boxes]:
        for b in box_list:
            b["quad"] = clip_quad(b["quad"], crop_w, crop_h)
            update_rect_from_quad(b)

    return cropped

def apply_effects(img: Image.Image, effects: List[str], word_boxes: List[Dict], char_boxes: List[Dict]) -> Image.Image:
    if "perspective" in effects:
        img = apply_perspective_like(img, word_boxes, char_boxes)
    if "rotation" in effects:
        img = apply_rotation(img, word_boxes, char_boxes)
    if "shadow" in effects:
        img = apply_shadow(img)
    if "glare" in effects:
        img = apply_glare(img)
    if "occlusion" in effects:
        img = apply_occlusion(img, word_boxes, char_boxes)
    if "low_light" in effects:
        img = apply_low_light(img)
    if "blur" in effects:
        img = apply_blur(img)
    if "noise" in effects:
        img = apply_noise(img)

    img = finalize_size(img, word_boxes, char_boxes)

    if "handheld" in effects:
        img = apply_handheld_crop(img, word_boxes, char_boxes)

    return img

# =========================================================
# annotation 저장
# =========================================================

def clean_box_output(boxes: List[Dict]) -> List[Dict]:
    out = []
    for b in boxes:
        item = dict(b)
        item["bbox"] = [round(v, 2) for v in item["bbox"]]
        item["quad"] = [[round(x, 2), round(y, 2)] for x, y in item["quad"]]
        out.append(item)
    return out

def build_sample_annotation(
    sample_id: str,
    image_rel_path: str,
    profile: Dict,
    lang_type: str,
    difficulty_type: str,
    effects: List[str],
    word_boxes: List[Dict],
    char_boxes: List[Dict],
) -> Dict:
    return {
        "id": sample_id,
        "image": image_rel_path,
        "lang_type": lang_type,
        "difficulty_type": difficulty_type,
        "effects": effects,
        "company_ko": profile["company_ko"],
        "company_en": profile["company_en"],
        "name_ko": profile["name_ko"],
        "name_en": profile["name_en"],
        "job_title_ko": profile["job_title_ko"],
        "job_title_en": profile["job_title_en"],
        "department_ko": profile["department_ko"],
        "department_en": profile["department_en"],
        "email": profile["email"],
        "phones": profile["phones"],
        "address_ko": profile["address_ko"],
        "address_en": profile["address_en"],
        "slogan_ko": profile["slogan_ko"],
        "slogan_en": profile["slogan_en"],
        "render_name": profile["render_name"],
        "render_company": profile["render_company"],
        "render_title": profile["render_title"],
        "render_department": profile["render_department"],
        "render_address": profile["render_address"],
        "render_slogan": profile["render_slogan"],
        "challenging_flags": profile["challenging_flags"],
        "word_boxes": clean_box_output(word_boxes),
        "char_boxes": clean_box_output(char_boxes),
    }

def build_labels_jsonl_entry(sample_ann: Dict) -> Dict:
    return {
        "id": sample_ann["id"],
        "image": sample_ann["image"],
        "lang_type": sample_ann["lang_type"],
        "difficulty_type": sample_ann["difficulty_type"],
        "effects": sample_ann["effects"],
        "company_ko": sample_ann["company_ko"],
        "company_en": sample_ann["company_en"],
        "name_ko": sample_ann["name_ko"],
        "name_en": sample_ann["name_en"],
        "job_title_ko": sample_ann["job_title_ko"],
        "job_title_en": sample_ann["job_title_en"],
        "department_ko": sample_ann["department_ko"],
        "department_en": sample_ann["department_en"],
        "email": sample_ann["email"],
        "phones": sample_ann["phones"],
        "address_ko": sample_ann["address_ko"],
        "address_en": sample_ann["address_en"],
        "slogan_ko": sample_ann["slogan_ko"],
        "slogan_en": sample_ann["slogan_en"],
        "challenging_flags": sample_ann["challenging_flags"],
    }

# =========================================================
# 샘플 생성
# =========================================================

def generate_one(sample_idx: int, companies: List[Dict[str, str]]) -> Dict:
    sample_id = f"card_{sample_idx:06d}"

    lang_type = weighted_choice(LANG_TYPE_WEIGHTS)
    difficulty_type = weighted_choice(DIFFICULTY_WEIGHTS)

    profile = generate_profile(lang_type, companies, difficulty_type)
    effects = choose_effects(difficulty_type)

    card, word_boxes, char_boxes = create_base_card(profile, difficulty_type)
    canvas = paste_card_on_canvas(card, word_boxes, char_boxes, difficulty_type, effects)
    final_img = apply_effects(canvas, effects, word_boxes, char_boxes)

    image_name = f"{sample_id}.png"
    annotation_name = f"{sample_id}.json"

    image_path = IMAGE_DIR / image_name
    annotation_path = ANNOTATION_DIR / annotation_name

    final_img.save(image_path)

    sample_ann = build_sample_annotation(
        sample_id=sample_id,
        image_rel_path=str(Path("images") / image_name),
        profile=profile,
        lang_type=lang_type,
        difficulty_type=difficulty_type,
        effects=effects,
        word_boxes=word_boxes,
        char_boxes=char_boxes,
    )

    with open(annotation_path, "w", encoding="utf-8") as f:
        json.dump(sample_ann, f, ensure_ascii=False, indent=2)

    return build_labels_jsonl_entry(sample_ann)

# =========================================================
# 실행
# =========================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    ensure_dirs()

    companies = fetch_companies_from_dart(DART_API_KEY, limit=8000)
    if companies:
        print(f"[INFO] DART 회사명 {len(companies)}개 로드 완료")
    else:
        print("[INFO] fallback 회사명으로 생성 진행")

    with open(LABEL_PATH, "w", encoding="utf-8") as f:
        for i in range(NUM_SAMPLES):
            entry = generate_one(i, companies)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"[INFO] {i + 1}/{NUM_SAMPLES} 생성 완료")

    print(f"[DONE] 이미지 경로: {IMAGxE_DIR}")
    print(f"[DONE] annotation 경로: {ANNOTATION_DIR}")
    print(f"[DONE] labels 경로: {LABEL_PATH}")

if __name__ == "__main__":
    main()
