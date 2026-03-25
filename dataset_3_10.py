import os
import re
import io
import json
import math
import random
import zipfile
import requests
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# =========================================================
# 설정
# =========================================================

OUTPUT_DIR = Path("./dataset_3_10")
IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"
LABEL_PATH = LABEL_DIR / "labels.jsonl"
LABEL_JSON_PATH = LABEL_DIR / "labels.json"

NUM_SAMPLES = 1000

# 언어 유형 비율
LANG_TYPE_WEIGHTS = {
    "english_only": 0.25,
    "korean_only": 0.25,
    "mixed": 0.50,
}

# 난이도 비율
DIFFICULTY_WEIGHTS = {
    "clean": 0.25,
    "hard": 0.45,
    "confusing": 0.15,
    "challenging": 0.15,
}

# hard 샘플에서 효과 비율
# blur 비중을 더 높게 둠
HARD_EFFECT_WEIGHTS = {
    "blur": 0.22,
    "rotation": 0.10,
    "perspective": 0.10,
    "shadow": 0.10,
    "occlusion": 0.08,
    "low_light": 0.08,
    "glare": 0.08,
    "background_clutter": 0.08,
    "handheld": 0.08,
    "noise": 0.08,
}

CARD_WIDTH = 1000
CARD_HEIGHT = 600

CANVAS_W = 1400
CANVAS_H = 1000

# DART API KEY 환경변수
DART_API_KEY = os.environ.get("DART_API_KEY", "").strip()

# 폰트 경로 후보
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
# 기본 데이터
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

# 영어-only용 약간 그럴듯한 회사명 fallback
FALLBACK_COMPANY_EN = [
    "Asterix Labs", "Bluewave Systems", "Novalink Partners", "Vertex AI",
    "Northline Tech", "Pixelbridge Studio", "Brighton Dynamics", "Nexton Works",
    "Lunaris Data", "Primecore Solutions", "Horizon Cloud", "Altair Robotics"
]

# 한국어-only fallback
FALLBACK_COMPANY_KO = [
    "한빛테크", "넥스트에이아이", "미래데이터", "블루웨이브", "프라임솔루션",
    "에이펙스랩", "하이브릿지", "루나리스", "온리원시스템", "비전네트웍스"
]

# =========================================================
# 유틸
# =========================================================

def ensure_dirs():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

def weighted_choice(weight_dict: Dict[str, float]) -> str:
    keys = list(weight_dict.keys())
    weights = list(weight_dict.values())
    return random.choices(keys, weights=weights, k=1)[0]

def maybe(prob: float) -> bool:
    return random.random() < prob

def safe_str(x) -> str:
    return "" if x is None else str(x).strip()


def _phone_kind(label: str, value: str) -> str:
    lab = safe_str(label).lower().replace(".", "").strip()
    digits = re.sub(r"\D", "", safe_str(value))
    mobile_aliases = {"m", "mobile", "hp", "h p", "cell", "handphone"}
    if lab in mobile_aliases:
        return "M"
    if digits.startswith(("010", "011", "016", "017", "018", "019")):
        return "M"
    return "T"


def normalize_output_phones(phones: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    출력 라벨용 phone 포맷:
    - label은 T/M만 사용
    - value는 원 문자열 유지
    - 각각 최대 1개씩만 유지하여 깔끔한 구조 보장
    """
    t_value: Optional[str] = None
    m_value: Optional[str] = None

    for ph in phones:
        value = safe_str(ph.get("value"))
        if not value:
            continue

        kind = _phone_kind(ph.get("label", ""), value)
        if kind == "T" and t_value is None:
            t_value = value
        elif kind == "M" and m_value is None:
            m_value = value

    out = []
    if t_value:
        out.append({"label": "T", "value": t_value})
    if m_value:
        out.append({"label": "M", "value": m_value})

    return out

def normalize_company_name(name: str) -> str:
    name = safe_str(name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip()
    return name

def load_font(size: int, bold_preferred: bool = False):
    candidates = FONT_CANDIDATES[:]
    if bold_preferred:
        candidates = sorted(candidates, key=lambda x: 0 if "Bold" in x or "bold" in x else 1)

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
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

def draw_multiline_text(draw, xy, text, font, fill, max_width, line_spacing=8):
    x, y = xy
    words = text.split(" ")
    lines = []
    cur = ""

    for word in words:
        test = word if not cur else cur + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing
    return y

# =========================================================
# DART 회사명 수집
# =========================================================

def fetch_companies_from_dart(api_key: str, limit: int = 5000) -> List[Dict[str, str]]:
    """
    OpenDART corpCode.xml ZIP을 내려받아 회사명/영문회사명을 파싱한다.
    """
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

    # 중복 제거
    uniq = {}
    for c in companies:
        key = (c["company_ko"], c["company_en"])
        if key not in uniq:
            uniq[key] = c

    companies = list(uniq.values())
    random.shuffle(companies)
    return companies[:limit]

# =========================================================
# 이름/직함/연락처 생성
# =========================================================

def generate_korean_name() -> str:
    return random.choice(KOREAN_LAST_NAMES) + random.choice(KOREAN_FIRST_NAMES)

def generate_english_name() -> str:
    return f"{random.choice(ENGLISH_FIRST_NAMES)} {random.choice(ENGLISH_LAST_NAMES)}"

def generate_name_pair() -> Tuple[str, str]:
    ko = generate_korean_name()
    en = generate_english_name()
    return ko, en

def generate_phone_number(kind: str = "office") -> str:
    if kind == "mobile":
        return f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"

    area_codes = ["02", "031", "032", "033", "041", "042", "051", "052", "053", "054", "055", "061", "062", "063", "064"]
    area = random.choice(area_codes)
    if area == "02":
        mid = random.randint(100, 9999)
    else:
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
    # 정교한 로마자 변환은 아니고 fallback용 단순 규칙
    cleaned = re.sub(r"[^가-힣A-Za-z0-9 ]", "", ko).strip()
    if not cleaned:
        return random.choice(FALLBACK_COMPANY_EN)
    # 간단 fallback
    return cleaned.upper()

def get_company_pair(companies: List[Dict[str, str]]) -> Tuple[str, str]:
    if companies:
        c = random.choice(companies)
        ko = c["company_ko"]
        en = c["company_en"] if c["company_en"] else romanize_company_simple(ko)
        return ko, en

    ko = random.choice(FALLBACK_COMPANY_KO)
    en = random.choice(FALLBACK_COMPANY_EN)
    return ko, en

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

    # 혼동 유발형: 번호 2~4개
    phone_count = random.randint(2, 4) if difficulty_type == "confusing" else random.randint(1, 3)
    label_set = random.choice(PHONE_LABELS_MIXED)

    for i in range(phone_count):
        label = label_set[i % len(label_set)]
        if label in ["M", "Mobile", "HP"]:
            val = generate_phone_number("mobile")
        else:
            val = generate_phone_number("office")
        phones.append({"label": label, "value": val})

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

    # 언어 타입 적용
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
        # mixed
        # 이름/회사명 둘 다 섞되, 모든 필드가 다 bilingual일 필요는 없음
        profile["render_name"] = f"{name_ko} | {name_en}" if maybe(0.7) else name_ko
        profile["render_company"] = f"{company_ko} | {company_en}" if maybe(0.7) else company_en
        profile["render_title"] = f"{title_ko} / {title_en}" if maybe(0.7) else title_en
        profile["render_department"] = f"{dept_ko} | {dept_en}" if maybe(0.7) else dept_ko
        profile["render_address"] = f"{address_ko}\n{address_en}" if maybe(0.5) else address_ko
        profile["render_slogan"] = f"{slogan_ko} / {slogan_en}" if maybe(0.5) else slogan_en

    # 고난도 해석형
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
# 렌더링
# =========================================================

def create_base_card(profile: Dict, difficulty_type: str) -> Image.Image:
    vertical = profile["challenging_flags"].get("vertical_card", False)

    width, height = (700, 1100) if vertical else (CARD_WIDTH, CARD_HEIGHT)

    bg = random_color(bright=True)
    card = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(card)

    fg = pick_text_color(bg)

    # 장식용 로고/바
    accent = random_color(dark=(sum(bg) > 400))
    if profile["challenging_flags"].get("big_logo_small_text", False):
        draw.ellipse((30, 30, 180, 180), fill=accent)
        draw.text((60, 80), profile["render_company"][:1], font=load_font(60, True), fill=fg)
    else:
        draw.rectangle((0, 0, width, 18), fill=accent)
        if maybe(0.4):
            draw.rectangle((0, height - 14, width, height), fill=accent)

    # 폰트
    company_font = load_font(38, True)
    name_font = load_font(46, True)
    title_font = load_font(28)
    body_font = load_font(26)
    small_font = load_font(20)

    if vertical:
        margin_x = 50
        y = 60
        max_w = width - 100
    else:
        margin_x = 50
        y = 55
        max_w = width - 100

    # 회사명
    if not profile["challenging_flags"].get("name_only_emphasis", False):
        y = draw_multiline_text(draw, (margin_x, y), profile["render_company"], company_font, fg, max_w)
        y += 15

    # 이름 크게
    name_font_size = 62 if profile["challenging_flags"].get("name_only_emphasis", False) else 46
    name_font = load_font(name_font_size, True)
    y = draw_multiline_text(draw, (margin_x, y), profile["render_name"], name_font, fg, max_w)
    y += 10

    # 직함 / 부서
    if maybe(0.85):
        y = draw_multiline_text(draw, (margin_x, y), profile["render_title"], title_font, fg, max_w)
    if maybe(0.8):
        y = draw_multiline_text(draw, (margin_x, y + 5), profile["render_department"], title_font, fg, max_w)
    y += 15

    # 구분선
    if maybe(0.8):
        draw.line((margin_x, y, width - margin_x, y), fill=fg, width=2)
        y += 18

    # 연락처
    for ph in profile["phones"]:
        line = f"{ph['label']}: {ph['value']}"
        draw.text((margin_x, y), line, font=body_font, fill=fg)
        y += 36

    # 이메일은 일부러 작게 넣을 수 있음
    email_font = small_font if difficulty_type == "confusing" and maybe(0.7) else body_font
    draw.text((margin_x, y + 8), profile["email"], font=email_font, fill=fg)
    y += 52

    # 주소 + 슬로건
    # confusing에서는 주소와 슬로건이 섞이게
    if difficulty_type == "confusing" and maybe(0.7):
        mixed_line = f"{profile['render_address']}   |   {profile['render_slogan']}"
        draw_multiline_text(draw, (margin_x, y), mixed_line, small_font, fg, max_w)
    else:
        if maybe(0.75):
            y = draw_multiline_text(draw, (margin_x, y), profile["render_address"], small_font, fg, max_w)
            y += 10
        if maybe(0.75):
            draw_multiline_text(draw, (margin_x, y), profile["render_slogan"], small_font, fg, max_w)

    return card

# =========================================================
# 이미지 효과
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
            x2, y2 = min(canvas.width, x1 + random.randint(20, 160)), min(canvas.height, y1 + random.randint(20, 120))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        else:
            x1, y1 = random.randint(0, canvas.width - 1), random.randint(0, canvas.height - 1)
            x2, y2 = min(canvas.width, x1 + random.randint(20, 160)), min(canvas.height, y1 + random.randint(20, 120))
            draw.ellipse((x1, y1, x2, y2), outline=color, width=2)

def paste_card_on_canvas(card: Image.Image, difficulty_type: str, applied_effects: List[str]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), random_color(bright=True))

    if "background_clutter" in applied_effects or difficulty_type in ["hard", "challenging"]:
        if maybe(0.7):
            add_background_clutter(canvas)

    # 카드가 캔버스보다 커져 배치 범위가 음수가 되는 경우를 방지
    max_card_w = CANVAS_W - 40
    max_card_h = CANVAS_H - 40
    if card.width > max_card_w or card.height > max_card_h:
        scale = min(max_card_w / max(1, card.width), max_card_h / max(1, card.height))
        scale = max(scale, 0.05)
        new_w = max(1, int(card.width * scale))
        new_h = max(1, int(card.height * scale))
        card = card.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 살짝 회전 전 shadow
    shadow = Image.new("RGBA", card.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle((10, 10, card.width - 5, card.height - 5), radius=18, fill=(0, 0, 0, 90))

    # handheld / random placement
    x_margin = min(120, max(0, (CANVAS_W - card.width) // 2))
    y_margin = min(100, max(0, (CANVAS_H - card.height) // 2))

    x_min = x_margin
    x_max = CANVAS_W - card.width - x_margin
    y_min = y_margin
    y_max = CANVAS_H - card.height - y_margin

    x = random.randint(x_min, x_max) if x_max >= x_min else max(0, (CANVAS_W - card.width) // 2)
    y = random.randint(y_min, y_max) if y_max >= y_min else max(0, (CANVAS_H - card.height) // 2)

    base = canvas.convert("RGBA")
    base.alpha_composite(shadow, (x + 18, y + 18))
    base.alpha_composite(card.convert("RGBA"), (x, y))
    return base.convert("RGB"), (x, y, x + card.width, y + card.height)

def apply_rotation(img: Image.Image) -> Image.Image:
    angle = random.uniform(-18, 18)
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(245, 245, 245))

def apply_blur(img: Image.Image) -> Image.Image:
    radius = random.uniform(1.2, 3.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_low_light(img: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.45, 0.8))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.0))
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
    # 대각선 그림자 느낌
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

def apply_occlusion(img: Image.Image) -> Image.Image:
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    # 손가락/물체 가림 비슷하게
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
    return overlay

def apply_perspective_like(img: Image.Image) -> Image.Image:
    # PIL 기본 기능만으로 간단한 원근 비슷한 느낌
    w, h = img.size
    left_pad = random.randint(0, 90)
    right_pad = random.randint(0, 90)
    top_pad = random.randint(0, 70)
    bottom_pad = random.randint(0, 70)

    new_w = w + left_pad + right_pad
    new_h = h + top_pad + bottom_pad
    canvas = Image.new("RGB", (new_w, new_h), (245, 245, 245))

    # 위아래 폭 차이를 주는 식의 간이 구현
    top_scale = random.uniform(0.86, 1.0)
    bottom_scale = random.uniform(0.86, 1.0)
    top_w = int(w * top_scale)
    bottom_w = int(w * bottom_scale)

    rows = []
    src = np.array(img)
    for y in range(h):
        ratio = y / max(h - 1, 1)
        row_w = int(top_w * (1 - ratio) + bottom_w * ratio)
        row = Image.fromarray(src[y:y+1, :, :]).resize((row_w, 1))
        rows.append(np.array(row)[0])

    out = np.full((new_h, new_w, 3), 245, dtype=np.uint8)
    y_start = top_pad
    for i, row in enumerate(rows):
        row_w = row.shape[0]
        x_offset = left_pad + (w - row_w) // 2 + random.randint(-8, 8)
        x_offset = max(0, min(new_w - row_w, x_offset))
        out[y_start + i, x_offset:x_offset + row_w] = row

    return Image.fromarray(out)

def apply_handheld_crop(img: Image.Image) -> Image.Image:
    # 살짝 확대 후 크롭해서 손으로 찍은 프레이밍 느낌
    scale = random.uniform(1.03, 1.15)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    zoomed = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    x = random.randint(0, new_w - CANVAS_W) if new_w > CANVAS_W else 0
    y = random.randint(0, new_h - CANVAS_H) if new_h > CANVAS_H else 0
    cropped = zoomed.crop((x, y, x + min(CANVAS_W, new_w), y + min(CANVAS_H, new_h)))
    return cropped

def finalize_size(img: Image.Image) -> Image.Image:
    # 최종 해상도 랜덤 다운샘플링 후 복원 -> 품질 저하
    if maybe(0.5):
        down = random.uniform(0.6, 0.9)
        w, h = img.size
        small = img.resize((max(200, int(w * down)), max(150, int(h * down))), Image.Resampling.BILINEAR)
        img = small.resize((w, h), Image.Resampling.BILINEAR)
    return img

def choose_effects(difficulty_type: str) -> List[str]:
    effects = []

    if difficulty_type == "clean":
        return effects

    if difficulty_type == "hard":
        n = random.randint(1, 3)
    elif difficulty_type == "confusing":
        n = random.randint(1, 2)
    else:  # challenging
        n = random.randint(2, 4)

    keys = list(HARD_EFFECT_WEIGHTS.keys())
    weights = list(HARD_EFFECT_WEIGHTS.values())
    while len(effects) < n:
        e = random.choices(keys, weights=weights, k=1)[0]
        if e not in effects:
            effects.append(e)

    return effects

def apply_effects(img: Image.Image, effects: List[str]) -> Image.Image:
    # 순서 중요
    if "perspective" in effects:
        img = apply_perspective_like(img)
    if "rotation" in effects:
        img = apply_rotation(img)
    if "shadow" in effects:
        img = apply_shadow(img)
    if "glare" in effects:
        img = apply_glare(img)
    if "occlusion" in effects:
        img = apply_occlusion(img)
    if "low_light" in effects:
        img = apply_low_light(img)
    if "blur" in effects:
        img = apply_blur(img)
    if "noise" in effects:
        img = apply_noise(img)
    img = finalize_size(img)
    if "handheld" in effects:
        img = apply_handheld_crop(img)
    return img

# =========================================================
# 샘플 생성
# =========================================================

def build_annotation(sample_id: str, image_rel_path: str, profile: Dict, lang_type: str, difficulty_type: str, effects: List[str]) -> Dict:
    phones = normalize_output_phones(profile.get("phones", []))
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
        "phones": phones,
    }

def generate_one(sample_idx: int, companies: List[Dict[str, str]]) -> Dict:
    sample_id = f"card_{sample_idx:06d}"

    lang_type = weighted_choice(LANG_TYPE_WEIGHTS)
    difficulty_type = weighted_choice(DIFFICULTY_WEIGHTS)

    profile = generate_profile(lang_type, companies, difficulty_type)
    effects = choose_effects(difficulty_type)

    card = create_base_card(profile, difficulty_type)
    canvas, _ = paste_card_on_canvas(card, difficulty_type, effects)
    final_img = apply_effects(canvas, effects)

    image_name = f"{sample_id}.png"
    image_path = IMAGE_DIR / image_name
    final_img.save(image_path, quality=95)

    ann = build_annotation(
        sample_id=sample_id,
        image_rel_path=str(Path("images") / image_name),
        profile=profile,
        lang_type=lang_type,
        difficulty_type=difficulty_type,
        effects=effects,
    )
    return ann

# =========================================================
# 실행
# =========================================================

def main():
    ensure_dirs()

    companies = fetch_companies_from_dart(DART_API_KEY, limit=8000)
    if companies:
        print(f"[INFO] DART 회사명 {len(companies)}개 로드 완료")
    else:
        print("[INFO] fallback 회사명으로 생성 진행")

    anns = []
    with open(LABEL_PATH, "w", encoding="utf-8") as f:
        for i in range(NUM_SAMPLES):
            ann = generate_one(i, companies)
            anns.append(ann)
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"[INFO] {i + 1}/{NUM_SAMPLES} 생성 완료")

    with open(LABEL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(anns, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 이미지 저장 경로: {IMAGE_DIR}")
    print(f"[DONE] 라벨 저장 경로: {LABEL_PATH}")
    print(f"[DONE] 라벨(JSON) 경로: {LABEL_JSON_PATH}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
