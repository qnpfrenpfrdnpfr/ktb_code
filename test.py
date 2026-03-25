"""
명함 학습 데이터 생성기
- DART API로 실제 회사명 수집
- 다양한 회사명 패턴 (주식회사, (주), Corp 등)
- 다양한 전화번호 패턴 (하이픈, 점, 괄호, 국가코드, 접두어 등)
- 시각적 노이즈 (블러, 노이즈, 그림자, 회전) 적용
"""

import os
import re
import json
import random
import string
import requests
import numpy as np
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
DART_API_KEY = os.environ.get("DART_API_KEY", "15c8269598c75ff944aa17ffcd54acba76f75608")
OUTPUT_DIR = Path("business_card_dataset")
IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DART API - 회사명 수집
# ─────────────────────────────────────────────

def fetch_dart_companies(api_key: str, max_count: int = 500) -> list[dict]:
    """
    DART Open API에서 상장법인 목록을 가져옵니다.
    엔드포인트: https://opendart.fss.or.kr/api/corpCode.xml
    (전체 corpCode.zip 대신 검색 API 사용)
    """
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    params = {"crtfc_key": api_key}

    print("[DART] 상장법인 코드 파일 다운로드 중...")
    resp = requests.get(url, params=params, timeout=30)

    if resp.status_code != 200:
        print(f"[DART] 오류: HTTP {resp.status_code}")
        return []

    # ZIP 파일로 반환됨 → 압축 해제
    import zipfile, io, xml.etree.ElementTree as ET

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            xml_name = [n for n in z.namelist() if n.endswith(".xml")][0]
            xml_data = z.read(xml_name)
    except Exception as e:
        print(f"[DART] ZIP 파싱 오류: {e}")
        return []

    root = ET.fromstring(xml_data)
    companies = []
    for corp in root.findall("list"):
        name = corp.findtext("corp_name", "").strip()
        code = corp.findtext("corp_code", "").strip()
        stock = corp.findtext("stock_code", "").strip()
        if name and stock:          # 상장기업만
            companies.append({"name": name, "corp_code": code, "stock_code": stock})
        if len(companies) >= max_count:
            break

    print(f"[DART] {len(companies)}개 회사 수집 완료")
    return companies


def get_fallback_companies() -> list[str]:
    """DART API 미사용 시 대체 샘플 회사명"""
    return [
        "삼성전자", "SK하이닉스", "LG전자", "현대자동차", "카카오",
        "네이버", "포스코", "한화시스템", "두산에너빌리티", "롯데케미칼",
        "CJ제일제당", "GS칼텍스", "KT", "신세계", "아모레퍼시픽",
        "하이브", "크래프톤", "카카오뱅크", "토스뱅크", "쿠팡",
        "배달의민족", "야놀자", "무신사", "마켓컬리", "오늘의집",
    ]


# ─────────────────────────────────────────────
# 2. 회사명 패턴 변형
# ─────────────────────────────────────────────

CORP_PREFIXES = ["주식회사 ", "(주) ", "㈜ ", "유한회사 ", "(유) "]
CORP_SUFFIXES = [
    " 주식회사", " (주)", " ㈜", " Corp.", " Corp", " Corporation",
    " Co., Ltd.", " Co.,Ltd.", " Co.Ltd.", " Ltd.", " Inc.", " LLC",
    " Group", " Holdings", " 그룹", " 홀딩스",
]

def vary_company_name(base_name: str) -> str:
    """회사명에 다양한 법인 표기 패턴을 붙여 반환"""
    r = random.random()
    if r < 0.25:
        return random.choice(CORP_PREFIXES) + base_name
    elif r < 0.50:
        return base_name + random.choice(CORP_SUFFIXES)
    elif r < 0.65:
        return base_name  # 순수 상호만
    else:
        # 접두 + 접미 동시 (현실에서도 종종 등장)
        return random.choice(CORP_PREFIXES) + base_name + random.choice(CORP_SUFFIXES)


# ─────────────────────────────────────────────
# 3. 전화번호 패턴 생성
# ─────────────────────────────────────────────

COMPANY_PHONE_PREFIXES = ["T", "Tel", "TEL", "T.", "Tel.", "P", "Phone", ""]
MOBILE_PHONE_PREFIXES = ["M", "Mobile", "Mobile.", "H.P.", "HP", ""]
PHONE_PREFIXES = COMPANY_PHONE_PREFIXES + MOBILE_PHONE_PREFIXES
COUNTRY_CODES  = ["+82", "0082", "82-", ""]
SEPARATORS     = ["-", ".", " ", ""]

AREA_CODES_KR = ["02", "031", "032", "033", "041", "042", "043", "051",
                 "052", "053", "054", "055", "061", "062", "063", "064"]
MOBILE_PREFIXES_KR = ["010", "011", "016", "017", "019"]


def _fmt(parts: list[str], sep: str) -> str:
    return sep.join(parts)


def generate_phone_number(kind: str = "", prefixes = None) -> str:
    """다양한 형식의 한국 전화번호 생성

    kind:
      - "mobile": 휴대폰
      - "landline": 유선(회사전화)
      - "" (default): 랜덤
    """
    sep  = random.choice(SEPARATORS)
    cc   = random.choice(COUNTRY_CODES)
    if prefixes is None:
        prefixes = PHONE_PREFIXES
    pref = random.choice(prefixes)

    if kind not in ("mobile", "landline"):
        kind = random.choice(["mobile", "landline"])

    if kind == "mobile":
        area = random.choice(MOBILE_PREFIXES_KR)
        mid  = str(random.randint(1000, 9999))
        last = str(random.randint(1000, 9999))
        # 국가코드 있으면 앞 0 제거
        if cc:
            area = area[1:]
    else:
        area = random.choice(AREA_CODES_KR)
        digits = 8 if area == "02" else 7
        mid  = str(random.randint(10**(digits//2-1), 10**(digits//2)-1))
        last = str(random.randint(1000, 9999))
        if cc:
            area = area[1:]

    number = _fmt([area, mid, last], sep)
    if cc:
        number = cc + number

    # 접두어 포맷
    if pref:
        joiner = random.choice([" ", ". ", ": ", " : "])
        return f"{pref}{joiner}{number}"
    return number


def generate_company_phone_number() -> str:
    """회사 전화(유선) 번호 생성"""
    return generate_phone_number(kind="landline", prefixes=COMPANY_PHONE_PREFIXES)


def generate_mobile_phone_number() -> str:
    """휴대폰 번호 생성"""
    return generate_phone_number(kind="mobile", prefixes=MOBILE_PHONE_PREFIXES)


def generate_email() -> str:
    names = ["info", "contact", "hr", "sales", "support", "admin", "help", "cs"]
    domains = ["gmail.com", "naver.com", "daum.net", "kakao.com", "company.co.kr",
               "corp.kr", "business.com", "outlook.com"]
    return f"{random.choice(names)}@{random.choice(domains)}"


def generate_address() -> str:
    cities = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
              "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도"]
    dongs = ["강남구 테헤란로", "서초구 반포대로", "마포구 월드컵북로",
             "중구 을지로", "종로구 종로", "해운대구 센텀중앙로"]
    num = random.randint(1, 500)
    floor = random.randint(1, 30)
    return f"{random.choice(cities)} {random.choice(dongs)} {num}, {floor}층"


# ─────────────────────────────────────────────
# 4. 명함 이미지 생성
# ─────────────────────────────────────────────

CARD_W, CARD_H = 856, 540      # 표준 명함 비율 (85.6 × 54 mm)
BG_COLORS = [
    (255, 255, 255), (245, 245, 245), (240, 248, 255),
    (255, 250, 240), (240, 255, 240), (30, 30, 30),
    (15, 40, 80), (80, 0, 0),
]
TEXT_COLORS_LIGHT = [(20, 20, 20), (40, 40, 80), (80, 20, 20)]
TEXT_COLORS_DARK  = [(240, 240, 240), (200, 220, 255), (255, 220, 200)]


@lru_cache(maxsize=256)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _pick_font(size: int):
    """한글 지원 폰트를 우선 탐색해 반환합니다.

    macOS에서 Linux 폰트 경로만 체크하면 기본 폰트로 fallback되어 한글이 깨질 수 있습니다.
    """
    candidates = [
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/NotoSansCJK.ttc",
        "/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NotoSansCJK-Regular.ttc",
        "/Library/Fonts/NanumGothic.ttf",

        # Linux
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",

        # Windows
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return _load_font(path, size)
            except Exception:
                continue

    raise RuntimeError(
        "한글 지원 폰트를 찾지 못했습니다. "
        "macOS면 AppleSDGothicNeo.ttc가 있어야 하고, "
        "Linux면 Noto Sans CJK 또는 NanumGothic 설치 후 경로를 candidates에 추가하세요."
    )


def draw_business_card(
    company: str,
    name: str,
    title: str,
    company_phone: str,
    mobile_phone: str,
    email: str,
    address: str,
    bg_color: tuple,
) -> Image.Image:
    """PIL로 명함 이미지를 그립니다."""
    img = Image.new("RGB", (CARD_W, CARD_H), bg_color)
    draw = ImageDraw.Draw(img)

    dark_bg = sum(bg_color) < 384
    tc = random.choice(TEXT_COLORS_DARK if dark_bg else TEXT_COLORS_LIGHT)
    accent = (random.randint(0,200), random.randint(0,100), random.randint(100,255))

    # 회사명 (강조)
    f_company = _pick_font(random.randint(28, 36))
    draw.text((50, 50), company, font=f_company, fill=accent)

    # 구분선
    line_y = 105
    draw.line([(50, line_y), (CARD_W - 50, line_y)], fill=tc, width=1)

    # 성명 + 직함
    f_name  = _pick_font(random.randint(30, 40))
    f_title = _pick_font(random.randint(20, 26))
    draw.text((50, 125), name,  font=f_name,  fill=tc)
    draw.text((50, 175), title, font=f_title, fill=tc)

    # 연락처 정보
    f_info = _pick_font(random.randint(18, 24))
    info_lines = []
    if company_phone:
        info_lines.append(company_phone)
    if mobile_phone:
        info_lines.append(mobile_phone)
    info_lines.append(email)

    info_y = 250
    info_gap = 34
    for i, line in enumerate(info_lines):
        draw.text((50, info_y + (i * info_gap)), line, font=f_info, fill=tc)

    # 주소 (길면 줄 바꿈)
    address_y = info_y + (len(info_lines) * info_gap) + 25
    if len(address) > 30:
        half = address.rfind(" ", 0, 30)
        if half == -1:
            half = 30
        draw.text((50, address_y), address[:half], font=f_info, fill=tc)
        draw.text((50, address_y + info_gap), address[half:].lstrip(), font=f_info, fill=tc)
    else:
        draw.text((50, address_y), address, font=f_info, fill=tc)

    return img


# ─────────────────────────────────────────────
# 5. 시각적 증강 (블러 / 노이즈 / 그림자 / 회전)
# ─────────────────────────────────────────────

def augment_image(img: Image.Image, config: dict) -> Image.Image:
    """
    config 키:
        blur       : float  — GaussianBlur radius (0 = 없음)
        noise      : float  — 가우시안 노이즈 표준편차 (0 = 없음)
        shadow     : bool   — 그림자 효과 추가
        rotation   : float  — 회전 각도 (도, 0 = 없음)
        perspective: bool   — 원근 왜곡
        jpeg_quality: int   — JPEG 압축 아티팩트 (100 = 없음)
    """
    arr = np.array(img).astype(np.float32)

    # 1) 가우시안 노이즈
    noise_std = config.get("noise", 0)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, arr.shape)
        arr = np.clip(arr + noise, 0, 255)

    img = Image.fromarray(arr.astype(np.uint8))

    # 2) 블러
    blur_r = config.get("blur", 0)
    if blur_r > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_r))

    # 3) 그림자 (이미지 위에 어두운 gradient 오버레이)
    if config.get("shadow", False):
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

    # 4) 원근 왜곡 (OpenCV)
    if config.get("perspective", False):
        arr_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = arr_cv.shape[:2]
        jitter = lambda: random.randint(-30, 30)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [jitter(), jitter()],
            [w + jitter(), jitter()],
            [w + jitter(), h + jitter()],
            [jitter(), h + jitter()],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        arr_cv = cv2.warpPerspective(arr_cv, M, (w, h), borderValue=(255, 255, 255))
        img = Image.fromarray(cv2.cvtColor(arr_cv, cv2.COLOR_BGR2RGB))

    # 5) 회전
    angle = config.get("rotation", 0)
    if abs(angle) > 0.1:
        img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    # 6) JPEG 압축 아티팩트
    quality = config.get("jpeg_quality", 100)
    if quality < 95:
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).copy()

    return img


def random_augment_config() -> dict:
    """랜덤 증강 설정 생성"""
    return {
        "blur":          random.choice([0, 0, 0.5, 1.0, 1.5, 2.0]),
        "noise":         random.choice([0, 0, 3, 7, 12, 20]),
        "shadow":        random.random() < 0.35,
        "rotation":      random.choice([0, 0, 0, random.uniform(-8, 8)]),
        "perspective":   random.random() < 0.25,
        "jpeg_quality":  random.choice([100, 100, 95, 85, 75, 60]),
    }


# ─────────────────────────────────────────────
# 6. 이름 / 직함 샘플
# ─────────────────────────────────────────────

KR_SURNAMES  = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
KR_NAMES_1   = ["민준", "서연", "지호", "수빈", "예진", "현우", "도현", "지수"]
TITLES_KR    = ["대표이사", "이사", "부장", "차장", "과장", "대리", "사원",
                "연구원", "선임연구원", "수석연구원", "팀장", "본부장", "전무"]

def random_korean_name() -> str:
    return random.choice(KR_SURNAMES) + random.choice(KR_NAMES_1)


# ─────────────────────────────────────────────
# 7. 전체 데이터셋 생성
# ─────────────────────────────────────────────

def generate_dataset(
    company_names: list[str],
    n_samples: int = 500,
    augment: bool = True,
):
    """
    company_names : 기본 회사명 리스트
    n_samples     : 생성할 이미지 수
    augment       : 시각적 증강 적용 여부
    """
    labels = []
    used_companies = (company_names * ((n_samples // len(company_names)) + 2))[:n_samples]
    random.shuffle(used_companies)

    print(f"\n[GEN] {n_samples}장 명함 이미지 생성 시작...")

    for idx, base_name in enumerate(used_companies):
        company = vary_company_name(base_name)
        name    = random_korean_name()
        title   = random.choice(TITLES_KR)
        phone_mode = random.choice(["company_only", "mobile_only", "both"])
        company_phone = generate_company_phone_number() if phone_mode in ("company_only", "both") else ""
        mobile_phone = generate_mobile_phone_number() if phone_mode in ("mobile_only", "both") else ""
        email   = generate_email()
        address = generate_address()
        bg      = random.choice(BG_COLORS)

        # 명함 이미지 생성
        card_img = draw_business_card(company, name, title, company_phone, mobile_phone, email, address, bg)

        # 증강 적용
        aug_cfg = {}
        if augment:
            aug_cfg = random_augment_config()
            card_img = augment_image(card_img, aug_cfg)

        # 저장
        filename = f"card_{idx:05d}.jpg"
        save_path = IMAGE_DIR / filename
        card_img.save(save_path, "JPEG", quality=95)

        # 라벨 저장
        label = {
            "file": filename,
            "company_raw": base_name,
            "company_display": company,
            "name": name,
            "title": title,
            # Backward compat: 기존 스크립트가 record["phone"]을 볼 수 있게 유지
            "phone": company_phone or mobile_phone,
            "phone_mode": phone_mode,
            "company_phone": company_phone,
            "mobile_phone": mobile_phone,
            "email": email,
            "address": address,
            "bg_color": bg,
            "augmentation": aug_cfg,
        }
        labels.append(label)

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{n_samples} 완료")

    # JSON 라벨 저장
    label_path = LABEL_DIR / "labels.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 이미지: {IMAGE_DIR}")
    print(f"[완료] 라벨:   {label_path}")
    return labels


# ─────────────────────────────────────────────
# 8. 메인 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── DART API로 회사명 수집 ──
    if DART_API_KEY and DART_API_KEY != "YOUR_DART_API_KEY_HERE":
        dart_corps = fetch_dart_companies(DART_API_KEY, max_count=500)
        company_names = [c["name"] for c in dart_corps] if dart_corps else get_fallback_companies()
    else:
        print("[INFO] DART_API_KEY 미설정 → 샘플 회사명 사용")
        company_names = get_fallback_companies()

    print(f"[INFO] 사용할 회사명 {len(company_names)}개")

    # ── 데이터셋 생성 ──
    labels = generate_dataset(
        company_names=company_names,
        n_samples=500,      # ← 원하는 개수로 변경
        augment=True,
    )

    # ── 간단한 통계 출력 ──
    print("\n[샘플 라벨 3건]")
    for l in labels[:3]:
        print(json.dumps(l, ensure_ascii=False, indent=2))
