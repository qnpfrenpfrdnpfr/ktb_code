"""
Synthetic Business Card Dataset Generator (KOR / ENG / MIX = 1:1:1)

목표
- OCR fine-tuning용 명함 이미지 + 라벨(labels_train.json) 생성
- "전화번호는 포맷 통일"이 아니라 "숫자 인식"에 초점:
  -> 라벨에 phone_digits(숫자만) + phone_raw(이미지에 찍힌 문자열) 둘 다 저장
- 한국어/영어/혼합 명함을 정확히 1:1:1 비율로 생성
- 실제 데이터(회사명/이름/직함/레이아웃) 수집/크롤링 결과를 넣기 쉽게 설계

설치
pip install pillow numpy opencv-python

실행 예시
python make_cards.py --out /root/business_card_dataset --n 3000

출력 구조
/root/business_card_dataset/
  images/
    card_000000.png
    ...
  labels/
    labels_train.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# =========================
# 0) 유틸
# =========================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def safe_load_lines(txt_path: Optional[str]) -> List[str]:
    if not txt_path:
        return []
    p = Path(txt_path)
    if not p.exists():
        return []
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def weighted_choice(items: List[str], weights: List[float]) -> str:
    return random.choices(items, weights=weights, k=1)[0]


def normalize_company_legal_prefix(name: str) -> str:
    # (주), 주식회사, Co., Ltd. 등은 "원문 그대로" 학습을 위해 여기서는 손대지 않음.
    # 필요시 후처리 단계에서 정규화 권장.
    return name.strip()


def has_korean(text: str) -> bool:
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)


# =========================
# 1) "수집/크롤링 결과"를 꽂기 위한 데이터 로더
# =========================

@dataclass
class RealWorldPools:
    kor_companies: List[str]
    eng_companies: List[str]
    kor_names: List[str]
    eng_names: List[str]
    kor_titles: List[str]
    eng_titles: List[str]
    departments: List[str]  # 한/영 혼용 가능
    layout_templates: List[str]  # "classic_left", "two_column", ...

    @staticmethod
    def from_files(
        kor_company_txt: Optional[str] = None,
        eng_company_txt: Optional[str] = None,
        kor_name_txt: Optional[str] = None,
        eng_name_txt: Optional[str] = None,
        kor_title_txt: Optional[str] = None,
        eng_title_txt: Optional[str] = None,
        dept_txt: Optional[str] = None,
        layout_txt: Optional[str] = None,
    ) -> "RealWorldPools":
        # 파일이 없으면 기본 샘플로 대체
        kor_companies = safe_load_lines(kor_company_txt) or [
            "(주) 삼성전자", "네이버", "카카오", "현대자동차", "LG CNS", "SK텔레콤", "KT",
            "배달의민족", "우아한형제들", "요기요", "쿠팡", "당근", "토스", "카카오뱅크",
            "신한은행", "KB국민은행", "하나은행", "메가커피", "스타벅스코리아", "이디야커피",
            "홍익돈까스", "한림창업투자", "성심당", "미소식당", "중앙정밀", "푸드랩코리아",
        ]
        eng_companies = safe_load_lines(eng_company_txt) or [
            "Samsung Electronics", "NAVER Corp.", "Kakao Corp.", "Hyundai Motor Company",
            "LG CNS", "SK Telecom", "KT Corp.", "Baemin", "Woowa Brothers", "Yogiyo",
            "Coupang", "Toss", "KakaoBank", "Shinhan Bank", "KB Kookmin Bank", "Hana Bank",
            "Starbucks Korea", "EDIYA Coffee", "Seongsimdang Bakery", "Miso Dining",
            "Central Precision", "FoodLab Korea",
        ]
        kor_names = safe_load_lines(kor_name_txt) or [
            "박지수", "김민지", "이서준", "정지수", "최유진", "한지민", "송하늘"
        ]
        eng_names = safe_load_lines(eng_name_txt) or [
            "Jisu Park", "Minji Kim", "Seojun Lee", "Jisoo Jung", "Eugene Choi", "Haneul Song"
        ]
        kor_titles = safe_load_lines(kor_title_txt) or [
            "연구원", "선임연구원", "수석연구원", "매니저", "팀장", "개발자", "AI 엔지니어"
        ]
        eng_titles = safe_load_lines(eng_title_txt) or [
            "Researcher", "Senior Researcher", "Principal Researcher", "Manager", "Team Lead",
            "Software Engineer", "Senior Engineer", "AI Engineer"
        ]
        departments = safe_load_lines(dept_txt) or [
            "AI Lab", "R&D센터", "플랫폼팀", "Business Development", "Security Team", "Data Team",
            "영업팀", "경영지원팀", "기획실", "운영팀", "마케팅팀", "서비스개발팀", "매장운영팀",
        ]
        layout_templates = safe_load_lines(layout_txt) or [
            "classic_left", "two_column", "centered", "minimal_top"
        ]

        return RealWorldPools(
            kor_companies=kor_companies,
            eng_companies=eng_companies,
            kor_names=kor_names,
            eng_names=eng_names,
            kor_titles=kor_titles,
            eng_titles=eng_titles,
            departments=departments,
            layout_templates=layout_templates,
        )


# =========================
# 2) 번호 생성 (숫자 인식에 초점)
# =========================

@dataclass
class PhoneSample:
    kind: str          # "company" | "mobile"
    phone_raw: str     # 이미지에 찍히는 문자열 (포맷 다양)
    phone_digits: str  # 라벨용: 숫자만
    phone_norm: str    # 라벨용: 표준 포맷 (예: 031-2222-3333)


def gen_kor_mobile_digits() -> str:
    # 010XXXXXXXX (총 11자리)
    mid = random.randint(1000, 9999)
    last = random.randint(1000, 9999)
    return f"010{mid:04d}{last:04d}"


def gen_kor_landline_digits() -> str:
    # 02 + 7~8자리 / 지역번호 3자리 + 7~8자리
    area = weighted_choice(["02", "031", "032", "051", "052", "053", "062", "064"], [2,1,1,1,1,1,1,1])
    if area == "02":
        # 02 + 7~8
        if random.random() < 0.5:
            a = random.randint(100, 999)
            b = random.randint(1000, 9999)
            return f"{area}{a:03d}{b:04d}"  # 10자리
        else:
            a = random.randint(1000, 9999)
            b = random.randint(1000, 9999)
            return f"{area}{a:04d}{b:04d}"  # 10자리
    else:
        # 3자리 지역번호 + 7~8
        if random.random() < 0.5:
            a = random.randint(100, 999)
            b = random.randint(1000, 9999)
            return f"{area}{a:03d}{b:04d}"  # 11자리
        else:
            a = random.randint(1000, 9999)
            b = random.randint(1000, 9999)
            return f"{area}{a:04d}{b:04d}"  # 11자리


def gen_service_digits() -> str:
    # 1588/1566/1544/1600 + 4자리
    prefix = random.choice(["1588", "1566", "1544", "1600", "1661"])
    tail = random.randint(1000, 9999)
    return f"{prefix}{tail:04d}"


CONFUSING_DIGIT_POOLS = [
    "23232323", "28282828", "50565656", "10101010", "33322233", "78787878",
    "01023233232", "01028282828", "0212333232", "03123233232"
]

def maybe_use_confusing_digits(base_digits: str) -> str:
    # 일정 비율로 "헷갈리는 숫자 조합"을 강제로 주입
    if random.random() < 0.25:
        cand = random.choice(CONFUSING_DIGIT_POOLS)
        # cand가 010... / 02... 형태면 그대로 쓰고, 아니면 base 길이에 맞춰 자르기
        if cand.isdigit():
            if len(cand) >= len(base_digits):
                return cand[:len(base_digits)]
            else:
                return (cand * (len(base_digits)//len(cand)+1))[:len(base_digits)]
        return base_digits
    return base_digits


def normalize_phone_digits(digits: str) -> str:
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


def format_phone_raw(digits: str, kind: str) -> str:
    """
    digits를 다양한 표기(공백/하이픈/괄호/+82/점/슬래시/내선 등)로 랜덤 렌더링
    라벨은 phone_digits(숫자만)로 평가 가능
    """
    # +82(국가코드) 여부
    use_cc = random.random() < 0.25

    # 구분자 스타일
    sep_style = random.choice(["-", " ", ".", "", "  "])
    wrap_style = random.choice(["none", "paren", "paren_space"])
    if kind == "company":
        prefix_label = random.choice([
            "T.", "T", "Tel.", "TEL", "Tel", "Phone", "Office", "대표번호", ""
        ])
    else:
        prefix_label = random.choice([
            "M.", "M", "Mobile", "Mobile.", "HP", "H.P.", "Cell", "휴대폰", ""
        ])

    # 내선/EXT
    use_ext = (kind == "company") and (random.random() < 0.14)
    ext = str(random.randint(10, 999))

    raw = digits

    # 국가코드 적용 (한국 번호에서만 의미 있게)
    if use_cc:
        # 휴대폰(010...) or 02... or 0xx...
        if raw.startswith("010"):
            raw = "82" + raw[1:]  # 010 -> 10
        elif raw.startswith("02"):
            raw = "82" + raw[1:]  # 02 -> 2
        elif raw.startswith("0") and len(raw) >= 10:
            raw = "82" + raw[1:]

    # 분할 (길이에 따라)
    parts = []
    if len(raw) == 11 and raw.startswith("010"):
        parts = [raw[:3], raw[3:7], raw[7:]]
    elif len(raw) == 10 and raw.startswith("02"):
        # 02-xxxx-xxxx or 02-xxx-xxxx
        if random.random() < 0.5:
            parts = [raw[:2], raw[2:6], raw[6:]]
        else:
            parts = [raw[:2], raw[2:5], raw[5:]]
    elif len(raw) in (10, 11) and raw.startswith("0"):
        # 0xx / 0xxx
        if len(raw) == 11:
            parts = [raw[:3], raw[3:7], raw[7:]]
        else:
            parts = [raw[:3], raw[3:6], raw[6:]]
    elif len(raw) == 8 and raw[:4] in ("1588","1566","1544","1600","1661"):
        parts = [raw[:4], raw[4:]]
    else:
        # fallback: 3-4-나머지
        parts = [raw[:3], raw[3:7], raw[7:]]

    # 괄호 적용
    if wrap_style == "paren":
        parts[0] = f"({parts[0]})"
    elif wrap_style == "paren_space":
        parts[0] = f"({parts[0]}) "

    joined = sep_style.join(parts)

    # +82 표기 적용
    if use_cc:
        plus_variant = random.choice(["+82", "82", "+82)", "+82)", "+82 "])
        if plus_variant in ["+82)", "+82)"]:
            joined = plus_variant + " " + joined
        else:
            joined = plus_variant + sep_style + joined

    # 라벨 붙이기
    if prefix_label:
        joined = f"{prefix_label} {joined}".strip()

    # 내선
    if use_ext:
        ext_style = random.choice([" ext.", " EXT ", " 내선 ", " x", " /"])
        joined = f"{joined}{ext_style}{ext}"

    return joined.strip()


def generate_phone_sample(kind: Optional[str] = None) -> PhoneSample:
    if kind is None:
        kind = weighted_choice(["company", "mobile"], [0.58, 0.42])

    if kind == "mobile":
        digits = gen_kor_mobile_digits()
    else:
        num_type = weighted_choice(["landline", "service"], [0.82, 0.18])
        digits = gen_kor_landline_digits() if num_type == "landline" else gen_service_digits()

    digits = maybe_use_confusing_digits(digits)

    raw = format_phone_raw(digits, kind=kind)
    phone_norm = normalize_phone_digits(digits)

    return PhoneSample(kind=kind, phone_raw=raw, phone_digits=digits, phone_norm=phone_norm)


# =========================
# 3) 레이아웃 템플릿
# =========================

@dataclass
class LayoutSpec:
    name: str
    # 각 요소 배치 영역 (x,y,w,h) - 상대좌표(0~1)
    company_box: Tuple[float, float, float, float]
    name_box: Tuple[float, float, float, float]
    title_box: Tuple[float, float, float, float]
    dept_box: Tuple[float, float, float, float]
    phone_box: Tuple[float, float, float, float]
    email_box: Tuple[float, float, float, float]
    address_box: Tuple[float, float, float, float]


LAYOUTS: Dict[str, LayoutSpec] = {
    "classic_left": LayoutSpec(
        name="classic_left",
        company_box=(0.06, 0.08, 0.88, 0.12),
        name_box=(0.06, 0.24, 0.60, 0.12),
        title_box=(0.06, 0.34, 0.60, 0.08),
        dept_box=(0.06, 0.42, 0.60, 0.08),
        phone_box=(0.06, 0.60, 0.88, 0.08),
        email_box=(0.06, 0.70, 0.88, 0.08),
        address_box=(0.06, 0.80, 0.88, 0.10),
    ),
    "two_column": LayoutSpec(
        name="two_column",
        company_box=(0.06, 0.08, 0.88, 0.12),
        name_box=(0.06, 0.26, 0.42, 0.12),
        title_box=(0.06, 0.38, 0.42, 0.08),
        dept_box=(0.06, 0.46, 0.42, 0.08),
        phone_box=(0.52, 0.28, 0.42, 0.08),
        email_box=(0.52, 0.38, 0.42, 0.08),
        address_box=(0.52, 0.48, 0.42, 0.18),
    ),
    "centered": LayoutSpec(
        name="centered",
        company_box=(0.10, 0.10, 0.80, 0.12),
        name_box=(0.10, 0.28, 0.80, 0.12),
        title_box=(0.10, 0.40, 0.80, 0.08),
        dept_box=(0.10, 0.48, 0.80, 0.08),
        phone_box=(0.10, 0.62, 0.80, 0.08),
        email_box=(0.10, 0.72, 0.80, 0.08),
        address_box=(0.10, 0.82, 0.80, 0.10),
    ),
    "minimal_top": LayoutSpec(
        name="minimal_top",
        company_box=(0.06, 0.08, 0.88, 0.10),
        name_box=(0.06, 0.20, 0.88, 0.10),
        title_box=(0.06, 0.30, 0.88, 0.06),
        dept_box=(0.06, 0.36, 0.88, 0.06),
        phone_box=(0.06, 0.56, 0.88, 0.08),
        email_box=(0.06, 0.66, 0.88, 0.08),
        address_box=(0.06, 0.78, 0.88, 0.12),
    ),
}


# =========================
# 4) 텍스트/명함 내용 생성 (KOR/ENG/MIX)
# =========================

def gen_email(name_eng: str) -> str:
    base = re.sub(r"[^a-zA-Z]", "", name_eng).lower() or "contact"
    domain = random.choice(["corp.com", "company.co.kr", "mail.com", "example.com", "biz.kr"])
    return f"{base[:12]}{random.randint(1,99)}@{domain}"


def gen_address(lang: str) -> str:
    if lang == "KOR":
        return random.choice([
            "서울특별시 강남구 테헤란로 123",
            "경기도 성남시 분당구 판교역로 235",
            "부산광역시 해운대구 센텀서로 45",
            "대전광역시 유성구 대학로 99",
        ])
    if lang == "ENG":
        return random.choice([
            "123 Teheran-ro, Gangnam-gu, Seoul",
            "235 Pangyoyeok-ro, Bundang-gu, Seongnam-si",
            "45 Centumseo-ro, Haeundae-gu, Busan",
            "99 Daehak-ro, Yuseong-gu, Daejeon",
        ])
    # MIX
    return random.choice([
        "서울특별시 강남구 Teheran-ro 123",
        "경기도 성남시 Bundang-gu 판교역로 235",
        "Busan 해운대구 센텀서로 45",
    ])


def sample_card_fields(pools: RealWorldPools, card_type: str) -> Dict[str, str]:
    """
    card_type: "KOR" | "ENG" | "MIX"
    """
    layout_name = random.choice(pools.layout_templates)
    layout_name = layout_name if layout_name in LAYOUTS else "classic_left"

    if card_type == "KOR":
        company = normalize_company_legal_prefix(random.choice(pools.kor_companies))
        name = random.choice(pools.kor_names)
        title = random.choice(pools.kor_titles)
        dept = random.choice(pools.departments)
        # 이메일은 영문이 자연스러우니 eng name을 하나 붙여서 생성
        name_eng = random.choice(pools.eng_names)
        email = gen_email(name_eng)
    elif card_type == "ENG":
        company = normalize_company_legal_prefix(random.choice(pools.eng_companies))
        name = random.choice(pools.eng_names)
        title = random.choice(pools.eng_titles)
        dept = random.choice(pools.departments)
        email = gen_email(name)
    else:  # MIX
        # 혼합 규칙: 회사/이름/직함을 랜덤 혼합
        company = normalize_company_legal_prefix(random.choice(pools.eng_companies if random.random()<0.5 else pools.kor_companies))
        name = random.choice(pools.kor_names if random.random()<0.5 else pools.eng_names)
        title = random.choice(pools.eng_titles if random.random()<0.5 else pools.kor_titles)
        dept = random.choice(pools.departments)
        # 이메일은 영어 기반이 유리
        name_eng = random.choice(pools.eng_names)
        email = gen_email(name_eng)

    phone = generate_phone_sample()
    address = gen_address(card_type)

    company_phone = phone.phone_norm if phone.kind == "company" else ""
    mobile_phone = phone.phone_norm if phone.kind == "mobile" else ""
    company_phone_raw = phone.phone_raw if phone.kind == "company" else ""
    mobile_phone_raw = phone.phone_raw if phone.kind == "mobile" else ""

    return {
        "layout": layout_name,
        "card_type": card_type,
        "company": company,
        "name": name,
        "job_title": title,
        "department": dept,
        "phone_raw": phone.phone_raw,
        "phone_digits": phone.phone_digits,
        "phone_kind": phone.kind,
        "company_phone": company_phone,
        "mobile_phone": mobile_phone,
        "company_phone_raw": company_phone_raw,
        "mobile_phone_raw": mobile_phone_raw,
        "email": email,
        "address": address,
    }


# =========================
# 5) 폰트 로딩 (한글/영문)
# =========================

def load_font(
    font_candidates: List[str],
    size: int,
    *,
    required: bool = False,
    fallback_candidates: Optional[List[str]] = None,
    purpose: str = "text",
) -> ImageFont.FreeTypeFont:
    """
    여러 경로 후보 중 존재하는 폰트를 사용.
    (리눅스 컨테이너에 따라 폰트 경로가 다름)
    """
    candidates = list(font_candidates)
    if fallback_candidates:
        candidates.extend(fallback_candidates)

    for fp in candidates:
        if fp and Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue

    if required:
        raise RuntimeError(
            f"Required font not found for {purpose}. "
            "Install a Korean-capable font (AppleSDGothicNeo/NotoSansCJK/NanumGothic) "
            "or update get_default_font_paths()."
        )

    # fallback: PIL 기본 폰트 (한글 미지원 가능)
    return ImageFont.load_default()


def get_default_font_paths() -> Dict[str, List[str]]:
    # Mac + Linux 경로를 같이 넣어두면 어디서 실행하든 안전
    return {
        "KOR": [
            # macOS
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/System/Library/Fonts/Supplemental/Noto Sans CJK KR.ttc",
            "/System/Library/Fonts/Supplemental/NotoSansCJK-Regular.ttc",
            "/Library/Fonts/AppleGothic.ttf",
            "/Library/Fonts/Noto Sans CJK KR.ttc",
            "/Library/Fonts/NotoSansCJK-Regular.ttc",

            # Linux fallback
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ],
        "KOR_BOLD": [
            # macOS (AppleSDGothicNeo는 weight를 내부에서 처리해서 bold 파일이 따로 없는 경우가 많음)
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/System/Library/Fonts/Supplemental/Noto Sans CJK KR.ttc",
            "/System/Library/Fonts/Supplemental/NotoSansCJK-Bold.ttc",
            "/Library/Fonts/AppleGothic.ttf",
            "/Library/Fonts/Noto Sans CJK KR.ttc",
            "/Library/Fonts/NotoSansCJK-Bold.ttc",

            # Linux fallback
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        ],
        "ENG": [
            # macOS
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",

            # Linux fallback
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ],
        "ENG_BOLD": [
            # macOS
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",

            # Linux fallback
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ],
        "MONO": [
            # macOS
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",

            # Linux fallback
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ],
    }

# =========================
# 6) 렌더링 + 노이즈/저화질/희귀 포맷
# =========================

def draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    box_px: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int],
    align: str = "left",
    max_lines: int = 2,
):
    """
    box_px: (x, y, w, h)
    텍스트가 박스를 넘으면 줄바꿈/축소를 간단히 처리
    """
    x, y, w, h = box_px

    # 간단한 줄바꿈: 공백 기준
    words = text.split(" ")
    lines = []
    cur = ""
    for wd in words:
        nxt = wd if not cur else (cur + " " + wd)
        if draw.textlength(nxt, font=font) <= w:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = wd
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)

    # 세로 중앙 정렬(대충)
    line_h = font.size + 4
    total_h = line_h * len(lines)
    start_y = y + max(0, (h - total_h)//2)

    for i, ln in enumerate(lines):
        if align == "center":
            tx = x + (w - int(draw.textlength(ln, font=font)))//2
        elif align == "right":
            tx = x + w - int(draw.textlength(ln, font=font))
        else:
            tx = x
        ty = start_y + i * line_h
        draw.text((tx, ty), ln, font=font, fill=fill)


def apply_random_background(card_w: int, card_h: int) -> Image.Image:
    # 현실감: 아주 옅은 색 배경, 그라데이션/패턴 약하게
    base = np.zeros((card_h, card_w, 3), dtype=np.uint8)
    color = np.array([
        random.randint(235, 255),
        random.randint(235, 255),
        random.randint(235, 255),
    ], dtype=np.uint8)
    base[:] = color

    # 약한 그라데이션
    if random.random() < 0.4:
        gx = np.linspace(0, random.randint(5, 25), card_w).astype(np.uint8)
        base[:, :, 0] = np.clip(base[:, :, 0] - gx[None, :], 0, 255)
        base[:, :, 1] = np.clip(base[:, :, 1] - gx[None, :], 0, 255)

    return Image.fromarray(base)


def apply_noise_and_degrade(img: Image.Image) -> Image.Image:
    """
    저화질/노이즈/블러/회전/원근/압축 artifact 등
    """
    # PIL -> numpy(BGR)
    arr = np.array(img)[:, :, ::-1].copy()

    h, w = arr.shape[:2]

    # 1) 작은 회전
    if random.random() < 0.35:
        angle = random.uniform(-3.5, 3.5)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        arr = cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 2) 원근 변환(촬영 느낌)
    if random.random() < 0.30:
        margin = random.randint(8, 25)
        pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        pts2 = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w-random.randint(0, margin), random.randint(0, margin)],
            [random.randint(0, margin), h-random.randint(0, margin)],
            [w-random.randint(0, margin), h-random.randint(0, margin)],
        ])
        P = cv2.getPerspectiveTransform(pts1, pts2)
        arr = cv2.warpPerspective(arr, P, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 3) 블러 (out-of-focus)
    if random.random() < 0.35:
        k = random.choice([3, 3, 5])
        arr = cv2.GaussianBlur(arr, (k, k), sigmaX=random.uniform(0.6, 1.6))

    # 4) 흔들림(모션 블러)
    if random.random() < 0.22:
        k = random.choice([5, 7, 9])
        kernel = np.zeros((k, k), dtype=np.float32)
        blur_axis = random.choice(["h", "v", "d1", "d2"])
        if blur_axis == "h":
            kernel[k // 2, :] = 1.0
        elif blur_axis == "v":
            kernel[:, k // 2] = 1.0
        elif blur_axis == "d1":
            np.fill_diagonal(kernel, 1.0)
        else:
            np.fill_diagonal(np.fliplr(kernel), 1.0)
        kernel /= kernel.sum()
        arr = cv2.filter2D(arr, -1, kernel)

    # 5) 거리감(축소 후 배경에 배치)
    if random.random() < 0.32:
        scale = random.uniform(0.55, 0.88)
        nw, nh = max(64, int(w * scale)), max(40, int(h * scale))
        small = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        bg = np.full_like(arr, random.randint(215, 250))
        ox = random.randint(0, w - nw)
        oy = random.randint(0, h - nh)
        bg[oy:oy + nh, ox:ox + nw] = small
        arr = bg

    # 6) 노이즈
    if random.random() < 0.55:
        noise = np.random.normal(0, random.uniform(3, 12), size=arr.shape).astype(np.float32)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 7) JPEG 압축 artifact
    if random.random() < 0.55:
        q = random.randint(35, 90)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        ok, enc = cv2.imencode(".jpg", arr, encode_param)
        if ok:
            arr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 8) 해상도 다운샘플링(저해상도)
    if random.random() < 0.40:
        scale = random.uniform(0.65, 0.95)
        nw, nh = int(w*scale), int(h*scale)
        arr_small = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        arr = cv2.resize(arr_small, (w, h), interpolation=cv2.INTER_CUBIC)

    # BGR -> RGB
    out = Image.fromarray(arr[:, :, ::-1])
    return out


def maybe_add_rare_phone_format(text: str) -> str:
    """
    희귀 포맷: +82) 같은 케이스 / 괄호 꼬임 / 기호 섞임
    """
    if random.random() < 0.15:
        # 괄호/기호를 일부러 애매하게
        text = text.replace("+82", "+82)")
    if random.random() < 0.15:
        text = text.replace("-", " - ")
    if random.random() < 0.10:
        text = text.replace(" ", "")
    return text


def render_card(fields: Dict[str, str], font_paths: Dict[str, List[str]], card_size=(900, 540)) -> Image.Image:
    card_w, card_h = card_size
    img = apply_random_background(card_w, card_h)
    draw = ImageDraw.Draw(img)

    layout = LAYOUTS[fields["layout"]]

    # 폰트 크기 (대략)
    company_size = random.randint(34, 44)
    name_size = random.randint(38, 52)
    small_size = random.randint(22, 28)
    mono_size = random.randint(22, 30)

    # 언어별 폰트 선택
    # (MIX는 요소별로 글자 포함 여부를 보고 대충 고르기)
    def pick_font(text: str, size: int, bold=False) -> ImageFont.FreeTypeFont:
        if has_korean(text):
            key = "KOR_BOLD" if bold else "KOR"
            return load_font(font_paths[key], size=size, required=True, purpose=f"korean-{key}")
        else:
            key = "ENG_BOLD" if bold else "ENG"
            return load_font(font_paths[key], size=size)

    company_font = pick_font(fields["company"], company_size, bold=True)
    name_font = pick_font(fields["name"], name_size, bold=True)
    title_font = pick_font(fields["job_title"], small_size, bold=False)
    dept_font = pick_font(fields["department"], small_size, bold=False)
    email_font = pick_font(fields["email"], small_size, bold=False)
    addr_font = pick_font(fields["address"], small_size, bold=False)
    phone_raw = fields.get("phone_raw_rendered", fields["phone_raw"])
    if has_korean(phone_raw):
        # phone_raw에 "내선" 같은 한글이 들어갈 수 있으므로 한글 지원 폰트를 우선 사용
        phone_font = pick_font(phone_raw, mono_size, bold=False)
    else:
        phone_font = load_font(font_paths["MONO"], size=mono_size, fallback_candidates=font_paths["ENG"], purpose="phone")

    # 색상
    ink = (random.randint(10, 40), random.randint(10, 40), random.randint(10, 40))
    ink2 = (random.randint(40, 80), random.randint(40, 80), random.randint(40, 80))

    def to_px(box_rel):
        x, y, w, h = box_rel
        return (int(x*card_w), int(y*card_h), int(w*card_w), int(h*card_h))

    # 회사 / 이름
    draw_text_in_box(draw, to_px(layout.company_box), fields["company"], company_font, ink, align="left", max_lines=2)
    draw_text_in_box(draw, to_px(layout.name_box), fields["name"], name_font, ink, align="left", max_lines=1)

    # 직함 / 부서
    draw_text_in_box(draw, to_px(layout.title_box), fields["job_title"], title_font, ink2, align="left", max_lines=1)
    draw_text_in_box(draw, to_px(layout.dept_box), fields["department"], dept_font, ink2, align="left", max_lines=1)

    # 전화번호 (희귀 포맷 살짝 추가)
    draw_text_in_box(draw, to_px(layout.phone_box), phone_raw, phone_font, ink, align="left", max_lines=1)

    # 이메일 / 주소
    draw_text_in_box(draw, to_px(layout.email_box), fields["email"], email_font, ink2, align="left", max_lines=1)
    draw_text_in_box(draw, to_px(layout.address_box), fields["address"], addr_font, ink2, align="left", max_lines=2)

    # 아주 약한 라인/장식 (레이아웃 다양성)
    if random.random() < 0.35:
        x0, y0, w0, h0 = to_px(layout.company_box)
        yline = y0 + h0 + random.randint(6, 14)
        draw.line([(int(0.06*card_w), yline), (int(0.94*card_w), yline)], fill=(200, 200, 200), width=1)

    return img


# =========================
# 7) 메인 생성 루프 (1:1:1)
# =========================

def build_balanced_types(n: int) -> List[str]:
    """
    정확히 1:1:1에 가깝게.
    n이 3으로 안 나눠지면 앞에서부터 채움.
    """
    base = ["KOR", "ENG", "MIX"]
    out = []
    q, r = divmod(n, 3)
    out.extend(base * q)
    out.extend(base[:r])
    random.shuffle(out)
    return out


def generate_dataset(
    out_root: Path,
    n: int,
    seed: int,
    pools: RealWorldPools,
    card_size=(900, 540),
):
    seed_everything(seed)

    img_dir = out_root / "images"
    lbl_dir = out_root / "labels"
    ensure_dir(img_dir)
    ensure_dir(lbl_dir)

    font_paths = get_default_font_paths()

    card_types = build_balanced_types(n)

    labels: List[Dict] = []
    for i in range(n):
        ctype = card_types[i]
        fields = sample_card_fields(pools, ctype)
        rendered_phone_raw = maybe_add_rare_phone_format(fields["phone_raw"])
        fields["phone_raw_rendered"] = rendered_phone_raw
        fields["company_phone_raw_rendered"] = rendered_phone_raw if fields["phone_kind"] == "company" else ""
        fields["mobile_phone_raw_rendered"] = rendered_phone_raw if fields["phone_kind"] == "mobile" else ""

        # 렌더링
        img = render_card(fields, font_paths, card_size=card_size)
        # 노이즈/저화질
        img = apply_noise_and_degrade(img)

        fname = f"card_{i:06d}.png"
        fpath = img_dir / fname
        img.save(fpath)

        # 라벨 (중요: phone_digits를 별도로 저장)
        # -> 학습 때 "숫자 인식"을 강화하고 싶으면 phone_digits를 정답으로 사용
        labels.append({
            "id": f"card_{i:06d}",
            "image": f"images/{fname}",
            "card_type": fields["card_type"],           # KOR/ENG/MIX
            "layout": fields["layout"],                # 레이아웃 타입
            "company": fields["company"],
            "name": fields["name"],
            "job_title": fields["job_title"],
            "department": fields["department"],
            "email": fields["email"],
            "address": fields["address"],
            "phone_raw": fields["phone_raw_rendered"],         # 이미지에 실제 렌더링된 문자열
            "phone_digits": fields["phone_digits"],            # 정답(숫자만)
            "phone_kind": fields["phone_kind"],                # company / mobile
            "company_phone_raw": fields["company_phone_raw_rendered"],
            "mobile_phone_raw": fields["mobile_phone_raw_rendered"],
            "company_phone": fields["company_phone"],          # 표준 포맷
            "mobile_phone": fields["mobile_phone"],            # 표준 포맷
        })

    (lbl_dir / "labels_train.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    # 생성 요약
    stats = {"KOR": 0, "ENG": 0, "MIX": 0}
    for t in card_types:
        stats[t] += 1
    print("[DONE] saved:", out_root)
    print("[STATS]", stats)
    print("[LABEL]", lbl_dir / "labels_train.json")


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    default_out = str((Path(__file__).resolve().parent / "LLaMA-Factory/data/business_card_dataset").resolve())
    ap.add_argument(
        "--out",
        type=str,
        default=default_out,
        help=f"output root dir (default: {default_out})",
    )
    ap.add_argument("--n", type=int, default=3000, help="num cards")
    ap.add_argument("--seed", type=int, default=42)

    # 수집/크롤링 결과를 파일로 넣고 싶으면 여기 지정
    ap.add_argument("--kor_company_txt", type=str, default=None)
    ap.add_argument("--eng_company_txt", type=str, default=None)
    ap.add_argument("--kor_name_txt", type=str, default=None)
    ap.add_argument("--eng_name_txt", type=str, default=None)
    ap.add_argument("--kor_title_txt", type=str, default=None)
    ap.add_argument("--eng_title_txt", type=str, default=None)
    ap.add_argument("--dept_txt", type=str, default=None)
    ap.add_argument("--layout_txt", type=str, default=None)

    args = ap.parse_args()

    pools = RealWorldPools.from_files(
        kor_company_txt=args.kor_company_txt,
        eng_company_txt=args.eng_company_txt,
        kor_name_txt=args.kor_name_txt,
        eng_name_txt=args.eng_name_txt,
        kor_title_txt=args.kor_title_txt,
        eng_title_txt=args.eng_title_txt,
        dept_txt=args.dept_txt,
        layout_txt=args.layout_txt,
    )

    generate_dataset(
        out_root=Path(args.out).expanduser().resolve(),
        n=args.n,
        seed=args.seed,
        pools=pools,
        card_size=(900, 540),
    )


if __name__ == "__main__":
    main()
