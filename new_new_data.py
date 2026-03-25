import os
import json
import random
import string
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

NUM_DATA = 1000

OUTPUT_DIR = Path("dataset")
IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_FILE = OUTPUT_DIR / "dataset.jsonl"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    os.environ.get("CARD_FONT", ""),
    # macOS
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size):
    for font_path in FONT_CANDIDATES:
        if not font_path:
            continue
        p = Path(font_path)
        if not p.exists():
            continue
        try:
            return ImageFont.truetype(str(p), size)
        except OSError:
            continue
    return ImageFont.load_default()

kor_first = ["지수","민준","서연","하늘","지민","수빈","지호","가은"]
kor_last = ["김","이","박","최","정","윤","조","장"]

eng_first = ["James","Emily","Daniel","Sophia","Ryan","Olivia"]
eng_last = ["Kim","Lee","Park","Choi","Smith","Brown"]

companies = [
"한빛테크","미래소프트","에이아이솔루션즈","네오플랫폼",
"BluePeak Tech","NovaWorks","VisionBridge","FutureLink"
]

jobs_kor = ["대표","이사","부장","매니저","개발자","연구원"]
jobs_eng = ["CEO","Director","Manager","Engineer","Researcher"]

departments = ["플랫폼팀","개발팀","연구소","AI팀",""]

domains = ["gmail.com","company.co.kr","corp.com","naver.com"]

def rand_kor_name():
    return random.choice(kor_last)+random.choice(kor_first)

def rand_eng_name():
    return random.choice(eng_first)+" "+random.choice(eng_last)

def rand_email(name):
    name = name.lower().replace(" ","")
    return f"{name}{random.randint(1,99)}@{random.choice(domains)}"

def rand_mobile():
    return f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}"

def rand_company_phone():
    return f"{random.choice(['02','031','032','051'])}-{random.randint(100,9999)}-{random.randint(1000,9999)}"


def draw_card(data):

    img = Image.new("RGB",(1024,640),(255,255,255))
    draw = ImageDraw.Draw(img)

    font_big = load_font(60)
    font_mid = load_font(32)
    font_small = load_font(28)

    draw.text((80,80),data["company"],font=font_mid,fill=(0,0,0))
    draw.text((80,200),data["name"],font=font_big,fill=(0,0,0))
    draw.text((80,300),data["job_title"],font=font_mid,fill=(0,0,0))
    draw.text((80,350),data["department"],font=font_mid,fill=(0,0,0))

    draw.text((80,450),"Tel "+data["company_phone"],font=font_small,fill=(0,0,0))
    draw.text((80,500),"Mobile "+data["mobile_phone"],font=font_small,fill=(0,0,0))
    draw.text((80,550),"E "+data["email"],font=font_small,fill=(0,0,0))

    return img


def augment(img):

    if random.random()<0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,2)))

    if random.random()<0.5:
        angle = random.uniform(-15,15)
        img = img.rotate(angle,expand=False,fillcolor=(255,255,255))

    if random.random()<0.5:
        arr = np.array(img)
        noise = np.random.normal(0,10,arr.shape)
        arr = np.clip(arr+noise,0,255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


with open(LABEL_FILE,"w",encoding="utf-8") as f:

    for i in range(NUM_DATA):

        lang_type = random.choice(["kor","eng","mix"])

        if lang_type=="kor":
            name = rand_kor_name()
            job = random.choice(jobs_kor)
        elif lang_type=="eng":
            name = rand_eng_name()
            job = random.choice(jobs_eng)
        else:
            name = rand_kor_name()+" / "+rand_eng_name()
            job = random.choice(jobs_kor)+" / "+random.choice(jobs_eng)

        data = {
            "name":name,
            "company":random.choice(companies),
            "job_title":job,
            "department":random.choice(departments),
            "email":rand_email(name),
            "company_phone":rand_company_phone(),
            "mobile_phone":rand_mobile()
        }

        img = draw_card(data)
        img = augment(img)

        image_path = IMAGE_DIR / f"card_{i:06}.png"
        img.save(image_path)

        label = {
            "id":f"card_{i:06}",
            "image":f"images/card_{i:06}.png",
            "messages":[
                {
                    "role":"user",
                    "content":"<image>\n이 이미지가 명함인지 판단하고 명함이라면 정보를 JSON으로 추출해줘."
                },
                {
                    "role":"assistant",
                    "content":{
                        "is_business_card":True,
                        "name":data["name"],
                        "company":data["company"],
                        "job_title":data["job_title"],
                        "department":data["department"],
                        "email":data["email"],
                        "company_phone":data["company_phone"],
                        "mobile_phone":data["mobile_phone"]
                    }
                }
            ]
        }

        f.write(json.dumps(label,ensure_ascii=False)+"\n")

print("완료: 1000개 생성")
