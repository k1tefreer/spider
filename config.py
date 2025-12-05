# config.py
import random

# User-Agent（保持即可）
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://esf.fang.com/",   # 加上更像真实用户
    }

# --------------------- 关键修改在这里 ---------------------

# 房天下北京二手房 URL 格式：
# https://esf.fang.com/house-a0{district}/i3{page}/
# 例子：海淀是 a0143
BASE_URL = "https://cq.esf.fang.com/house-a0{district}/i3{page}/"

DISTRICTS = [
    ("56", "渝中区"),
    ("58", "两江新区"),
    ("59", "南岸区"),
    ("60", "沙坪坝区"),
    ("61", "九龙坡区"),
    ("62", "大渡口区"),
    ("63", "北碚区"),
    ("11841", "合川区"),
    ("11830", "彭水县"),
]

MAX_PAGE_PER_DISTRICT = 100   # 越大越多

