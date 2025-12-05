# crawler.py
import time
import requests
from bs4 import BeautifulSoup
import re
import csv
from pathlib import Path

from config import get_headers, BASE_URL, DISTRICTS, MAX_PAGE_PER_DISTRICT

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CSV_FILE = DATA_DIR / "raw_house_data.csv"


def get_html(url, retries=3, timeout=10):
    """请求网页，返回 HTML 文本"""
    for i in range(retries):
        try:
            resp = requests.get(url, headers=get_headers(), timeout=timeout)
            if resp.status_code == 200:
                resp.encoding = resp.apparent_encoding
                return resp.text
            else:
                print(f"[WARN] 请求失败 {resp.status_code} {url}")
        except Exception as e:
            print(f"[ERROR] 请求异常: {e}")
        time.sleep(1 + i)  # 简单休眠，避免过快
    return None

def parse_list_page(html, district_name):
    """
    解析房天下（二手房）列表页 HTML，返回房源信息 list[dict]
    """
    # 为了调试，把原始 html 存一份下来（只会覆盖，不会越存越多）
    debug_file = Path(__file__).resolve().parent / "data" / f"debug_{district_name}.html"
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DEBUG] 已保存调试 HTML 到: {debug_file}")

    soup = BeautifulSoup(html, "lxml")
    house_list = []

    # 房天下二手房列表结构大致为：
    # <div class="shop_list shop_list_4">
    #   <dl>  ← 每一套房源
    #       <dd class="info rel floatr">
    #           <p class="title">  <span class="tit_shop">标题</span>  </p>
    #           <p class="tel_shop">若干字段：户型 / 面积 / 楼层 / 年代 / 朝向 ...</p>
    #           <p class="add_shop"> <span>位置/小区</span> </p>
    #           <dd class="price_right">
    #               <span class="red">总价</span>
    #               <span>单价</span>
    #           </dd>
    #       </dd>
    #   </dl>
    # </div>

    items = soup.select("div.shop_list.shop_list_4 dl")
    print(f"[DEBUG] dl 条数: {len(items)}")

    if not items:
        print("[WARN] 当前页面未解析到房源 dl，检查是否被反爬或 URL/结构变化")
        return house_list

    for dl in items:
        # 标题
        title_tag = dl.select_one("span.tit_shop")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # 户型 / 面积 / 楼层 / 年代 等信息在 p.tel_shop 里
        tel_p = dl.select_one("p.tel_shop")
        if tel_p:
            # 把里面所有小段文本拼成一行，方便后面用正则抽取面积/楼层/年份
            info_str = " | ".join(s.strip() for s in tel_p.stripped_strings)
        else:
            info_str = ""

        # 地址 / 小区
        addr_tag = dl.select_one("p.add_shop span")
        location = addr_tag.get_text(strip=True) if addr_tag else ""

        # 总价（单位：万）
        price_tag = dl.select_one("dd.price_right span.red")
        total_price_str = price_tag.get_text(strip=True) if price_tag else ""

        # 单价（如 "50000元/㎡"）
        unit_tag = dl.select_one("dd.price_right span:nth-of-type(2)")
        unit_price_str = unit_tag.get_text(strip=True) if unit_tag else ""

        # ===== 利用前面写好的正则函数抽取数值 =====
        area = extract_area(info_str)              # 面积 m²
        floor_level = extract_floor(info_str)      # 楼层信息
        build_year = extract_year(info_str)        # 年份
        total_price = extract_total_price(total_price_str)   # 万
        unit_price = extract_unit_price(unit_price_str)      # 元/㎡

        house_list.append({
            "district": district_name,            # 这里是区的编码（143/13/11），后面可以自己映射成中文
            "title": title,
            "total_price_wan": total_price,
            "unit_price_yuan_m2": unit_price,
            "area_m2": area,
            "floor_info": floor_level,
            "build_year": build_year,
            "location": location,
            "raw_info": info_str,                 # 原始描述信息，方便后续调试
        })

    return house_list




def extract_total_price(text):
    # 如 "500万" -> 500
    if not text:
        return None
    m = re.search(r"([\d\.]+)", text)
    return float(m.group(1)) if m else None


def extract_unit_price(text):
    # 如 "5万/㎡" 或 "50000元/㎡"
    if not text:
        return None
    # 先提取数字
    m = re.search(r"([\d\.]+)", text)
    if not m:
        return None
    value = float(m.group(1))
    # 看单位
    if "万" in text and "㎡" in text:
        # 5万/㎡ -> 50000 元/㎡
        return value * 10000
    else:
        # 默认就是元/㎡
        return value


def extract_area(text):
    if not text:
        return None
    # 从 "3室2厅 | 89㎡ | 高层/18层" 中提取 89
    m = re.search(r"([\d\.]+)\s*㎡", text)
    return float(m.group(1)) if m else None


def extract_floor(text):
    # 简单提取楼层信息，比如 "高层/18层"
    if not text:
        return None
    m = re.search(r"(低层|中层|高层|顶层|地下室|底层|未知|\d+层)", text)
    return m.group(1) if m else None


def extract_year(text):
    # 提取 "2010年建"、"2010年" 这样的年份
    if not text:
        return None
    m = re.search(r"(\d{4})\s*年", text)
    return int(m.group(1)) if m else None

def save_to_csv(rows, file_path, mode="a", write_header=False):
    """
    将房源数据 list[dict] 写入 CSV
    """
    fieldnames = [
        "district", "title",
        "total_price_wan", "unit_price_yuan_m2", "area_m2",
        "floor_info", "build_year", "location", "raw_info",
    ]
    with open(file_path, mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def crawl_district(district):
    """
    爬取某个 district 的多页数据
    """
    all_rows = []
    for page in range(1, MAX_PAGE_PER_DISTRICT + 1):
        url = BASE_URL.format(district=district, page=page)
        print(f"[INFO] 爬取：{url}")
        html = get_html(url)
        if not html:
            print(f"[WARN] {url} 未获取到 HTML，跳过")
            continue

        rows = parse_list_page(html, district_name=district)
        if not rows:
            print(f"[INFO] {url} 未解析到房源，可能翻页结束，停止该区爬取")
            break

        print(f"[INFO] 本页解析到 {len(rows)} 条")
        all_rows.extend(rows)

        # 简单限速，防止过快
        time.sleep(1)

    return all_rows


def main():
    # 如果文件已存在，可以先删或改名，以免重复
    if CSV_FILE.exists():
        print(f"[INFO] 删除已有文件: {CSV_FILE}")
        CSV_FILE.unlink()

    first = True
    total_count = 0

    for dist in DISTRICTS:
        rows = crawl_district(dist)
        print(f"[INFO] 区域 {dist} 共爬到 {len(rows)} 条")
        total_count += len(rows)

        save_to_csv(rows, CSV_FILE, mode="a", write_header=first)
        first = False

    print(f"[DONE] 全部完成，总计 {total_count} 条数据，保存在 {CSV_FILE}")


if __name__ == "__main__":
    main()

