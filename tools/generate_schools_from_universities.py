import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

NINE_EIGHT_FIVE = {
    "北京大学", "清华大学", "中国人民大学", "北京师范大学", "北京航空航天大学", "北京理工大学",
    "中国农业大学", "中央民族大学", "南开大学", "天津大学", "大连理工大学", "东北大学",
    "吉林大学", "哈尔滨工业大学", "复旦大学", "上海交通大学", "同济大学", "华东师范大学",
    "南京大学", "东南大学", "浙江大学", "中国科学技术大学", "厦门大学", "山东大学",
    "中国海洋大学", "武汉大学", "华中科技大学", "中南大学", "湖南大学", "国防科技大学",
    "中山大学", "华南理工大学", "重庆大学", "四川大学", "电子科技大学", "西安交通大学",
    "西北工业大学", "西北农林科技大学", "兰州大学",
}

TWO_ONE_ONE_EXTRA = {
    "北京科技大学", "北京交通大学", "北京邮电大学", "北京化工大学", "北京工业大学", "北京林业大学",
    "北京中医药大学", "北京外国语大学", "中国传媒大学", "中央财经大学", "对外经济贸易大学",
    "北京体育大学", "中央音乐学院", "中国政法大学", "华北电力大学", "中国矿业大学（北京）",
    "中国石油大学（北京）", "中国地质大学（北京）", "天津医科大学",
    "河北工业大学", "太原理工大学", "内蒙古大学",
    "大连海事大学", "辽宁大学", "东北师范大学", "延边大学", "哈尔滨工程大学",
    "东北林业大学", "东北农业大学", "华东理工大学", "东华大学", "上海财经大学",
    "上海外国语大学", "上海大学", "海军军医大学", "南京航空航天大学", "南京理工大学", "河海大学",
    "南京农业大学", "中国药科大学", "南京师范大学", "苏州大学", "江南大学", "中国矿业大学",
    "合肥工业大学", "安徽大学", "福州大学", "南昌大学",
    "中国石油大学（华东）", "郑州大学", "武汉理工大学", "华中师范大学",
    "华中农业大学", "中南财经政法大学", "中国地质大学（武汉）", "湖南师范大学",
    "暨南大学", "华南师范大学", "广西大学", "海南大学", "西南大学",
    "西南交通大学", "西南财经大学", "四川农业大学",
    "贵州大学", "云南大学", "西藏大学", "西安电子科技大学",
    "陕西师范大学", "西北大学", "长安大学", "空军军医大学", "青海大学", "宁夏大学",
    "新疆大学", "石河子大学",
}

DOUBLE_FIRST_CLASS_EXTRA = {
    "北京协和医学院", "首都师范大学", "外交学院", "中国人民公安大学", "中国音乐学院",
    "中央美术学院", "中央戏剧学院", "中国科学院大学", "天津工业大学", "天津中医药大学",
    "山西大学", "上海海洋大学", "上海中医药大学", "上海体育大学", "上海音乐学院",
    "上海科技大学", "南京邮电大学", "南京信息工程大学", "南京林业大学", "南京医科大学",
    "南京中医药大学", "中国美术学院", "宁波大学", "河南大学", "湘潭大学", "华南农业大学",
    "广州医科大学", "广州中医药大学", "南方科技大学", "西南石油大学", "成都理工大学",
    "成都中医药大学",
}

REGION_TIER = {
    "北京": "一线",
    "上海": "一线",
    "广东": "一线",
    "天津": "二线",
    "重庆": "新一线",
    "江苏": "新一线",
    "浙江": "新一线",
    "湖北": "新一线",
    "四川": "新一线",
    "陕西": "新一线",
    "山东": "二线",
    "福建": "二线",
    "辽宁": "二线",
    "湖南": "二线",
    "河南": "二线",
    "安徽": "二线",
}


def infer_type(name: str) -> str:
    if any(key in name for key in ["医科", "医学院", "中医药", "协和医学院", "军医"]):
        return "医药"
    if "师范" in name:
        return "师范"
    if any(key in name for key in ["财经", "经济贸易", "对外经贸", "工商", "审计", "政法", "外交", "国际关系", "公安", "司法"]):
        return "财经政法"
    if any(key in name for key in ["音乐", "美术", "戏剧", "戏曲", "电影", "艺术", "传媒", "外国语", "语言"]):
        return "语言艺术"
    if "体育" in name:
        return "体育"
    if any(key in name for key in ["农业", "农林", "林业", "海洋"]):
        return "农林海洋"
    if any(key in name for key in ["理工", "工业", "科技", "工程", "交通", "邮电", "电力", "电子", "信息", "航空", "航天", "石油", "地质", "矿业", "建筑", "化工", "民航", "海事"]):
        return "工科"
    if "民族" in name:
        return "民族"
    return "综合"


def infer_level(name: str) -> str:
    if name in NINE_EIGHT_FIVE:
        return "985"
    if name in TWO_ONE_ONE_EXTRA:
        return "211"
    if name in DOUBLE_FIRST_CLASS_EXTRA:
        return "双一流"
    return "普通一本"


def infer_employment_rate(level: str) -> float:
    return {"985": 0.95, "211": 0.91, "双一流": 0.89, "普通一本": 0.84}.get(level, 0.82)


def infer_average_salary(level: str, tier: str, school_type: str) -> int:
    base = {"985": 16000, "211": 12500, "双一流": 11500, "普通一本": 8500}.get(level, 8000)
    base += {"一线": 1800, "新一线": 1000, "二线": 300, "三线": -300, "四线": -800}.get(tier, 0)
    if school_type == "医药":
        base += 800
    elif school_type == "工科":
        base += 700
    elif school_type == "财经政法":
        base += 500
    return max(5500, base)


def parse_universities() -> list[dict]:
    source = (ROOT / "universities_by_region.txt").read_text(encoding="utf-8")
    schools = []
    index = 1

    for block in re.split(r"\n\s*\n", source.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2 or not lines[0].endswith("："):
            continue

        region = lines[0][:-1]
        names = [name.strip() for name in "".join(lines[1:]).rstrip("。").split("、") if name.strip()]
        tier = REGION_TIER.get(region, "二线")

        for name in names:
            level = infer_level(name)
            school_type = infer_type(name)
            tags = [region, level]
            if school_type != "综合":
                tags.append(school_type)

            schools.append({
                "id": f"school_{index:04d}",
                "name": name,
                "province": region,
                "level": level,
                "city": region,
                "tier": tier,
                "type": school_type,
                "employment_rate": infer_employment_rate(level),
                "average_salary": infer_average_salary(level, tier, school_type),
                "tags": tags,
            })
            index += 1

    return schools


if __name__ == "__main__":
    schools = parse_universities()
    target = ROOT / "data" / "schools.json"
    target.write_text(json.dumps(schools, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(schools)} schools to {target}")
