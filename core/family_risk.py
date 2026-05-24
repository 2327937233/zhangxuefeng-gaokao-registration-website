from __future__ import annotations

from typing import Any


RISK_TAG_ORDER = [
    "调剂风险高",
    "专业组混杂",
    "城市就业资源弱",
    "专业出口窄",
    "需要读研",
    "家庭试错成本高",
    "文科就业不确定",
    "医学培养周期长",
    "工科行业波动",
    "学校名气强但专业一般",
]

HUMANITIES_CATEGORIES = {"文学", "历史学", "艺术学"}
SOFT_PATH_CATEGORIES = {"经济学", "管理学", "法学"}
STRONG_LEVELS = {"985", "211", "双一流"}
RESOURCE_TIERS = {"一线", "新一线"}


def normalize_family_level(family_background: str | None) -> str:
    family = str(family_background or "").strip()
    if "富裕" in family or "资源" in family or "有矿" in family:
        return "富裕"
    if "中产" in family:
        return "中产"
    return "普通"


def build_family_risk_profile(
    school: dict[str, Any] | None,
    major: dict[str, Any] | None,
    family_background: str | None = None,
    risk_level: str | None = None,
) -> dict[str, Any]:
    """Build family-aware risk labels for one school-major plan.

    The labels are intentionally conservative: they do not replace official
    admission data, but they force each recommendation to name the hidden
    family-cost risks a parent should verify.
    """
    school = school or {}
    major = major or {}
    family = normalize_family_level(family_background)
    tags = _classify_risk_tags(school, major, family, risk_level)
    return {
        "risk_tags": tags,
        "family_strategy": _family_strategy(family, tags),
        "family_risk_summary": _family_summary(family, tags),
        "family_actions": _family_actions(family, tags),
    }


def _classify_risk_tags(
    school: dict[str, Any],
    major: dict[str, Any],
    family: str,
    risk_level: str | None,
) -> list[str]:
    risk_level = str(risk_level or "")
    school_level = str(school.get("level") or "")
    school_type = str(school.get("type") or "")
    school_tier = str(school.get("tier") or "")
    major_name = str(major.get("name") or "")
    major_category = str(major.get("category") or "")
    major_tags = set(major.get("tags") or [])
    risk_factors = [str(item) for item in major.get("risk_factors") or []]
    employment_rate = major.get("employment_rate")

    tags: list[str] = []

    if risk_level == "冲":
        _add(tags, "调剂风险高")
        if school_level in STRONG_LEVELS or major_category in {"工学", "理学", "管理学", "经济学"}:
            _add(tags, "专业组混杂")

    if school_tier and school_tier not in RESOURCE_TIERS:
        _add(tags, "城市就业资源弱")

    if _is_narrow_exit_major(major_category, major_tags, risk_factors, employment_rate):
        _add(tags, "专业出口窄")

    if major.get("requires_grad_school") or "需深造" in major_tags:
        _add(tags, "需要读研")

    if major_category in HUMANITIES_CATEGORIES or any(key in major_name for key in ["新闻", "传播", "历史", "中文", "汉语言"]):
        _add(tags, "文科就业不确定")

    if major_category == "医学" or "医学" in major_name or school_type == "医药":
        _add(tags, "医学培养周期长")

    if major_category == "工学" and _has_engineering_cycle_risk(major_name, risk_factors, major_tags):
        _add(tags, "工科行业波动")

    if school_level in STRONG_LEVELS and _is_brand_stronger_than_major(school_type, major_category, major_name):
        _add(tags, "学校名气强但专业一般")

    if family == "普通" and _ordinary_family_cost_is_high(tags, major_tags, risk_level, major_category):
        _add(tags, "家庭试错成本高")
    elif family == "中产" and len(tags) >= 3 and ("需要读研" in tags or "调剂风险高" in tags):
        _add(tags, "家庭试错成本高")

    return [tag for tag in RISK_TAG_ORDER if tag in tags]


def _add(tags: list[str], tag: str) -> None:
    if tag not in tags:
        tags.append(tag)


def _is_narrow_exit_major(
    major_category: str,
    major_tags: set[str],
    risk_factors: list[str],
    employment_rate: Any,
) -> bool:
    if {"天坑", "看背景", "转行率高"} & major_tags:
        return True
    if major_category in HUMANITIES_CATEGORIES:
        return True
    if any(any(key in item for key in ["就业面窄", "行业萎缩", "门槛低", "高度依赖人脉"]) for item in risk_factors):
        return True
    try:
        return float(employment_rate) < 0.72
    except (TypeError, ValueError):
        return False


def _has_engineering_cycle_risk(major_name: str, risk_factors: list[str], major_tags: set[str]) -> bool:
    cycle_words = ["行业竞争", "泡沫", "替代", "下行", "波动", "艰苦", "低端岗位"]
    if any(any(word in item for word in cycle_words) for item in risk_factors):
        return True
    if {"传统行业", "下行", "前沿"} & major_tags:
        return True
    return any(key in major_name for key in ["计算机", "人工智能", "软件", "土木", "材料", "化工"])


def _is_brand_stronger_than_major(school_type: str, major_category: str, major_name: str) -> bool:
    if school_type in {"工科"} and major_category in {"工学", "理学", "管理学"}:
        return False
    if school_type == "师范" and major_category in {"教育学", "文学", "历史学", "理学", "法学"}:
        return False
    if school_type == "医药" and major_category == "医学":
        return False
    if school_type == "财经政法" and major_category in {"经济学", "管理学", "法学"}:
        return False
    if school_type == "农林海洋" and major_category in {"农学", "工学", "理学"}:
        return False
    if school_type == "语言艺术" and major_category in {"文学", "艺术学", "教育学"}:
        return False
    if school_type == "综合":
        return major_category in HUMANITIES_CATEGORIES and not any(key in major_name for key in ["汉语言", "法学"])
    return bool(school_type and major_category)


def _ordinary_family_cost_is_high(
    tags: list[str],
    major_tags: set[str],
    risk_level: str,
    major_category: str,
) -> bool:
    high_cost_tags = {"调剂风险高", "专业组混杂", "需要读研", "专业出口窄", "医学培养周期长", "文科就业不确定", "学校名气强但专业一般"}
    if set(tags) & high_cost_tags:
        return True
    if {"看背景", "天坑", "需深造"} & major_tags:
        return True
    return risk_level == "冲" and major_category in SOFT_PATH_CATEGORIES


def _family_strategy(family: str, tags: list[str]) -> str:
    if family == "富裕":
        if "需要读研" in tags or "医学培养周期长" in tags:
            return "资源型家庭可以用深造、城市资源和行业人脉换上限，但要提前算清时间成本。"
        if "调剂风险高" in tags:
            return "资源型家庭可以保留冲高空间，但必须先确认调剂后还能接受。"
        return "资源型家庭可以更看平台和长期上限，同时保留转专业、读研和实习资源。"
    if family == "中产":
        if "家庭试错成本高" in tags:
            return "中产家庭可以有试错，但不要把全部筹码压在高波动或强深造路径上。"
        return "中产家庭适合平台和专业并重，冲高可以有，但稳档要能接住就业出口。"
    if "家庭试错成本高" in tags:
        return "普通家庭优先确定性，先保专业出口、城市实习和调剂底线，再谈学校名气。"
    return "普通家庭可以看这条路，但必须把就业出口、专业组和官方招生规则核验清楚。"


def _family_summary(family: str, tags: list[str]) -> str:
    if not tags:
        return _family_strategy(family, tags)
    tag_text = "、".join(tags[:4])
    if len(tags) > 4:
        tag_text += "等"
    return f"{_family_strategy(family, tags)}本方案需重点盯：{tag_text}。"


def _family_actions(family: str, tags: list[str]) -> list[str]:
    actions_by_tag = {
        "调剂风险高": "先查专业组内所有可调剂专业，不能只看目标专业名称。",
        "专业组混杂": "把同组专业逐个列出来，确认有没有完全不能接受的方向。",
        "城市就业资源弱": "核验当地实习半径、校招企业和毕业生主要去向。",
        "专业出口窄": "看普通毕业生中位路径，不要只看少数头部案例。",
        "需要读研": "提前确认家庭能否承受考研、保研或继续深造的时间成本。",
        "家庭试错成本高": "稳保档必须能接住，不要用全表去赌一个看起来体面的名头。",
        "文科就业不确定": "优先看考编、考公、实习作品和城市岗位，而不是只看兴趣。",
        "医学培养周期长": "算清本科、规培、读研和执业资格周期，再判断家庭现金流。",
        "工科行业波动": "核验课程、项目、实习和行业场景，别只追热门词。",
        "学校名气强但专业一般": "单独查学院资源和培养方案，防止学校牌子盖住专业弱点。",
    }
    actions = [actions_by_tag[tag] for tag in tags if tag in actions_by_tag]
    if family == "富裕" and actions:
        actions.append("如果家庭资源能补实习和深造，可以把它当上限方案，但仍要保留稳妥备选。")
    elif family == "普通" and actions:
        actions.append("普通家庭最后用官方投档表、招生章程和就业质量报告逐项核验。")
    return actions[:4]
