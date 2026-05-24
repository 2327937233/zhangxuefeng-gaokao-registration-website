import json
import re
from typing import List, Optional

from core.models import (
    UserPreferences, PlanOption, RecommendResponse, RecommendRequest,
    CompareRequest, CompareResult, InsightRequest, InsightResponse,
    PressureTestRequest, PressureTestResponse, AnalyzeRequest, AnalyzeResponse,
    ThinkingStep, UserProfile,
)
from core.family_risk import build_family_risk_profile
from core.zxf_engine import data_source, ZXFEvaluator
from core.llm_client import llm_client

evaluator = ZXFEvaluator()


class AgentEngine:
    """zhiyuan-agent 后端 Agent 引擎：融合 LLM + 规则引擎 + 数据检索"""

    # 冲稳保概率区间（按家庭策略动态覆盖）
    PROB_RANGES = {
        "冲": (55, 72),
        "稳": (78, 90),
        "保": (92, 98),
    }

    # 风险偏好策略：由考生画像的 risk_appetite 决定冲稳保比例和阈值
    RISK_STRATEGIES = {
        "激进": {
            "label": "激进型",
            "description": "追求上限，可承受更高风险，冲的比例更大",
            "ratio": {"冲": 0.35, "稳": 0.45, "保": 0.20},
            "prob_ranges": {"冲": (45, 70), "稳": (72, 88), "保": (90, 98)},
            "risk_threshold_offset": -10,
            "flag_focus": "提醒不要过于激进而忽略专业质量",
        },
        "均衡": {
            "label": "均衡型",
            "description": "攻守平衡，兼顾学校层次和录取确定性",
            "ratio": {"冲": 0.25, "稳": 0.50, "保": 0.25},
            "prob_ranges": {"冲": (55, 72), "稳": (78, 90), "保": (92, 98)},
            "risk_threshold_offset": 0,
            "flag_focus": "提醒稳档质量，避免冲高学校被调剂到差专业",
        },
        "稳妥": {
            "label": "保守型",
            "description": "确保下限，优先录取确定性，保底要扎实",
            "ratio": {"冲": 0.15, "稳": 0.45, "保": 0.40},
            "prob_ranges": {"冲": (62, 76), "稳": (80, 92), "保": (94, 99)},
            "risk_threshold_offset": +8,
            "flag_focus": "提醒保底必须足够稳固，避免滑档",
        },
    }

    # 家庭内容偏好：由 family_background 决定推荐内容的倾向性（选什么）
    FAMILY_CONTENT_BIAS = {
        "富裕": {
            "label": "资源导向",
            "description": "可承受深造周期，可借助家庭资源进入门槛型行业",
            # 专业标签加分（这些标签的专业更适合富裕家庭）
            "major_tag_bonus": {"看背景": 20, "管理": 15, "金融": 15, "需深造": 8},
            # 专业标签减分（这些标签的专业对富裕家庭不是最优）
            "major_tag_penalty": {},
            # 学校类型偏好
            "school_type_bonus": {"财经": 10, "政法": 10, "综合": 5},
            # 城市层级权重上浮（更看重一线/新一线平台资源）
            "city_tier_bonus": {"一线": 8, "新一线": 5},
            # 深造容忍度：对 requires_grad_school 的扣分减少
            "grad_school_penalty": 0,
            # 天坑容忍度
            "tiankeng_penalty": 0,
            # 就业率权重（富裕家庭不那么看重）
            "employment_weight": 0.8,
        },
        "中产": {
            "label": "平衡导向",
            "description": "兼顾技术壁垒和就业确定性，追求性价比",
            "major_tag_bonus": {"技术壁垒": 10, "热门": 5, "稳定": 5},
            "major_tag_penalty": {"看背景": -10},
            "school_type_bonus": {"理工": 8, "工科": 8, "综合": 5},
            "city_tier_bonus": {"一线": 5, "新一线": 5},
            "grad_school_penalty": -5,
            "tiankeng_penalty": -15,
            "employment_weight": 1.0,
        },
        "普通": {
            "label": "生存导向",
            "description": "优先本科可变现、就业率高、不依赖家庭背景的专业",
            "major_tag_bonus": {"技术壁垒": 15, "热门": 10, "稳定": 10, "本科可就业": 15},
            "major_tag_penalty": {"看背景": -30, "天坑": -40, "需深造": -20},
            "school_type_bonus": {"理工": 12, "工科": 12, "师范": 8},
            "city_tier_bonus": {},
            "grad_school_penalty": -20,
            "tiankeng_penalty": -35,
            "employment_weight": 1.5,
        },
    }

    # 学校层次到分数映射（模拟）
    SCHOOL_LEVELS = ["985", "211", "双一流", "普通一本", "普通二本"]

    ELITE_985 = {
        "北京大学", "清华大学", "复旦大学", "上海交通大学", "浙江大学", "中国科学技术大学",
        "南京大学", "中国人民大学", "北京航空航天大学", "同济大学", "北京理工大学",
        "哈尔滨工业大学", "西安交通大学", "华中科技大学", "武汉大学", "中山大学",
    }

    CITY_PROVINCE_HINTS = {
        "青岛": "山东",
        "济南": "山东",
        "南京": "江苏",
        "苏州": "江苏",
        "深圳": "广东",
        "广州": "广东",
        "武汉": "湖北",
        "成都": "四川",
        "杭州": "浙江",
    }

    PROVINCE_NAMES = {
        "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海", "江苏", "浙江",
        "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "广西", "海南", "重庆",
        "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
    }

    CITY_SCHOOL_HINTS = {
        "西安": [
            "西安交通大学", "西北工业大学", "西安电子科技大学", "陕西师范大学", "长安大学",
            "西安建筑科技大学", "西安理工大学", "西安科技大学", "西安工业大学", "西安石油大学",
            "西安工程大学", "陕西科技大学", "西安邮电大学", "西安财经大学",
        ],
        "长沙": [
            "中南大学", "湖南大学", "湖南师范大学", "长沙理工大学", "湖南农业大学",
            "湖南中医药大学", "湖南工商大学", "长沙学院", "湖南第一师范学院",
        ],
    }

    REGION_GROUP_ALIASES = {
        "南方": ["上海", "江苏", "浙江", "福建", "广东", "湖北", "湖南", "重庆", "四川"],
        "南方城市": ["上海", "南京", "苏州", "杭州", "宁波", "广州", "深圳", "武汉", "成都", "重庆"],
        "江浙沪": ["上海", "江苏", "浙江"],
        "长三角": ["上海", "江苏", "浙江", "安徽"],
        "华东": ["上海", "江苏", "浙江", "安徽", "福建", "江西"],
        "华南": ["广东", "广西", "海南"],
        "珠三角": ["广州", "深圳", "佛山", "东莞"],
        "西南": ["重庆", "四川", "贵州", "云南"],
    }

    RELATED_MAJOR_GROUPS = {
        "汉语言文学": ["小学教育", "教育学", "历史学", "英语", "法学", "网络与新媒体"],
        "历史学": ["汉语言文学", "小学教育", "教育学", "地理科学", "法学"],
        "地理科学": ["历史学", "汉语言文学", "小学教育", "教育学"],
        "英语": ["汉语言文学", "小学教育", "教育学", "网络与新媒体"],
        "新闻学": ["网络与新媒体", "汉语言文学", "广告学", "英语"],
        "网络与新媒体": ["新闻学", "汉语言文学", "广告学"],
        "法学": ["知识产权", "社会学", "汉语言文学", "思想政治教育"],
        "金融学": ["经济学", "会计学", "财务管理", "工商管理"],
        "计算机科学与技术": ["软件工程", "数据科学与大数据技术", "信息安全", "人工智能"],
        "软件工程": ["计算机科学与技术", "数据科学与大数据技术", "信息安全"],
        "电子信息工程": ["通信工程", "自动化", "电气工程及其自动化", "计算机科学与技术"],
        "临床医学": ["护理学", "药学", "动物医学"],
    }

    def __init__(self):
        self.data = data_source

    # ============================================================
    # 1. 智能志愿推荐
    # ============================================================
    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        """基于用户画像生成冲稳保志愿方案"""
        user = request.user
        limit = request.limit

        normalized_city_preference = self._normalize_location_preferences(user.city_preference)
        if normalized_city_preference != (user.city_preference or []):
            user = user.model_copy(update={"city_preference": normalized_city_preference or None})

        # 1. 规则引擎筛选候选
        candidates = self._filter_candidates(user)

        # 2. 按冲稳保分层
        chong, wen, bao = self._stratify_candidates(candidates, user, limit)

        # 3. 组装方案（按风险偏好动态分配冲稳保比例）
        strategy = self._get_risk_strategy(user.risk_appetite)
        plans = []
        order = 1

        # 冲
        for c in chong:
            plans.append(self._build_plan_option(c, user, "冲", order))
            order += 1

        # 稳
        for c in wen:
            plans.append(self._build_plan_option(c, user, "稳", order))
            order += 1

        # 保
        for c in bao:
            plans.append(self._build_plan_option(c, user, "保", order))
            order += 1

        plans = self._repair_plan_risk_order(plans, user)
        used_zero_candidate_fallback = False
        if not plans:
            plans = self._build_zero_candidate_fallback_plans(user, limit)
            used_zero_candidate_fallback = bool(plans)
        plans = plans[:limit]  # 确保不超过用户请求的数量

        # 4. 生成分析摘要
        summary = self._generate_recommend_summary(plans, user, used_zero_candidate_fallback)

        # 5. 红旗信号
        red_flags = self._collect_recommend_flags(plans, user)
        if user.major_preference and not plans:
            red_flags.append(
                f"按「{'、'.join(user.major_preference or [])}」专业方向没有筛出足够候选，已保留专业硬约束；请补充招生专业目录或由用户明确切换专业后再重算。"
            )
        if user.city_preference and not plans:
            red_flags.append(
                f"按「{'、'.join(user.city_preference or [])}」地区偏好没有筛出足够候选，已保留地区硬约束；请补充目标地区招生计划或放宽城市后再重算。"
            )
        if used_zero_candidate_fallback:
            red_flags.append(
                "原始硬约束没有形成可用冲稳保方案，已按三条替代路径生成可同步方案：放宽城市但保专业、保城市但换相近专业、保稳妥但降低学校层次。"
            )

        # 6. LLM 增强分析（如果可用）
        thinking = self._llm_enhance_recommend(plans, user)

        return RecommendResponse(
            plans=plans,
            summary=summary,
            chong_count=sum(1 for plan in plans if plan.risk_level == "冲"),
            wen_count=sum(1 for plan in plans if plan.risk_level == "稳"),
            bao_count=sum(1 for plan in plans if plan.risk_level == "保"),
            thinking_process=thinking,
            red_flags=red_flags,
        )

    def _normalize_location_preferences(self, preferences: Optional[List[str]]) -> List[str]:
        """把“上海北京”这类粘连输入拆成可匹配的地区偏好。"""
        if not preferences:
            return []

        known_locations = sorted(
            set(self.PROVINCE_NAMES) | set(self.CITY_PROVINCE_HINTS.keys()) | set(self.CITY_SCHOOL_HINTS.keys()),
            key=len,
            reverse=True,
        )
        normalized: List[str] = []

        def add_location(location: str) -> None:
            value = location.removesuffix("省").removesuffix("市").removesuffix("自治区")
            if value and value not in normalized:
                normalized.append(value)

        for preference in preferences:
            raw = str(preference or "").strip()
            if not raw:
                continue
            pieces = [p for p in re.split(r"[、,，\s/]+", raw) if p]
            for piece in pieces:
                value = piece.removesuffix("省").removesuffix("市").removesuffix("自治区")
                alias_hits = [
                    (value.find(alias), alias)
                    for alias in self.REGION_GROUP_ALIASES
                    if alias in value
                ]
                if alias_hits:
                    for _, alias in sorted(alias_hits, key=lambda item: item[0]):
                        for location in self.REGION_GROUP_ALIASES[alias]:
                            add_location(location)
                    continue
                hits = [
                    (value.find(location), location)
                    for location in known_locations
                    if location in value and value != location
                ]
                if hits:
                    for _, location in sorted(hits, key=lambda item: item[0]):
                        add_location(location)
                    continue
                add_location(value)
        return normalized

    def _repair_plan_risk_order(self, plans: List[PlanOption], user: UserPreferences) -> List[PlanOption]:
        """最终兜底：同批方案里更难的院校专业组合优先进入更高风险档。"""
        if len(plans) < 2:
            return plans

        school_by_name = {s["name"]: s for s in self.data.schools.values()}
        major_by_name = {m["name"]: m for m in self.data.majors.values()}
        strategy = self._get_risk_strategy(user.risk_appetite)
        total = len(plans)
        chong_count = max(1, round(total * strategy["ratio"]["冲"]))
        bao_count = max(1, round(total * strategy["ratio"]["保"]))
        if chong_count + bao_count >= total:
            chong_count = max(1, min(chong_count, total - 2))
            bao_count = max(1, total - chong_count - 1)
        wen_count = max(1, total - chong_count - bao_count)
        target_risks = ["冲"] * chong_count + ["稳"] * wen_count + ["保"] * bao_count

        ranked = sorted(
            plans,
            key=lambda plan: -self._admission_difficulty_score(
                school_by_name.get(plan.school, {"name": plan.school}),
                major_by_name.get(plan.major, {"name": plan.major}),
            ),
        )

        repaired: List[PlanOption] = []
        for index, plan in enumerate(ranked, start=1):
            new_risk = target_risks[index - 1] if index - 1 < len(target_risks) else plan.risk_level
            school = school_by_name.get(plan.school, {"name": plan.school})
            major = major_by_name.get(plan.major, {"name": plan.major})
            combo = {"school": school, "major": major, "match_score": 0}
            updates = {"order": index, "risk_level": new_risk}
            if new_risk != plan.risk_level:
                family_risk = build_family_risk_profile(school, major, user.family_background, new_risk)
                updates["probability"] = self._estimate_probability(combo, user, new_risk)
                updates["reason"] = self._generate_reason(school, major, user, new_risk, family_risk)
                updates["risk_tags"] = family_risk["risk_tags"]
                updates["family_strategy"] = family_risk["family_strategy"]
                updates["family_risk_summary"] = family_risk["family_risk_summary"]
                if family_risk["risk_tags"]:
                    updates["risk_warning"] = f"{plan.school}-{plan.major}需重点核验：{'、'.join(family_risk['risk_tags'][:3])}"
            repaired.append(plan.model_copy(update=updates))
        return repaired

    def _filter_candidates(self, user: UserPreferences) -> List[dict]:
        """规则引擎：根据分数和城市偏好筛选候选院校+专业组合"""
        results = []
        schools = list(self.data.schools.values())
        majors = list(self.data.majors.values())

        for school in schools:
            if self._is_military_school(school) and not user.allow_military_schools:
                continue
            if not self._is_school_reasonable_for_profile(school, user):
                continue

            # 城市过滤
            if user.city_preference and not self._matches_location_preference(school, user.city_preference):
                continue

            for major in majors:
                if not self._is_subject_compatible(major, user):
                    continue
                if not self._is_school_major_compatible(school, major):
                    continue
                if self._is_obvious_score_waste(school, major, user):
                    continue

                # 专业方向过滤
                if user.major_preference:
                    if not self._matches_major_preference(major, user.major_preference):
                        continue

                combo = {
                    "school": school,
                    "major": major,
                    "match_score": self._calculate_match_score(school, major, user),
                }
                results.append(combo)

        # 按匹配度排序
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return self._diversify_by_city_preference(results, user)

    def _is_military_school(self, school: dict) -> bool:
        name = school.get("name", "")
        return any(key in name for key in ["国防", "军医", "陆军", "海军", "空军", "火箭军", "武警", "部队"])

    def _is_obvious_score_waste(self, school: dict, major: dict, user: UserPreferences) -> bool:
        """过滤高分画像下明显浪费分数的院校组合。"""
        level = school.get("level", "")
        name = school.get("name", "")
        school_type = school.get("type", "")
        major_name = major.get("name", "")
        major_category = major.get("category", "")
        difficulty = self._admission_difficulty_score(school, major)
        rank = user.rank
        score = user.score or 0

        high_profile = (rank is not None and rank <= 25000) or score >= 610
        if not high_profile:
            return False

        if level == "普通二本":
            return True

        if level == "普通一本" and difficulty < 64:
            if major_category in ["经济学", "医学", "教育学", "农学", "艺术学"] and user.city_preference:
                return False
            return True

        if (
            level == "普通一本"
            and school_type in ["农林海洋", "师范", "民族"]
            and self._is_hot_engineering_major(major_name)
            and not any(key in name for key in ["邮电", "电子", "信息", "科技", "理工", "工业"])
        ):
            return True

        return False

    def _matches_location_preference(self, school: dict, preferences: list[str]) -> bool:
        province = school.get("province", "")
        city = school.get("city", "")
        tier = school.get("tier", "")
        name = school.get("name", "")
        tags = school.get("tags", [])
        for pref in preferences:
            city_school_names = self.CITY_SCHOOL_HINTS.get(pref, [])
            if self._is_province_preference(pref):
                if pref == province or pref in tags:
                    return True
                continue
            if pref in tier:
                return True
            if pref == city or pref in city or pref in name or pref in tags:
                return True
            if name in city_school_names:
                return True
        return False

    def _matched_location_key(self, school: dict, preferences: list[str]) -> str:
        province = school.get("province", "")
        city = school.get("city", "")
        name = school.get("name", "")
        tags = school.get("tags", [])
        for pref in preferences:
            city_school_names = self.CITY_SCHOOL_HINTS.get(pref, [])
            if self._is_province_preference(pref):
                if pref == province or pref in tags:
                    return pref
                continue
            if pref == city or pref in city or pref in name or pref in tags:
                return pref
            if name in city_school_names:
                return pref
        return ""

    def _is_province_preference(self, preference: str) -> bool:
        value = str(preference or "").strip()
        value = value.removesuffix("省").removesuffix("市").removesuffix("自治区")
        return value in self.PROVINCE_NAMES

    def _diversify_by_city_preference(self, candidates: List[dict], user: UserPreferences) -> List[dict]:
        prefs = user.city_preference or []
        if len(prefs) < 2 or not candidates:
            return candidates

        buckets: dict[str, list[dict]] = {pref: [] for pref in prefs}
        others: list[dict] = []
        seen_ids = set()
        for item in candidates:
            key = self._matched_location_key(item["school"], prefs)
            if key and key in buckets:
                buckets[key].append(item)
            else:
                others.append(item)

        diversified: list[dict] = []
        while True:
            added = False
            for pref in prefs:
                bucket = buckets.get(pref) or []
                while bucket:
                    item = bucket.pop(0)
                    item_id = (item["school"].get("name"), item["major"].get("name"))
                    if item_id in seen_ids:
                        continue
                    diversified.append(item)
                    seen_ids.add(item_id)
                    added = True
                    break
            if not added:
                break

        for item in others:
            item_id = (item["school"].get("name"), item["major"].get("name"))
            if item_id not in seen_ids:
                diversified.append(item)
                seen_ids.add(item_id)
        return diversified or candidates

    def _is_subject_compatible(self, major: dict, user: UserPreferences) -> bool:
        subjects = user.subjects or ""
        category = major.get("category", "")
        name = major.get("name", "")
        tags = major.get("tags", [])

        humanities_subjects = any(key in subjects for key in ["政史地", "史政地", "历史", "地理", "政治"])
        has_science_gate = any(key in subjects for key in ["物", "化", "生"])

        if humanities_subjects and not has_science_gate:
            if category in ["工学", "医学"] or name in ["计算机科学与技术", "人工智能", "电子信息工程", "临床医学", "生物科学", "土木工程", "数学与应用数学"]:
                return False
            if "政史地适配" in tags or category in ["文学", "历史学", "法学", "经济学", "教育学"]:
                return True

        return True

    def _matches_major_preference(self, major: dict, preferences: Optional[list[str]]) -> bool:
        """按标准化专业名匹配，避免“自动化”误命中“机械设计制造及其自动化”。"""
        if not preferences:
            return True
        name = major.get("name", "")
        tags = major.get("tags", [])
        for pref in preferences:
            pref = str(pref).strip()
            if not pref:
                continue
            if pref in ["设计", "艺术", "美术", "音乐"]:
                if any(key in name for key in ["视觉传达", "数字媒体艺术", "艺术", "美术", "音乐", "广告"]):
                    return True
                continue
            if pref == name or name.startswith(pref):
                return True
            if len(pref) >= 2 and pref in name and pref not in ["自动化"]:
                return True
            if pref in tags:
                return True
        return False

    def _is_school_reasonable_for_profile(self, school: dict, user: UserPreferences) -> bool:
        """保守过滤明显不匹配的学校层次。

        这里不是官方投档线，只是为了避免本地库在没有历年位次数据时推荐离谱院校。
        真实冲稳保仍必须以后续考试院投档位次为准。
        """
        level = school.get("level", "")
        name = school.get("name", "")
        rank = user.rank
        score = user.score

        if rank:
            if name in self.ELITE_985:
                return rank <= 5000
            if level == "985":
                return rank <= 30000
            if level == "211":
                return rank <= 85000
            if level == "双一流":
                return rank <= 110000
            if level == "普通一本":
                return rank <= 300000
            return True

        if score:
            if name in self.ELITE_985:
                return score >= 660
            if level == "985":
                return score >= 600
            if level == "211":
                return score >= 540
            if level == "双一流":
                return score >= 520
            if level == "普通一本":
                return score >= 450
            return True

        return True

    def _is_school_major_compatible(self, school: dict, major: dict) -> bool:
        school_type = school.get("type", "")
        school_name = school.get("name", "")
        major_category = major.get("category", "")
        major_name = major.get("name", "")
        major_tags = major.get("tags", [])

        if major_category == "经济学":
            if any(key in school_name for key in ["医药", "中医", "体育", "美术", "音乐"]):
                return False
            if school_type in ["医药", "体育", "语言艺术"]:
                return False
            return True
        if major_category == "医学":
            if major_name == "动物医学":
                return school_type in ["农林海洋", "综合"] or "农业" in school_name
            return school_type in ["医药", "综合"] or any(key in school_name for key in ["医科", "医学", "医药", "中医", "华中科技"])
        if major_category == "农学":
            return school_type in ["农林海洋", "综合"] or "农业" in school_name
        if major_category == "教育学":
            return school_type in ["师范", "综合", "体育"] or any(key in school_name for key in ["师范", "教育", "体育"])
        if major_category == "艺术学":
            return school_type in ["语言艺术", "综合", "师范"] or any(key in school_name for key in ["艺术", "美术", "音乐", "传媒"])
        if major_category == "法学":
            return school_type in ["财经政法", "综合", "师范", "民族"] or any(key in school_name for key in ["政法", "财经", "民族"])
        if school_type == "医药" and major_category != "医学":
            return False
        if school_type == "工科" and major_category not in ["工学", "理学", "管理学"]:
            return False
        if school_type == "语言艺术" and major_category not in ["文学", "艺术学", "教育学"]:
            return False
        if school_type == "财经政法" and major_category not in ["经济学", "管理学", "法学"]:
            return False
        if school_type == "体育" and major_category not in ["教育学"]:
            return False
        if school_type == "师范" and major_category not in ["教育学", "文学", "历史学", "理学", "法学"]:
            return False
        if school_type == "农林海洋" and major_category not in ["农学", "工学", "理学", "管理学"]:
            return False
        if school_type == "民族" and major_category == "医学":
            return False
        if school_type == "农林海洋" and major_name in ["历史学", "汉语言文学"]:
            return False
        if school_type == "财经政法" and major_name in ["地理科学", "历史学"]:
            return False
        return True

    def _calculate_match_score(self, school: dict, major: dict, user: UserPreferences) -> float:
        """计算学校+专业与用户的匹配分数。

        家庭背景（family_background）决定内容偏好——选什么专业、什么类型学校。
        风险偏好（risk_appetite）决定冲稳保策略——报多激进。
        """
        score = 50.0
        bias = self._get_family_content_bias(user.family_background)

        # 城市偏好加分（叠加家庭城市权重）
        if user.city_preference:
            if any(self._is_province_preference(pref) and pref == school.get("province", "") for pref in user.city_preference):
                score += 15
            elif any((not self._is_province_preference(pref)) and pref in school.get("city", "") for pref in user.city_preference):
                score += 15
            elif any((not self._is_province_preference(pref)) and pref in school.get("name", "") for pref in user.city_preference):
                score += 14
            elif any((not self._is_province_preference(pref)) and school.get("name", "") in self.CITY_SCHOOL_HINTS.get(pref, []) for pref in user.city_preference):
                score += 14
            elif any(self._is_province_preference(pref) and pref in school.get("province", "") for pref in user.city_preference):
                score += 12
            elif any(pref in school.get("tier", "") for pref in user.city_preference):
                score += 8
            elif school["tier"] in ["一线", "新一线"]:
                score += 5

        # 家庭城市层级偏好（富裕家庭更看重一线/新一线平台）
        tier = school.get("tier", "")
        if tier in bias.get("city_tier_bonus", {}):
            score += bias["city_tier_bonus"][tier]

        # 专业偏好加分
        if user.major_preference:
            if self._matches_major_preference(major, user.major_preference):
                score += 20

        # 学校层次
        level_bonus = {"985": 20, "211": 12, "双一流": 10, "普通一本": 5, "普通二本": 0}
        score += level_bonus.get(school["level"], 0)

        # 学校类型偏好（家庭内容导向）
        school_type = school.get("type", "")
        for st_key, st_bonus in bias.get("school_type_bonus", {}).items():
            if st_key in school_type:
                score += st_bonus

        if "师范" in major.get("tags", []) and school_type == "师范":
            score += 24
        elif "师范" in major.get("tags", []) and school_type == "综合":
            score += 6

        # 就业率（普通家庭权重更高）
        emp = major.get("employment_rate", 0.5)
        emp_weight = bias.get("employment_weight", 1.0)
        score += emp * 20 * emp_weight

        # 薪资中位数
        salary = major.get("salary_median_5yr", 8000)
        score += min(salary / 500, 15)

        # 不可替代性
        irp = major.get("irreplaceability", 50)
        score += irp / 10

        # 家庭条件适配（内容层）
        tags = major.get("tags", [])
        for tag, bonus in bias.get("major_tag_bonus", {}).items():
            if tag in tags:
                score += bonus
        for tag, penalty in bias.get("major_tag_penalty", {}).items():
            if tag in tags:
                score += penalty

        # 深造要求适配
        if major.get("requires_grad_school", False):
            score += bias.get("grad_school_penalty", -10)

        # 天坑专业适配
        if "天坑" in tags:
            score += bias.get("tiankeng_penalty", -30)

        return score

    def _get_risk_strategy(self, risk_appetite: str | None) -> dict:
        """根据风险偏好获取对应的冲稳保策略。"""
        if not risk_appetite:
            return self.RISK_STRATEGIES["均衡"]
        for key in ("激进", "均衡", "稳妥"):
            if key in risk_appetite:
                return self.RISK_STRATEGIES[key]
        return self.RISK_STRATEGIES["均衡"]

    def _get_family_content_bias(self, family_background: str | None) -> dict:
        """根据家庭背景获取对应的内容偏好配置（决定选什么专业/学校）。"""
        if not family_background:
            return self.FAMILY_CONTENT_BIAS["普通"]
        for key in ("富裕", "中产", "普通"):
            if key in family_background:
                return self.FAMILY_CONTENT_BIAS[key]
        return self.FAMILY_CONTENT_BIAS["普通"]

    def _stratify_candidates(self, candidates: List[dict], user: UserPreferences, limit: int):
        """将候选分为冲/稳/保三层（按家庭策略动态调整期望数量）"""
        candidates = sorted(
            candidates,
            key=lambda item: (
                self._admission_difficulty_score(item["school"], item["major"]),
                item.get("match_score", 0),
            ),
            reverse=True,
        )
        n = len(candidates)
        if n == 0:
            return [], [], []

        strategy = self._get_risk_strategy(user.risk_appetite)
        desired_chong = max(1, round(limit * strategy["ratio"]["冲"]))
        desired_wen = max(2, round(limit * strategy["ratio"]["稳"]))
        desired_bao = max(1, round(limit * strategy["ratio"]["保"]))

        chong: list[dict] = []
        wen: list[dict] = []
        bao: list[dict] = []
        for item in candidates:
            risk = self._profile_risk_bucket(item["school"], item["major"], user)
            if risk == "冲":
                chong.append(item)
            elif risk == "稳":
                wen.append(item)
            elif risk == "保":
                bao.append(item)

        selected_school_names: set[str] = set()
        selected_chong = self._take_unique_schools(chong, desired_chong, selected_school_names)
        selected_wen = self._take_unique_schools(wen, desired_wen, selected_school_names)
        selected_bao = self._take_unique_schools(bao, desired_bao, selected_school_names)
        selected_ids = {id(item) for item in selected_chong + selected_wen + selected_bao}
        remaining = [item for item in candidates if id(item) not in selected_ids]
        for item in remaining:
            if len(selected_chong) + len(selected_wen) + len(selected_bao) >= limit:
                break
            school_name = item["school"].get("name")
            if school_name in selected_school_names:
                continue
            actual_risk = self._profile_risk_bucket(item["school"], item["major"], user)
            if actual_risk == "冲":
                selected_chong.append(item)
                selected_school_names.add(school_name)
            elif actual_risk == "稳":
                selected_wen.append(item)
                selected_school_names.add(school_name)
            elif actual_risk == "保":
                selected_bao.append(item)
                selected_school_names.add(school_name)

        return selected_chong, selected_wen, selected_bao

    def _take_unique_schools(self, items: list[dict], limit: int, seen_school_names: set[str]) -> list[dict]:
        selected: list[dict] = []
        for item in items:
            school_name = item["school"].get("name", "")
            if school_name in seen_school_names:
                continue
            selected.append(item)
            seen_school_names.add(school_name)
            if len(selected) >= limit:
                break
        return selected

    def _build_zero_candidate_fallback_plans(self, user: UserPreferences, limit: int) -> List[PlanOption]:
        """When hard constraints produce no usable plan, build syncable alternatives.

        The three alternatives map directly to the product strategy:
        1) relax city, keep major;
        2) keep city, switch to adjacent majors;
        3) keep conservative posture, lower school level / accept safer local choices.
        """
        fallback_specs: list[dict] = [
            {
                "strategy": "放宽城市但保专业",
                "reason": "原城市约束过窄，先保住用户最在意的专业方向，把搜索半径放到其他可接受省市。",
                "user": user.model_copy(update={"city_preference": None}),
                "risk": "稳",
                "limit": 3,
                "sort": "match",
            },
            {
                "strategy": "保城市但换相近专业",
                "reason": "城市不动，但把专业从单一名称扩展到同类出口，优先保住生活半径和家庭支持成本。",
                "user": user.model_copy(update={"major_preference": self._related_major_preferences(user.major_preference)}),
                "risk": "稳",
                "limit": 3,
                "sort": "match",
            },
            {
                "strategy": "保稳妥但降低学校层次",
                "reason": "城市和专业先不动，把目标从冲学校名气改成保录取与专业入口，适合普通家庭兜底。",
                "user": user,
                "risk": "保",
                "limit": 4,
                "sort": "safe",
            },
        ]

        plans: list[PlanOption] = []
        seen_pairs: set[tuple[str, str]] = set()
        order = 1
        for spec in fallback_specs:
            spec_user: UserPreferences = spec["user"]
            if spec["strategy"] == "保城市但换相近专业" and not spec_user.major_preference:
                continue
            candidates = self._filter_candidates(spec_user)
            candidates = self._sort_fallback_candidates(candidates, spec_user, spec["sort"])
            added = 0
            for combo in candidates:
                pair = (combo["school"].get("name", ""), combo["major"].get("name", ""))
                if not all(pair) or pair in seen_pairs:
                    continue
                risk = spec["risk"]
                plan = self._build_plan_option(combo, spec_user, risk, order)
                plan = self._with_fallback_strategy(plan, spec["strategy"], spec["reason"])
                plans.append(plan)
                seen_pairs.add(pair)
                order += 1
                added += 1
                if added >= spec["limit"] or len(plans) >= limit:
                    break
            if len(plans) >= limit:
                break
        return plans[:limit]

    def _sort_fallback_candidates(self, candidates: list[dict], user: UserPreferences, mode: str) -> list[dict]:
        if mode == "safe":
            return sorted(
                candidates,
                key=lambda item: (
                    self._admission_difficulty_score(item["school"], item["major"]),
                    -item.get("match_score", 0),
                ),
            )
        return sorted(
            candidates,
            key=lambda item: (
                item.get("match_score", 0),
                self._admission_difficulty_score(item["school"], item["major"]),
            ),
            reverse=True,
        )

    def _with_fallback_strategy(self, plan: PlanOption, strategy: str, reason: str) -> PlanOption:
        strategy_note = f"兜底路径【{strategy}】：{reason}"
        original_reason = (plan.reason or "").strip()
        updates = {
            "fallback_strategy": strategy,
            "fallback_reason": reason,
            "reason": f"{strategy_note}；{original_reason}" if original_reason else strategy_note,
            "tags": [strategy, *(plan.tags or [])][:4],
        }
        if plan.family_risk_summary:
            updates["family_risk_summary"] = f"{strategy_note} {plan.family_risk_summary}"
        if plan.risk_warning:
            updates["risk_warning"] = f"{strategy}：{plan.risk_warning}"
        else:
            updates["risk_warning"] = strategy_note
        return plan.model_copy(update=updates)

    def _related_major_preferences(self, preferences: Optional[list[str]]) -> list[str]:
        if not preferences:
            return []
        known_major_names = {major.get("name") for major in self.data.majors.values()}
        related: list[str] = []

        def add_major(name: str) -> None:
            if name in known_major_names and name not in related:
                related.append(name)

        for pref in preferences:
            pref = str(pref or "").strip()
            if not pref:
                continue
            if pref in self.RELATED_MAJOR_GROUPS:
                for item in self.RELATED_MAJOR_GROUPS[pref]:
                    add_major(item)
                continue
            matched = next((name for name in known_major_names if pref == name or (len(pref) >= 2 and pref in name)), "")
            if matched:
                for item in self.RELATED_MAJOR_GROUPS.get(matched, []):
                    add_major(item)
                continue
            if any(key in pref for key in ["中文", "汉语言", "文学"]):
                for item in ["汉语言文学", "小学教育", "教育学", "历史学", "英语"]:
                    add_major(item)
            elif any(key in pref for key in ["师范", "教育", "教师"]):
                for item in ["小学教育", "教育学", "汉语言文学", "历史学", "地理科学"]:
                    add_major(item)
            elif any(key in pref for key in ["计算机", "软件", "人工智能", "数据"]):
                for item in ["软件工程", "计算机科学与技术", "数据科学与大数据技术", "信息安全"]:
                    add_major(item)
        return [item for item in related if item not in set(preferences or [])]

    def _profile_risk_bucket(self, school: dict, major: dict, user: UserPreferences) -> str:
        """按用户画像把院校专业组合粗分为冲/稳/保/废。

        根据家庭背景动态调整阈值：富裕家庭冲的门槛更低，
        普通家庭冲的门槛更高，确保不同风险承受能力得到匹配。
        """
        difficulty = self._admission_difficulty_score(school, major)
        major_category = (major or {}).get("category", "")
        rank = user.rank
        score = user.score or 0
        strategy = self._get_risk_strategy(user.risk_appetite)
        off = strategy["risk_threshold_offset"]

        if (rank is not None and rank <= 15000) or score >= 620:
            if difficulty >= 102 + off:
                return "冲"
            if difficulty >= 86 + off:
                return "稳"
            if difficulty >= 64 + off:
                return "保"
            if major_category in ["经济学", "医学", "教育学", "农学", "艺术学"] and user.city_preference and difficulty >= 58:
                return "保"
            return "废"

        if (rank is not None and rank <= 35000) or score >= 590:
            if difficulty >= 96 + off:
                return "冲"
            if difficulty >= 76 + off:
                return "稳"
            if difficulty >= 60 + off:
                return "保"
            if major_category in ["经济学", "医学", "教育学", "农学", "艺术学"] and user.city_preference and difficulty >= 54:
                return "保"
            return "废"

        if score >= 540:
            if difficulty >= 82 + off:
                return "冲"
            if difficulty >= 63 + off:
                return "稳"
            if difficulty >= 52 + off:
                return "保"
            return "废"

        if difficulty >= 70 + off:
            return "冲"
        if difficulty >= 54 + off:
            return "稳"
        return "保"

    def _admission_difficulty_score(self, school: dict, major: dict | None = None) -> float:
        """投档难度代理分：用于冲稳保分层，不作为官方录取概率。

        本地库没有逐省逐专业组历年位次时，分层不能只看城市/专业匹配度。
        这里用学校层次、行业辨识度、城市、平均薪资和专业热度做保守排序，
        后续仍必须回到省考试院投档表核验。
        """
        level = school.get("level", "")
        tier = school.get("tier", "")
        name = school.get("name", "")
        school_type = school.get("type", "")
        tags = school.get("tags", []) or []
        avg_salary = school.get("average_salary") or 0
        major_name = (major or {}).get("name", "")

        score = {
            "985": 95,
            "211": 82,
            "双一流": 76,
            "普通一本": 55,
            "普通二本": 38,
        }.get(level, 50)

        score += {"一线": 5, "新一线": 3, "二线": 1}.get(tier, 0)
        score += min(max((avg_salary - 9000) / 900, -4), 8)

        if any(key in name for key in ["邮电", "电子科技", "信息工程"]):
            score += 8
        if any(key in name for key in ["建筑", "农业", "海洋", "石油", "矿业"]) and level == "普通一本":
            score -= 4
        if school_type in ["工科", "理工"] and self._is_hot_engineering_major(major_name):
            score += 2
        if any(tag in tags for tag in ["计算机", "电子信息", "通信", "网络安全"]):
            score += 3

        if self._is_hot_engineering_major(major_name):
            score += 4
        elif major_name in ["临床医学", "口腔医学", "法学"]:
            score += 3
        elif major and major.get("requires_grad_school"):
            score -= 2

        return score

    def _is_hot_engineering_major(self, major_name: str) -> bool:
        return any(
            key in major_name
            for key in ["计算机", "软件", "人工智能", "电子信息", "通信", "信息安全", "网络", "数据科学", "自动化"]
        )

    def _build_plan_option(self, combo: dict, user: UserPreferences, risk: str, order: int) -> PlanOption:
        school = combo["school"]
        major = combo["major"]
        prob = self._estimate_probability(combo, user, risk)
        family_risk = build_family_risk_profile(school, major, user.family_background, risk)

        # 生成推荐理由
        reason = self._generate_reason(school, major, user, risk, family_risk)

        # 风险警告
        risk_warning = None
        tags = major.get("tags", [])[:2]
        if "天坑" in major.get("tags", []):
            risk_warning = f"{major['name']}被标记为高风险专业，需谨慎评估"
        if major.get("requires_grad_school", False):
            risk_warning = f"{major['name']}本科竞争力弱，必须深造"
        if not risk_warning and family_risk["risk_tags"]:
            risk_warning = f"{school['name']}-{major['name']}需重点核验：{'、'.join(family_risk['risk_tags'][:3])}"

        return PlanOption(
            order=order,
            school=school["name"],
            major=major["name"],
            school_level=school.get("level", "待核验"),
            major_group=f"{major['category']}类",
            risk_level=risk,
            probability=prob,
            median_salary_5yr=major.get("salary_median_5yr"),
            irreplaceability=major.get("irreplaceability"),
            reason=reason,
            risk_warning=risk_warning,
            risk_tags=family_risk["risk_tags"],
            family_strategy=family_risk["family_strategy"],
            family_risk_summary=family_risk["family_risk_summary"],
            tags=tags,
            fortune500_pass=school["level"] in ["985", "211", "双一流"],
        )

    def _estimate_probability(self, combo: dict, user: UserPreferences, risk: str) -> int:
        """稳定生成规则模拟概率，按风险偏好动态调整基线。"""
        school = combo["school"]
        level = school.get("level", "")
        name = school.get("name", "")
        strategy = self._get_risk_strategy(user.risk_appetite)
        ranges = strategy["prob_ranges"]
        lo, hi = ranges.get(risk, (55, 90))
        base = (lo + hi) // 2

        rank = user.rank
        if rank:
            if name in self.ELITE_985:
                if rank > 3000:
                    base -= 8
                elif rank <= 1000:
                    base += 5
            elif level == "985":
                if rank <= 12000:
                    base += 4
                elif rank > 25000:
                    base -= 7
            elif level == "211":
                if rank <= 40000:
                    base += 4
                elif rank > 75000:
                    base -= 6
            elif level == "双一流":
                if rank <= 50000:
                    base += 3
                elif rank > 95000:
                    base -= 5
            elif level == "普通一本":
                if rank <= 80000:
                    base += 5
                elif rank > 170000:
                    base -= 5
        else:
            score = user.score or 0
            if name in self.ELITE_985 and score < 680:
                base -= 8
            elif level == "985" and score >= 630:
                base += 4
            elif level == "211" and score >= 580:
                base += 4
            elif level == "双一流" and score >= 560:
                base += 3

        return max(lo - 5, min(hi + 1, base))

    def _generate_reason(
        self,
        school: dict,
        major: dict,
        user: UserPreferences,
        risk: str,
        family_risk: dict | None = None,
    ) -> str:
        """生成更有区分度的就业倒推推荐理由。"""
        family_risk = family_risk or build_family_risk_profile(school, major, user.family_background, risk)
        parts = [
            self._risk_position_text(risk, school, user.risk_appetite or "均衡"),
            self._school_position_text(school, major),
            self._major_fit_text(major, user),
            self._trend_text(major, school),
            family_risk.get("family_risk_summary", ""),
        ]

        salary = major.get("salary_median_5yr")
        if salary:
            parts.append(f"收入只能按本地专业库粗估，普通毕业生几年后参考约{salary // 1000}K，不能当官方工资条")

        if user.city_preference:
            city_hit = school.get("city") in user.city_preference or school.get("province") in user.city_preference
            if city_hit:
                parts.append("城市偏好匹配，实习、就业和家庭支持成本更容易接住")
            elif school.get("tier") in ["一线", "新一线"]:
                parts.append(f"{school.get('city')}资源强，但生活成本和同城竞争也要一起算")

        return "；".join(part for part in parts if part)

    def _risk_position_text(self, risk: str, school: dict, risk_appetite: str = "均衡") -> str:
        """基于风险偏好（而非家庭背景）解释冲稳保的定位。"""
        level = school.get("level", "院校")
        name = school.get("name", "该校")
        if risk == "冲":
            if "激进" in risk_appetite:
                return f"冲档对{name}的{level}平台发起挑战，激进策略下可承担更高风险去搏上限"
            if "稳妥" in risk_appetite:
                return f"冲档机会有限，{name}的{level}上限值得尝试，但稳妥策略下务必确保后续稳保充足"
            return f"冲档看的是{name}的{level}上限，作用是抬志愿天花板，后面必须接稳保"
        if risk == "保":
            if "激进" in risk_appetite:
                return f"保底对激进策略是安全网，{name}给当前分数留退路，防止冲高失败无学可上"
            if "稳妥" in risk_appetite:
                return f"保底是稳妥策略的命根子，{name}必须确保100%能录，防止滑档后没有退路"
            return f"保底不是随便填，{name}的价值在于给当前分数留安全垫，防止滑到更被动的位置"
        if "激进" in risk_appetite:
            return f"稳档重点看{name}能不能兼顾{level}层次和优质专业，是激进方案中的压舱石"
        if "稳妥" in risk_appetite:
            return f"稳档对稳妥策略最重要，{name}的{level}层次和{school.get('type', '')}专业特色要能确保就业出口"
        return f"稳档重点看{name}能不能同时守住学校层次和专业出口，是志愿表的骨架选择"

    def _school_position_text(self, school: dict, major: dict) -> str:
        level = school.get("level", "")
        school_type = school.get("type", "")
        city = school.get("city") or school.get("province") or "所在城市"
        major_category = major.get("category", "")

        if level == "985":
            base = "985平台的筛选价值强，适合优先保住学历门槛和校友资源"
        elif level == "211":
            base = "211平台在简历初筛里有现实作用，适合普通家庭先拿可解释的学历标签"
        elif level == "双一流":
            base = "双一流要看具体学科实力，不能误写成211，但行业特色强的专业值得单独评估"
        elif level == "普通一本":
            base = "普通一本不能只看牌子，要把城市、专业和录取确定性放在一起算"
        else:
            base = f"{level or '院校'}层次需要结合专业组和历年位次细核"

        if school_type and school_type != "综合":
            if school_type in ["理工", "工科"] and major_category == "工学":
                base += "，理工底色和工科专业更顺，培养资源通常更集中"
            elif school_type in ["财经", "财经政法"] and major_category in ["经济学", "管理学", "法学"]:
                base += "，财经政法资源对口，但要警惕行业人脉和实习门槛"
            elif school_type == "师范":
                base += "，师范院校更适合走教师、考编或稳定公共部门路径"
            elif school_type in ["农林", "海洋"]:
                base += "，行业特色明显，适合接受垂直赛道的人，不适合只追泛热门"
            elif school_type == "医药":
                base += "，医药培养周期长，必须把资格证、规培和深造成本算进去"
            else:
                base += f"，{school_type}类院校要核对该专业是不是学校强项"

        return f"{base}；{city}带来的就业半径也要纳入判断"

    def _major_fit_text(self, major: dict, user: UserPreferences) -> str:
        name = major.get("name", "该专业")
        tags = major.get("tags", [])
        if major.get("requires_grad_school"):
            return f"{name}本科直接变现不一定够，适合能接受考研或继续深造的家庭"
        if "看背景" in tags:
            return f"{name}就业分化明显，普通家庭要优先看实习、证书和学校城市资源"
        if "天坑" in tags:
            return f"{name}不适合只凭兴趣硬上，读研、转行和行业周期风险都要提前准备"
        if "技术壁垒" in tags or (major.get("irreplaceability") or 0) >= 75:
            return f"{name}更看项目能力和硬技能，学得深比只拿专业名字更重要"
        if (major.get("employment_rate") or 0) >= 0.88:
            return f"{name}出口相对更直，适合先追求就业确定性"
        return f"{name}需要结合培养方案看具体方向，不能只按专业大类想象就业"

    def _trend_text(self, major: dict, school: dict) -> str:
        name = major.get("name", "该专业")
        category = major.get("category", "")
        tags = major.get("tags", [])
        school_type = school.get("type", "")
        if category == "工学" or "技术壁垒" in tags:
            return f"未来趋势上，{name}低端重复岗位会被自动化挤压，真正值钱的是工程实践、项目经历和复合能力"
        if category in ["经济学", "管理学", "法学"]:
            return f"未来趋势上，{name}会更看城市平台、实习质量和证书，学校资源差异会被放大"
        if school_type in ["医药"] or category == "医学":
            return "未来趋势上，医疗健康需求长期在，但培养周期和准入门槛不会低"
        if school_type in ["师范"] or category in ["教育学"]:
            return "未来趋势上，教师岗位更看编制、地区人口和学科需求，稳定但竞争会更细"
        return f"未来趋势上，{name}要看行业周期和个人能力积累，不能只用当下冷热判断"

    def _generate_recommend_summary(
        self,
        plans: List[PlanOption],
        user: UserPreferences,
        used_zero_candidate_fallback: bool = False,
    ) -> str:
        family = user.family_background or "普通家庭"
        risk = user.risk_appetite or "均衡"
        strategy = self._get_risk_strategy(risk)
        content_bias = self._get_family_content_bias(family)
        chong_cnt = sum(1 for p in plans if p.risk_level == "冲")
        wen_cnt = sum(1 for p in plans if p.risk_level == "稳")
        bao_cnt = sum(1 for p in plans if p.risk_level == "保")

        fallback_text = ""
        if used_zero_candidate_fallback:
            fallback_names = []
            for plan in plans:
                if plan.fallback_strategy and plan.fallback_strategy not in fallback_names:
                    fallback_names.append(plan.fallback_strategy)
            fallback_text = (
                f"原始硬约束没有形成可用方案，已启动0候选兜底：{'、'.join(fallback_names)}。"
                "这些是替代路径方案，不等同于完全满足原始条件。"
            )

        return (
            f"基于你的条件（{user.province}，{user.score}分，位次{user.rank or '未填'}，{family}），"
            f"报考策略为【{strategy['label']}】（{risk}）：{strategy['description']}。"
            f"内容导向为【{content_bias['label']}】：{content_bias['description']}。"
            f"共筛选{len(plans)}个志愿（冲{chong_cnt}/稳{wen_cnt}/保{bao_cnt}）。"
            f"{fallback_text}"
            f"主回答先讲冲稳保定位、专业出口、城市资源和调剂风险；具体模拟概率和收入估算只进入结构化方案。"
            f"{'结构合理，兼顾层次与确定性。' if len(plans) >= 6 else '建议继续完善志愿表。'}"
        )

    def _get_salary_range(self, plans: List[PlanOption]) -> str:
        salaries = [p.median_salary_5yr for p in plans if p.median_salary_5yr]
        if not salaries:
            return "暂无数据"
        low = min(salaries) // 1000
        high = max(salaries) // 1000
        if low == high:
            return f"约{low}K"
        return f"{low}K-{high}K"

    def _collect_recommend_flags(self, plans: List[PlanOption], user: UserPreferences) -> List[str]:
        flags = []
        family = user.family_background or "普通家庭"
        risk = user.risk_appetite or "均衡"
        strategy = self._get_risk_strategy(risk)
        content_bias = self._get_family_content_bias(family)
        if not user.rank:
            flags.append("缺少全省位次，只能用分数做粗略层次过滤；院校匹配精度会明显下降")

        # 按风险偏好添加核心提示
        flags.append(strategy["flag_focus"])

        # 按家庭背景添加内容导向提示
        flags.append(f"内容导向：{content_bias['label']}——{content_bias['description']}")

        # 统计各档数量，检查结构是否合理
        chong = [p for p in plans if p.risk_level == "冲"]
        bao = [p for p in plans if p.risk_level == "保"]
        if risk == "稳妥" and len(bao) < 2:
            flags.append("保守型策略下保底志愿数量偏少，建议增加保底")
        if risk == "激进" and len(chong) == 0:
            flags.append("激进型策略下没有冲档志愿，可适当提高目标院校层次")

        for p in plans:
            if p.risk_level == "冲" and risk == "稳妥":
                flags.append(f"{p.school}为冲刺志愿，稳妥策略下务必确认后面有充足稳保底")
            if p.risk_warning:
                flags.append(p.risk_warning)
            if p.family_risk_summary:
                flags.append(f"{p.school}风险标签：{p.family_risk_summary}")

        return list(set(flags))[:6]

    def _llm_enhance_recommend(self, plans: List[PlanOption], user: UserPreferences) -> List[ThinkingStep]:
        """使用 LLM 增强分析过程"""
        if not llm_client.is_available():
            return [
                ThinkingStep(step="规则引擎筛选", analysis="基于分数、城市、专业偏好进行初筛"),
                ThinkingStep(step="冲稳保分层", analysis="按规则模拟录取概率将候选分为三层"),
                ThinkingStep(step="8维度评分", analysis="应用就业倒推法、社会筛子论等启发式评估"),
            ]

        try:
            prompt = self._build_recommend_llm_prompt(plans, user)
            response = llm_client.client.messages.create(
                model=llm_client.model,
                max_tokens=1500,
                system=llm_client.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""
            return self._parse_thinking_from_text(text)
        except Exception:
            return [
                ThinkingStep(step="规则引擎筛选", analysis="基于分数、城市、专业偏好进行初筛"),
                ThinkingStep(step="冲稳保分层", analysis="按规则模拟录取概率将候选分为三层"),
                ThinkingStep(step="8维度评分", analysis="应用就业倒推法、社会筛子论等启发式评估"),
            ]

    def _build_recommend_llm_prompt(self, plans: List[PlanOption], user: UserPreferences) -> str:
        plan_str = "\n".join([
            f"{p.order}. [{p.risk_level}] {p.school} - {p.major}（"
            f"{f'替代路径：{p.fallback_strategy}，' if p.fallback_strategy else ''}"
            f"概率{p.probability}%，薪资{p.median_salary_5yr or 'N/A'}，家庭风险标签：{'、'.join(p.risk_tags) if p.risk_tags else '暂无'}）"
            for p in plans
        ])
        risk_strategy = self._get_risk_strategy(user.risk_appetite)
        content_bias = self._get_family_content_bias(user.family_background)
        return (
            f"请作为张雪峰，为以下志愿方案提供简要的思维过程分析。"
            f"\n\n考生信息：{user.province}，{user.score}分，{user.family_background or '普通家庭'}"
            f"\n\n报考策略（由风险偏好决定）：{risk_strategy['label']} —— {risk_strategy['description']}"
            f"\n\n内容导向（由家庭条件决定）：{content_bias['label']} —— {content_bias['description']}"
            f"\n\n推荐方案：\n{plan_str}"
            f"\n\n请用2-3句话说明你的分析思路，格式为："
            f"\n1. [步骤名]：[分析内容]"
            f"\n2. [步骤名]：[分析内容]"
            f"\n3. [步骤名]：[分析内容]"
        )

    def _parse_thinking_from_text(self, text: str) -> List[ThinkingStep]:
        thinking = []
        for line in text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                content = line.lstrip("-• ").lstrip("0123456789.)")
                if "：" in content or ":" in content:
                    parts = content.replace(":", "：", 1).split("：", 1)
                    thinking.append(ThinkingStep(step=parts[0].strip(), analysis=parts[1].strip()))
                else:
                    thinking.append(ThinkingStep(step="分析", analysis=content))
        if not thinking:
            thinking.append(ThinkingStep(step="综合分析", analysis="基于张雪峰思维框架进行推荐"))
        return thinking

    # ============================================================
    # 2. 方案对比
    # ============================================================
    def compare(self, request: CompareRequest) -> CompareResult:
        """对比多个志愿方案，给出最优选择"""
        plans = request.plans
        user = request.user

        if not plans:
            return CompareResult(
                best_choice=None,
                comparison_table="",
                dimension_scores={},
                final_verdict="没有提供可对比的方案",
                thinking_process=[ThinkingStep(step="输入检查", analysis="未收到任何方案")],
            )

        # 计算每个方案的多维度得分
        dim_scores = {}
        for p in plans:
            dim_scores[p.school + "-" + p.major] = self._score_plan_dimensions(p, user)

        # 选择最优
        best = max(plans, key=lambda x: dim_scores.get(x.school + "-" + x.major, {}).get("total", 0))

        # 生成对比表
        table = self._generate_comparison_table(plans, dim_scores)

        # 最终裁决
        verdict = self._generate_verdict(best, plans, user)

        # 思维过程
        thinking = [
            ThinkingStep(step="多维度量化", analysis="从就业、不可替代性、家庭适配、城市等8个维度打分"),
            ThinkingStep(step="权重调整", analysis=f"根据{user.family_background or '普通家庭'}背景调整权重配置"),
            ThinkingStep(step="最优选择", analysis=f"{best.school}-{best.major}综合得分最高"),
        ]

        return CompareResult(
            best_choice=best,
            comparison_table=table,
            dimension_scores=dim_scores,
            final_verdict=verdict,
            thinking_process=thinking,
        )

    def _score_plan_dimensions(self, plan: PlanOption, user: UserPreferences) -> dict:
        scores = {}
        school = self.data.get_school(plan.school)
        major_name = plan.major
        major = self.data.get_major(major_name)

        # 复用 zxf_engine 的评分逻辑
        scores["employment_reversal"] = evaluator._score_employment_reversal(major, school)
        scores["social_sieve"] = evaluator._score_social_sieve(school, major)
        scores["irreplaceability"] = evaluator._score_irreplaceability(major)
        scores["median_principle"] = evaluator._score_median(major, school)

        from core.models import UserProfile
        profile = UserProfile(
            score=user.score,
            province=user.province,
            family_background=user.family_background,
            city_preference=user.city_preference,
        )
        scores["family_background"] = evaluator._score_family_fit(major, profile)
        scores["city_priority"] = evaluator._score_city_priority(school, profile)
        scores["fortune500_test"] = evaluator._score_fortune500(school, major)
        scores["ten_year_test"] = evaluator._score_ten_year(major, profile)

        # 总加权分
        weights = evaluator.WEIGHTS.get(user.family_background or "普通家庭", evaluator.WEIGHTS["普通家庭"])
        total = sum(scores.get(k, 50) * weights.get(k, 1.0) for k in weights.keys()) / sum(weights.values())
        scores["total"] = round(total)

        return scores

    def _generate_comparison_table(self, plans: List[PlanOption], dim_scores: dict) -> str:
        lines = ["| 方案 | 总分 | 就业 | 筛子 | 不可替代 | 家庭适配 | 城市 | 10年后 |"]
        lines.append("|------|------|------|--------|----------|----------|------|--------|")
        for p in plans:
            key = p.school + "-" + p.major
            sc = dim_scores.get(key, {})
            lines.append(
                f"| {p.school}-{p.major} | {sc.get('total', 'N/A')} | "
                f"{sc.get('employment_reversal', 'N/A')} | {sc.get('social_sieve', 'N/A')} | "
                f"{sc.get('irreplaceability', 'N/A')} | {sc.get('family_background', 'N/A')} | "
                f"{sc.get('city_priority', 'N/A')} | {sc.get('ten_year_test', 'N/A')} |"
            )
        return "\n".join(lines)

    def _generate_verdict(self, best: PlanOption, plans: List[PlanOption], user: UserPreferences) -> str:
        family = user.family_background or "普通家庭"
        return (
            f"基于{family}背景，最优选择是 **{best.school}的{best.major}**。"
            f"理由：规则模拟录取概率{best.probability}% {'风险可控' if best.probability > 75 else '有一定风险'}，"
            f"{'本地估算中位数薪资' + str(best.median_salary_5yr // 1000) + 'K' if best.median_salary_5yr else '薪资数据待补充'}，"
            f"{'通过500强测试' if best.fortune500_pass else '未达500强校招门槛'}。"
        )

    # ============================================================
    # 3. 数据洞察
    # ============================================================
    def insights(self, request: InsightRequest) -> InsightResponse:
        """生成专业/院校/行业的深度数据洞察"""
        target_type = request.target_type
        target_name = request.target_name
        user = request.user

        major = self.data.get_major(target_name) if target_type == "major" else None
        school = self.data.get_school(target_name) if target_type == "school" else None

        if not major and not school:
            return InsightResponse(
                target=target_name,
                target_type=target_type,
                overview="未找到相关数据",
                trend_analysis="暂无数据",
                risk_factors=["数据缺失"],
                opportunities=[],
                similar_options=[],
                thinking_process=[ThinkingStep(step="数据检索", analysis="未在知识库中找到匹配记录")],
            )

        if target_type == "major" and major:
            return self._major_insights(major, user)
        elif target_type == "school" and school:
            return self._school_insights(school, user)
        else:
            return InsightResponse(
                target=target_name,
                target_type=target_type,
                overview=f"{target_name} 数据洞察",
                trend_analysis="行业整体趋势稳定" if target_type == "industry" else "",
                risk_factors=[],
                opportunities=[],
                similar_options=[],
                thinking_process=[ThinkingStep(step="分析", analysis="基于中位数原则进行评估")],
            )

    def _major_insights(self, major: dict, user: Optional[UserPreferences]) -> InsightResponse:
        overview = major.get("description", "")
        salary = major.get("salary_median_5yr")
        emp = major.get("employment_rate")
        irp = major.get("irreplaceability")

        # 趋势分析
        if salary and salary > 15000:
            trend = "薪资水平处于头部区间，市场需求旺盛"
        elif salary and salary > 10000:
            trend = "薪资中等偏上，就业稳定性较好"
        else:
            trend = "薪资竞争力一般，需关注长期发展"

        risk_factors = major.get("risk_factors", [])
        tags = major.get("tags", [])

        opportunities = []
        if "热门" in tags:
            opportunities.append("市场需求持续旺盛")
        if "技术壁垒" in tags:
            opportunities.append("高不可替代性带来议价权")
        if "稳定" in tags:
            opportunities.append("就业稳定性高，抗周期能力强")

        # 相似选项
        similar = []
        category = major.get("category", "")
        for m in self.data.majors.values():
            if m["id"] != major["id"] and m["category"] == category:
                similar.append(m["name"])
        similar = similar[:5]

        thinking = [
            ThinkingStep(step="中位数原则", analysis=f"该专业5年后本地估算中位数薪资{salary or 'N/A'}，只能用于方向参考"),
            ThinkingStep(step="就业倒推", analysis=f"本地估算就业率{int(emp*100) if emp else 'N/A'}%，{'高于' if emp and emp > 0.85 else '低于'}平均水平"),
            ThinkingStep(step="不可替代性", analysis=f"技术壁垒估算评分{irp or 'N/A'}/100，{'高壁垒' if irp and irp > 80 else '中等壁垒' if irp and irp > 60 else '低壁垒'}"),
        ]

        return InsightResponse(
            target=major["name"],
            target_type="major",
            overview=overview,
            median_salary=salary,
            employment_rate=emp,
            irreplaceability=irp,
            trend_analysis=trend,
            risk_factors=risk_factors,
            opportunities=opportunities,
            similar_options=similar,
            thinking_process=thinking,
        )

    def _school_insights(self, school: dict, user: Optional[UserPreferences]) -> InsightResponse:
        overview = f"{school['name']}是一所{school['level']}院校，位于{school['tier']}城市{school['city']}。"
        salary = school.get("average_salary")
        emp = school.get("employment_rate")

        trend = f"位于{school['city']}，{'一线城市资源丰富' if school['tier'] == '一线' else '新一线城市发展潜力大' if school['tier'] == '新一线' else '城市资源一般'}"

        thinking = [
            ThinkingStep(step="社会筛子", analysis=f"{school['level']}层次在社会筛选中处于{'顶层' if school['level'] == '985' else '中上层' if school['level'] in ['211', '双一流'] else '中层'}位置"),
            ThinkingStep(step="城市资源", analysis=f"{school['city']}为{school['tier']}城市，实习和就业机会{'丰富' if school['tier'] in ['一线', '新一线'] else '一般'}"),
        ]

        return InsightResponse(
            target=school["name"],
            target_type="school",
            overview=overview,
            median_salary=salary,
            employment_rate=emp,
            trend_analysis=trend,
            risk_factors=[],
            opportunities=[f"{tag}优势" for tag in school.get("tags", [])[:3]],
            similar_options=[],
            thinking_process=thinking,
        )

    # ============================================================
    # 4. 10年后压迫测试
    # ============================================================
    def pressure_test(self, request: PressureTestRequest) -> PressureTestResponse:
        """模拟10年后极端场景"""
        plan = request.plan
        user = request.user
        compare = request.compare_with

        major = self.data.get_major(plan.major)
        compare_major = self.data.get_major(compare.major) if compare else None

        # 模拟10年后薪资（简化：中位数薪资 * 1.5~2.0 倍增长系数）
        year_10 = int((major.get("salary_median_5yr", 8000) if major else 8000) * 1.8) if major else None
        year_10_cmp = int((compare_major.get("salary_median_5yr", 8000) if compare_major else 8000) * 1.8) if compare_major else None

        # 构建场景
        scenario = self._build_pressure_scenario(plan, compare, user, year_10, year_10_cmp)

        # 分析
        analysis = self._build_pressure_analysis(plan, compare, year_10, year_10_cmp, user)

        # 结论
        acceptable, conclusion = self._build_pressure_conclusion(plan, compare, year_10, year_10_cmp, user)

        thinking = [
            ThinkingStep(step="设定基准", analysis=f"{plan.major}10年后估算中位数薪资{year_10//1000 if year_10 else 'N/A'}K"),
            ThinkingStep(step="极端对比", analysis=f"与{compare.major if compare else '另一选择'}形成{'明显差距' if year_10_cmp and year_10 and abs(year_10 - year_10_cmp) > 5000 else '较小差距'}"),
            ThinkingStep(step="压力测试", analysis="将极端结果展示给用户，迫使其面对现实"),
        ]

        return PressureTestResponse(
            scenario=scenario,
            year_10_salary_median=year_10,
            year_10_salary_compare=year_10_cmp,
            analysis=analysis,
            stress_conclusion=conclusion,
            acceptable=acceptable,
            thinking_process=thinking,
        )

    def _build_pressure_scenario(self, plan: PlanOption, compare: Optional[PlanOption],
                                  user: UserPreferences, y10: Optional[int], y10_cmp: Optional[int]) -> str:
        base = (
            f"假设你的孩子选择了 **{plan.school}的{plan.major}**，"
            f"10年后（2036年）的估算中位数月收入约为 **{y10//1000 if y10 else 'N/A'}K**。"
        )
        if compare:
            base += (
                f"\n而当年分数相近、选择了 **{compare.school}的{compare.major}** 的同学，"
                f"10年后的估算中位数月收入约为 **{y10_cmp//1000 if y10_cmp else 'N/A'}K**。"
            )
        return base

    def _build_pressure_analysis(self, plan: PlanOption, compare: Optional[PlanOption],
                                  y10: Optional[int], y10_cmp: Optional[int], user: UserPreferences) -> str:
        family = user.family_background or "普通家庭"
        parts = [f"基于{family}背景分析："]

        if y10 and y10_cmp:
            diff = y10 - y10_cmp
            if diff > 5000:
                parts.append(f"你的选择薪资优势{diff//1000}K/月，年收入差距约{diff*12//10000}万。")
            elif diff < -5000:
                parts.append(f"你的选择薪资落后{abs(diff)//1000}K/月，年收入差距约{abs(diff)*12//10000}万。")
            else:
                parts.append("两个选择长期薪资差距不大，更应关注工作满意度和生活质量。")

        major = self.data.get_major(plan.major)
        if major and "AI替代" in str(major.get("risk_factors", [])):
            parts.append(f"⚠️ 注意：{plan.major}面临AI替代风险，10年后就业格局可能与今天不同。")

        return "\n".join(parts)

    def _build_pressure_conclusion(self, plan: PlanOption, compare: Optional[PlanOption],
                                    y10: Optional[int], y10_cmp: Optional[int], user: UserPreferences):
        if not y10 or not y10_cmp:
            return True, "数据不足以做出判断，建议补充更多信息。"

        diff = y10 - y10_cmp
        if diff < -8000:
            return False, (
                f"你能不能接受孩子工作十年后，收入比当年分数不如他的人低{abs(diff)//1000}K/月？"
                f"如果接受不了，现在就重新考虑。"
            )
        elif diff < -3000:
            return False, (
                f"10年后收入差距{abs(diff)//1000}K/月，虽然不算巨大，但长期累积也不容忽视。"
                f"慎重考虑。"
            )
        else:
            return True, "长期薪资差距在可接受范围内，这个选择从经济角度是合理的。"

    # ============================================================
    # 5. 深度分析
    # ============================================================
    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """深度分析单个院校或专业"""
        target_type = request.target_type
        target_name = request.target_name
        school_name = request.school_name
        user = request.user

        major = self.data.get_major(target_name) if target_type == "major" else None
        school = self.data.get_school(school_name or target_name) if target_type == "school" or school_name else None

        if target_type == "major" and not major:
            return AnalyzeResponse(
                target=target_name,
                target_type=target_type,
                deep_analysis="未找到该专业数据",
                eight_dimensions={},
                suitability_score=50,
                for_whom="",
                against_whom="",
                thinking_process=[ThinkingStep(step="检索", analysis="未找到匹配数据")],
                zxf_quote="数据不足，无法判断。",
            )

        # 计算8维度得分
        dims = {}
        if major and school:
            dims = {
                "就业倒推": evaluator._score_employment_reversal(major, school),
                "社会筛子": evaluator._score_social_sieve(school, major),
                "不可替代性": evaluator._score_irreplaceability(major),
                "中位数原则": evaluator._score_median(major, school),
                "家庭适配": evaluator._score_family_fit(major, UserProfile(
                    score=user.score if user else None,
                    family_background=user.family_background if user else "普通家庭",
                )),
                "城市优先": evaluator._score_city_priority(school, UserProfile(
                    city_preference=user.city_preference if user else None,
                )),
                "500强测试": evaluator._score_fortune500(school, major),
                "10年后测试": evaluator._score_ten_year(major, UserProfile(
                    family_background=user.family_background if user else "普通家庭",
                )),
            }
            total = sum(dims.values()) / len(dims)
        elif major:
            dims = {
                "就业倒推": evaluator._score_employment_reversal(major, None),
                "不可替代性": evaluator._score_irreplaceability(major),
                "中位数原则": evaluator._score_median(major, None),
                "10年后测试": evaluator._score_ten_year(major, UserProfile()),
            }
            total = sum(dims.values()) / len(dims)
        elif school:
            dims = {
                "社会筛子": evaluator._score_social_sieve(school, None),
                "城市优先": evaluator._score_city_priority(school, UserProfile()),
                "500强测试": evaluator._score_fortune500(school, None),
            }
            total = sum(dims.values()) / len(dims)
        else:
            total = 50

        # 深度分析文本
        analysis = self._build_deep_analysis(major, school, user)

        # 适合谁 / 不适合谁
        for_whom, against_whom = self._build_suitability(major, school)

        # 引用
        quote = self._select_quote(major, school)

        thinking = [
            ThinkingStep(step="8维度扫描", analysis="应用张雪峰8条决策启发式进行全面评估"),
            ThinkingStep(step="数据交叉验证", analysis="结合本地估算就业率、估算薪资中位数、不可替代性等指标"),
            ThinkingStep(step="结论输出", analysis=f"综合评分{round(total)}分，{'推荐' if total > 70 else '谨慎考虑' if total > 50 else '不推荐'}"),
        ]

        return AnalyzeResponse(
            target=target_name + (f"@{school_name}" if school_name else ""),
            target_type=target_type,
            deep_analysis=analysis,
            eight_dimensions=dims,
            suitability_score=round(total),
            for_whom=for_whom,
            against_whom=against_whom,
            thinking_process=thinking,
            zxf_quote=quote,
        )

    def _build_deep_analysis(self, major: Optional[dict], school: Optional[dict], user: Optional[UserPreferences]) -> str:
        parts = []
        if major:
            parts.append(f"**{major['name']}**（{major['category']}）")
            parts.append(f"估算就业率：{int(major['employment_rate']*100)}% | 5年估算中位数薪资：{major['salary_median_5yr']//1000}K")
            parts.append(f"技术壁垒估算：{major['irreplaceability']}/100 | {'需深造' if major['requires_grad_school'] else '本科可就业'}")
            parts.append(f"标签：{', '.join(major['tags'])}")
            parts.append(f"风险因素：{', '.join(major['risk_factors'])}")
        if school:
            parts.append(f"\n**{school['name']}**（{school['level']}）")
            parts.append(f"位于{school['tier']}城市{school['city']} | 平均薪资{school['average_salary']//1000}K")
            parts.append(f"标签：{', '.join(school['tags'])}")
        return "\n".join(parts)

    def _build_suitability(self, major: Optional[dict], school: Optional[dict]):
        if not major:
            return "适合对该院校层次认可的学生", "不适合对专业有特定要求的学生"

        tags = major.get("tags", [])
        if "天坑" in tags:
            for_whom = "有学术追求、能接受长期深造、家庭经济条件优越的学生"
            against = "普通家庭、追求快速就业、经济压力大的学生"
        elif "热门" in tags and "技术壁垒" in tags:
            for_whom = "数理基础好、愿意持续学习技术、追求高薪就业的学生"
            against = "对技术完全不感兴趣、抗拒持续学习新技能的学生"
        elif "看背景" in tags:
            for_whom = "家庭有相关行业资源、能接受从基层做起的学生"
            against = "普通家庭、没有行业人脉、期望高起点就业的学生"
        else:
            for_whom = "对该领域有真实兴趣、愿意深耕的学生"
            against = "随波逐流、没有明确方向的学生"

        return for_whom, against

    def _select_quote(self, major: Optional[dict], school: Optional[dict]) -> str:
        if not major:
            return "选择比努力更重要，但'有得选'的前提是你足够努力。"

        tags = major.get("tags", [])
        if "天坑" in tags:
            return "生化环材四天王，没读博士别逞强。"
        if "看背景" in tags:
            return "金融不能碰，除非家里是搞金融的。"
        if "热门" in tags and major.get("salary_median_5yr", 0) > 15000:
            return "你的工资，永远和你的不可替代性成正比。"
        return "不看前3%的天才，不看后5%的极端，看中间50%的普通人。"


agent_engine = AgentEngine()
