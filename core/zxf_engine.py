import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any
from core.models import OptionItem, UserProfile, EvaluationResult, HeuristicScore


class DataSource:
    """mock数据源，后续可替换为真实数据库"""

    def __init__(self):
        from data import majors, schools
        self.majors = {m["id"]: m for m in majors}
        self.majors_by_name = {m["name"]: m for m in majors}
        enriched_schools = self._merge_school_locations(schools)
        self.schools = {s["id"]: s for s in enriched_schools}
        self.schools_by_name = {s["name"]: s for s in enriched_schools}
        self.school_locations = {
            s["name"]: {"province": s.get("province", ""), "city": s.get("city", "")}
            for s in enriched_schools
        }

    def _merge_school_locations(self, schools: list[dict]) -> list[dict]:
        location_map = self._load_school_location_map()
        if not location_map:
            return [dict(s) for s in schools]

        merged = []
        for school in schools:
            item = dict(school)
            location = location_map.get(self._normalize_school_name(item.get("name", "")))
            if location:
                item["province"] = location.get("province") or item.get("province")
                item["city"] = location.get("city") or item.get("city")
                tags = list(item.get("tags") or [])
                for value in [item.get("province"), item.get("city")]:
                    if value and value not in tags:
                        tags.append(value)
                item["tags"] = tags
                item["location_source"] = "高校省市地址"
            merged.append(item)
        return merged

    def _load_school_location_map(self) -> dict[str, dict]:
        root = Path(__file__).resolve().parent.parent
        candidates = sorted(root.glob("高校省市地址*.json"))
        if not candidates:
            return {}

        try:
            rows = json.loads(candidates[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        locations: dict[str, dict] = {}
        for row in rows if isinstance(rows, list) else []:
            name = str(row.get("name", "")).strip()
            province = str(row.get("province", "")).strip()
            city = str(row.get("city", "")).strip()
            if not name or not province or not city:
                continue
            locations[self._normalize_school_name(name)] = {"province": province, "city": city}
        return locations

    @staticmethod
    def _normalize_school_name(name: str) -> str:
        value = str(name or "").strip()
        value = value.replace("（", "(").replace("）", ")")
        return re.sub(r"\s+", "", value)

    def get_major(self, name: str) -> dict | None:
        return self.majors_by_name.get(name)

    def get_school(self, name: str) -> dict | None:
        return self.schools_by_name.get(name)

    def search_majors(self, keyword: str = "", category: str = "", tags: list = None) -> list:
        results = []
        for m in self.majors.values():
            if keyword and keyword not in m["name"]:
                continue
            if category and m["category"] != category:
                continue
            if tags and not any(t in m["tags"] for t in tags):
                continue
            results.append(m)
        return results

    def search_schools(self, city: str = "", level: str = "", tier: str = "") -> list:
        results = []
        for s in self.schools.values():
            if city and city not in s["city"]:
                continue
            if level and s["level"] != level:
                continue
            if tier and s["tier"] != tier:
                continue
            results.append(s)
        return results


class ZXFEvaluator:
    """张雪峰决策评估引擎：基于5个心智模型 + 8条决策启发式的规则引擎"""

    # 权重配置：不同家庭条件对各项指标的重视程度不同
    WEIGHTS = {
        "普通家庭": {
            "employment_reversal": 1.3,
            "social_sieve": 1.2,
            "irreplaceability": 1.2,
            "median_principle": 1.2,
            "family_background": 1.3,
            "city_priority": 1.0,
            "fortune500_test": 1.1,
            "ten_year_test": 1.1,
        },
        "中产": {
            "employment_reversal": 1.1,
            "social_sieve": 1.1,
            "irreplaceability": 1.1,
            "median_principle": 1.0,
            "family_background": 1.0,
            "city_priority": 1.1,
            "fortune500_test": 1.0,
            "ten_year_test": 1.0,
        },
        "富裕": {
            "employment_reversal": 0.8,
            "social_sieve": 0.9,
            "irreplaceability": 0.9,
            "median_principle": 0.8,
            "family_background": 0.7,
            "city_priority": 1.2,
            "fortune500_test": 0.8,
            "ten_year_test": 0.7,
        },
    }

    # 灵魂追问字段定义
    SOUL_QUESTIONS = [
        {"field": "score", "question": "孩子高考多少分？全省位次大概是多少？", "priority": "critical"},
        {"field": "province", "question": "哪个省的？", "priority": "critical"},
        {"field": "family_background", "question": "家里是做什么的？经济条件怎么样？", "priority": "critical"},
        {"field": "rank", "question": "全省排名/位次是多少？", "priority": "high"},
        {"field": "city_preference", "question": "能接受去哪些城市？有没有绝对不能去的？", "priority": "high"},
        {"field": "willing_grad_school", "question": "孩子能接受读研/读博吗？", "priority": "high"},
        {"field": "industry_acceptance", "question": "对什么行业完全不感兴趣？", "priority": "medium"},
    ]

    def __init__(self):
        self.data = DataSource()

    def generate_soul_questions(self, known_info: dict) -> dict:
        """根据已知信息生成灵魂追问"""
        missing = []
        for sq in self.SOUL_QUESTIONS:
            if sq["field"] not in known_info or known_info[sq["field"]] is None:
                missing.append(sq)

        critical = [m["field"] for m in missing if m["priority"] == "critical"]

        return {
            "questions": missing,
            "missing_critical": critical,
        }

    def evaluate(self, options: List[OptionItem], profile: UserProfile) -> EvaluationResult:
        """对一组选项进行综合评估"""
        if not options:
            return EvaluationResult(
                recommendation="",
                scores={},
                analysis="没有提供可评估的选项。",
                red_flags=[],
                uncertainties=[],
            )

        family = profile.family_background or "普通家庭"
        weights = self.WEIGHTS.get(family, self.WEIGHTS["普通家庭"])

        all_scores = {}
        red_flags = []
        uncertainties = []

        for idx, opt in enumerate(options):
            scores = self._evaluate_option(opt, profile)
            weighted_total = sum(
                scores.get(k, 50) * weights.get(k, 1.0)
                for k in weights.keys()
            ) / sum(weights.values())
            scores["total"] = round(weighted_total)
            all_scores[f"option_{idx}"] = scores

            # 收集红旗信号
            flags = self._collect_red_flags(opt, profile, scores)
            red_flags.extend(flags)

            # 收集不确定性
            uncs = self._collect_uncertainties(opt, profile)
            uncertainties.extend(uncs)

        # 推荐最优选项
        best_key = max(all_scores, key=lambda k: all_scores[k]["total"])
        best_idx = int(best_key.split("_")[1])
        best_option = options[best_idx]

        analysis = self._generate_analysis(options, all_scores, best_option, profile)

        return EvaluationResult(
            recommendation=best_key,
            scores=all_scores,
            analysis=analysis,
            red_flags=list(set(red_flags)),
            uncertainties=list(set(uncertainties)),
        )

    def _evaluate_option(self, opt: OptionItem, profile: UserProfile) -> dict:
        """对单个选项应用8条启发式评分"""
        scores = {}

        # 获取专业和学校数据
        major = self.data.get_major(opt.name) if opt.type == "major" else None
        school = self.data.get_school(opt.school) if opt.school else None

        # 1. 就业倒推法 (employment_reversal): 就业率、薪资、行业前景
        scores["employment_reversal"] = self._score_employment_reversal(major, school)

        # 2. 社会筛子论 (social_sieve): 学历/学校在社会筛选中的位置
        scores["social_sieve"] = self._score_social_sieve(school, major)

        # 3. 不可替代性 (irreplaceability): 技术壁垒
        scores["irreplaceability"] = self._score_irreplaceability(major)

        # 4. 中位数原则 (median_principle): 普通毕业生的真实去向
        scores["median_principle"] = self._score_median(major, school)

        # 5. 家庭背景分流 (family_background)
        scores["family_background"] = self._score_family_fit(major, profile)

        # 6. 城市优先 (city_priority)
        scores["city_priority"] = self._score_city_priority(school, profile)

        # 7. 500强测试 (fortune500_test)
        scores["fortune500_test"] = self._score_fortune500(school, major)

        # 8. 10年后压迫测试 (ten_year_test)
        scores["ten_year_test"] = self._score_ten_year(major, profile)

        return scores

    def _score_employment_reversal(self, major: dict | None, school: dict | None) -> int:
        """就业倒推法：就业率越高、薪资越高，得分越高"""
        if not major:
            return 50
        emp = major.get("employment_rate", 0.5)
        salary = major.get("salary_median_5yr", 8000)
        # 就业率权重60%，薪资权重40%
        emp_score = emp * 100
        salary_score = min(salary / 250, 100)  # 25000封顶
        return round(emp_score * 0.6 + salary_score * 0.4)

    def _score_social_sieve(self, school: dict | None, major: dict | None) -> int:
        """社会筛子论：学校层次越高、就业率越高，得分越高"""
        if not school and not major:
            return 50
        school_level_map = {
            "985": 95,
            "211": 80,
            "双一流": 76,
            "普通一本": 65,
            "普通二本": 45,
        }
        school_score = school_level_map.get(school["level"], 50) if school else 50
        major_emp = major.get("employment_rate", 0.5) * 100 if major else 50
        return round(school_score * 0.6 + major_emp * 0.4)

    def _score_irreplaceability(self, major: dict | None) -> int:
        """不可替代性：技术壁垒高的专业得分高"""
        if not major:
            return 50
        return major.get("irreplaceability", 50)

    def _score_median(self, major: dict | None, school: dict | None) -> int:
        """中位数原则：普通毕业生的真实水平"""
        if not major:
            return 50
        entry = major.get("salary_entry", 5000)
        median = major.get("salary_median_5yr", 8000)
        # 起薪权重30%，5年中位数权重70%
        entry_score = min(entry / 150, 100)
        median_score = min(median / 250, 100)
        return round(entry_score * 0.3 + median_score * 0.7)

    def _score_family_fit(self, major: dict | None, profile: UserProfile) -> int:
        """家庭背景分流：普通家庭需要确定性高的选择"""
        if not major:
            return 50
        family = profile.family_background or "普通家庭"

        # 天坑专业标签对普通家庭是致命扣分项
        tags = major.get("tags", [])
        risk_factors = major.get("risk_factors", [])

        base = 70
        if "天坑" in tags:
            if family == "普通家庭":
                base -= 35
            elif family == "中产":
                base -= 20
            else:
                base -= 10

        # 需要深造的专业对普通家庭压力大
        if major.get("requires_grad_school", False):
            if family == "普通家庭":
                base -= 15
            elif family == "中产":
                base -= 5

        # 看背景的专业对没背景的家庭不利
        if "看背景" in tags:
            if family == "普通家庭":
                base -= 25
            elif family == "中产":
                base -= 10

        return max(10, min(100, base))

    def _score_city_priority(self, school: dict | None, profile: UserProfile) -> int:
        """城市优先：学校所在城市与用户偏好的匹配度"""
        if not school:
            return 50
        city = school.get("city", "")
        tier = school.get("tier", "")
        prefs = profile.city_preference or []

        # 城市层级得分
        tier_score = {"一线": 95, "新一线": 85, "二线": 70, "三线": 50, "四线": 30}
        score = tier_score.get(tier, 50)

        # 匹配用户偏好
        if prefs:
            if tier in prefs:
                score += 10
            else:
                score -= 15

        return max(10, min(100, score))

    def _score_fortune500(self, school: dict | None, major: dict | None) -> int:
        """500强测试：名校+热门专业更容易进入大企业"""
        if not school and not major:
            return 50
        school_level_map = {"985": 95, "211": 80, "双一流": 76, "普通一本": 55, "普通二本": 30}
        school_score = school_level_map.get(school["level"], 50) if school else 50

        # 热门专业加分
        hot_tags = {"热门", "高薪", "技术壁垒", "前沿"}
        major_tags = set(major.get("tags", [])) if major else set()
        bonus = 10 if major_tags & hot_tags else 0

        return min(100, school_score + bonus)

    def _score_ten_year(self, major: dict | None, profile: UserProfile) -> int:
        """10年后压迫测试：这个选择10年后是否还能站得住"""
        if not major:
            return 50
        risks = major.get("risk_factors", [])
        tags = major.get("tags", [])
        base = 70

        # AI替代风险
        if any("AI替代" in r for r in risks):
            base -= 20
        if any("替代" in r for r in risks):
            base -= 15

        # 行业萎缩
        if any("萎缩" in r or "下行" in r for r in risks):
            base -= 20

        # 技术壁垒高的专业10年后更稳
        if "技术壁垒" in tags or major.get("irreplaceability", 0) > 80:
            base += 15

        # 稳定型专业
        if "稳定" in tags:
            base += 10

        return max(10, min(100, base))

    def _collect_red_flags(self, opt: OptionItem, profile: UserProfile, scores: dict) -> list:
        """收集红旗信号：需要警惕的重大问题"""
        flags = []
        family = profile.family_background or "普通家庭"
        major = self.data.get_major(opt.name) if opt.type == "major" else None

        if not major:
            return flags

        if "天坑" in major.get("tags", []) and family == "普通家庭":
            flags.append(f"{opt.name}被标记为'天坑'专业，普通家庭需谨慎")

        if major.get("requires_grad_school", False) and family == "普通家庭":
            flags.append(f"{opt.name}本科几乎无法就业，必须深造，周期长、成本高")

        if "看背景" in major.get("tags", []) and family != "富裕":
            flags.append(f"{opt.name}极其依赖家庭背景，没资源进去是边缘岗位")

        if scores.get("employment_reversal", 100) < 40:
            flags.append(f"{opt.name}就业率/薪资数据不乐观")

        if scores.get("ten_year_test", 100) < 40:
            flags.append(f"{opt.name}10年后前景存在较大风险")

        return flags

    def _collect_uncertainties(self, opt: OptionItem, profile: UserProfile) -> list:
        """收集需要进一步确认的信息"""
        uncs = []
        if not profile.rank:
            uncs.append("缺少全省位次信息，无法精确匹配院校")
        if not profile.willing_grad_school:
            uncs.append("未确认是否接受深造")
        return uncs

    def _generate_analysis(self, options: List[OptionItem], all_scores: dict,
                           best_option: OptionItem, profile: UserProfile) -> str:
        """生成文本分析"""
        family = profile.family_background or "普通家庭"
        best_key = max(all_scores, key=lambda k: all_scores[k]["total"])
        best_scores = all_scores[best_key]

        lines = [
            f"基于你提供的信息（家庭背景：{family}），对{len(options)}个选项进行了8维度评估。",
            "",
            f"**最优推荐：{best_option.name}**（综合得分 {best_scores['total']}）",
            "",
            "**各维度得分对比：**",
        ]

        for idx, opt in enumerate(options):
            key = f"option_{idx}"
            sc = all_scores[key]
            lines.append(f"- {opt.name}：总分{sc['total']} | 就业{sc['employment_reversal']} | 筛子{sc['social_sieve']} | 不可替代{sc['irreplaceability']} | 家庭适配{sc['family_background']} | 城市{sc['city_priority']} | 10年后{sc['ten_year_test']}")

        lines.extend([
            "",
            "**关键判断：**",
        ])

        # 根据最高分维度生成判断
        best_dim = max(
            [(k, v) for k, v in best_scores.items() if k != "total"],
            key=lambda x: x[1]
        )
        dim_names = {
            "employment_reversal": "就业数据",
            "social_sieve": "社会认可度",
            "irreplaceability": "技术壁垒",
            "median_principle": "中位数收益",
            "family_background": "家庭适配度",
            "city_priority": "城市优势",
            "fortune500_test": "企业认可度",
            "ten_year_test": "长期前景",
        }
        lines.append(f"- {best_option.name}在'{dim_names.get(best_dim[0], best_dim[0])}'维度表现最强（{best_dim[1]}分）")

        if best_scores["family_background"] >= 80:
            lines.append(f"- 该选项与'{family}'背景高度适配，试错成本可控")
        elif best_scores["family_background"] < 50:
            lines.append(f"- ⚠️ 该选项与'{family}'背景匹配度较低，需充分评估风险")

        if best_scores["ten_year_test"] >= 80:
            lines.append("- 10年后压迫测试通过：该方向长期趋势稳定")
        elif best_scores["ten_year_test"] < 50:
            lines.append("- ⚠️ 10年后压迫测试警告：需警惕行业长期风险")

        return "\n".join(lines)


data_source = DataSource()
evaluator = ZXFEvaluator()
