import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.agent_engine import agent_engine
from core.answer_guard import answer_guard
from core.family_risk import build_family_risk_profile
from core.llm_client import llm_client
from core.models import ConsultRecommendationPlan, ConsultRequest, ConsultResponse, InsightRequest, RecommendRequest, RecommendResponse, ThinkingStep, UserPreferences, UserProfile
from core.research_client import ResearchResult, web_research_client
from data import majors, school_admissions_urls, schools


PROVINCES = [
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海", "江苏", "浙江",
    "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "广西", "海南", "重庆",
    "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
]

COMMON_CITY_NAMES = [
    "北京", "上海", "天津", "重庆", "南京", "苏州", "无锡", "常州", "南通", "徐州",
    "杭州", "宁波", "温州", "广州", "深圳", "佛山", "东莞", "成都", "武汉", "西安",
    "郑州", "长沙", "合肥", "福州", "厦门", "南昌", "济南", "青岛", "烟台", "威海",
    "潍坊", "临沂", "淄博", "济宁", "泰安", "哈尔滨", "长春", "沈阳", "大连",
]

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


def _load_location_city_names() -> list[str]:
    root = Path(__file__).resolve().parent.parent
    candidates = sorted(root.glob("高校省市地址*.json"))
    if not candidates:
        return []
    try:
        rows = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    cities = []
    for row in rows if isinstance(rows, list) else []:
        city = str(row.get("city", "")).strip()
        if city and city not in cities:
            cities.append(city)
    return cities


for _city_name in _load_location_city_names():
    if _city_name not in COMMON_CITY_NAMES:
        COMMON_CITY_NAMES.append(_city_name)

FACT_KEYWORDS = [
    "行业", "政策", "就业", "薪资", "工资", "中位数",
    "录取", "分数线", "位次", "排名", "保研", "招生", "500强", "前景", "数据",
    "壁垒", "不可替代", "压力测试", "10年后", "十年后",
]

RECOMMEND_KEYWORDS = [
    "推荐", "报考", "志愿", "冲稳保", "去哪里", "去哪", "能上", "适合哪些学校",
    "该报", "报什么", "报哪些", "报什么样", "哪些学校", "什么学校", "什么样学校",
    "什么样的学校", "学校专业", "院校专业", "该冲", "冲什么", "冲哪些", "冲刺",
    "该稳", "稳哪些", "保底", "能报", "选什么学校", "选哪些学校",
]
INSIGHT_KEYWORDS = [
    "怎么样", "靠谱吗", "值不值", "前景", "就业", "薪资", "分析",
    "可以吗", "可不可以", "能不能", "适合", "想学", "要不要学", "能学", "好吗",
    "壁垒", "不可替代", "压力测试", "10年后", "十年后",
]

MAJOR_ALIASES = {
    "计算机": "计算机科学与技术",
    "人工智能": "人工智能",
    "电子信息": "电子信息工程",
    "软件": "软件工程",
    "通信": "通信工程",
    "电气": "电气工程及其自动化",
    "电气自动化": "电气工程及其自动化",
    "机械": "机械设计制造及其自动化",
    "自动化": "自动化",
    "大数据": "数据科学与大数据技术",
    "数据科学": "数据科学与大数据技术",
    "信息安全": "信息安全",
    "物联网": "物联网工程",
    "金融": "金融学",
    "经济": "经济学",
    "会计": "会计学",
    "统计": "应用统计学",
    "新闻": "新闻学",
    "医学": "临床医学",
    "口腔": "口腔医学",
    "法学": "法学",
    "数学": "数学与应用数学",
    "物理": "物理学",
    "化学": "化学",
    "土木": "土木工程",
    "建筑": "建筑学",
    "材料": "材料科学与工程",
    "化工": "化学工程与工艺",
    "生物": "生物科学",
    "地理": "地理科学",
    "历史": "历史学",
    "地理历史": "地理科学",
    "中文": "汉语言文学",
    "汉语言": "汉语言文学",
    "英语": "英语",
}

SCHOOL_ALIASES = {
    "华电": "华北电力大学",
    "华北电力": "华北电力大学",
    "华北电力大学北京": "华北电力大学",
    "华北电力大学北京校区": "华北电力大学",
    "华北电力北京校区": "华北电力大学",
}


@dataclass
class IntentResult:
    intent: str
    school_names: list[str]
    major_names: list[str]
    needs_research: bool


class ConsultOrchestrator:
    """把自然语言咨询编排为：参数提取 -> 联网研究 -> Agent调用 -> LLM表达。"""

    def __init__(self):
        self.school_names = [item["name"] for item in schools]
        self.major_names = [item["name"] for item in majors]
        self.school_by_name = {item["name"]: item for item in schools}
        self.major_by_name = {item["name"]: item for item in majors}
        self.school_admissions_by_name = school_admissions_urls.get("schools", {}) if isinstance(school_admissions_urls, dict) else {}

    def consult(self, request: ConsultRequest, history: list[dict] | None = None) -> ConsultResponse:
        enriched = self._enrich_request_context(request)
        intent = self._detect_intent(enriched)
        research_results = self._research_if_needed(enriched, intent)
        extra_parts = []
        recommendation_plans: list[ConsultRecommendationPlan] = []
        citations = [item.url for item in research_results]
        research_status = self._research_status_text(research_results)

        if intent.intent == "pressure_test":
            return self._build_pressure_test_response(enriched, intent, citations)

        if research_results:
            extra_parts.append(web_research_client.build_summary(research_results))
            extra_parts.append(research_status)

        profile_context = self._build_profile_context(enriched)
        if profile_context:
            extra_parts.append(profile_context)
        major_scope_context = self._build_major_scope_context(enriched, intent)
        if major_scope_context:
            extra_parts.append(major_scope_context)
        if self._is_fact_data_question(enriched.question):
            extra_parts.append(
                "本轮识别为数据/事实咨询：只回答用户正在问的中位数、薪资、就业或500强招聘问题；"
                "不要自动改写成冲稳保院校推荐，也不要输出[院校推荐]段落。"
            )

        if intent.intent == "recommend":
            user = self._build_user_preferences(enriched)
            if user:
                recommend = agent_engine.recommend(RecommendRequest(user=user, limit=10))
                plan_research = self._research_recommendation_plans(enriched, recommend)
                if plan_research:
                    research_results.extend(plan_research)
                    citations = [item.url for item in research_results]
                    research_status = self._research_status_text(research_results)
                    extra_parts.append(web_research_client.build_summary(plan_research))
                    extra_parts.append(research_status)
                extra_parts.append(self._format_recommend_context(recommend, user))
                recommendation_plans = self._build_structured_recommendations(
                    recommend=recommend,
                    user=user,
                    citations=citations,
                )
                if recommendation_plans:
                    extra_parts.append(
                        "结构化推荐已由后端 Agent 生成；最终话术必须继续交给 API 模型结合检索摘要表达，"
                        "不要用本地模板直接覆盖模型回答。"
                    )
            else:
                extra_parts.append(
                    "Agent推荐状态：信息不足，无法调用 /api/agent/recommend。"
                    "必须追问省份、分数、位次、选科、城市偏好、专业偏好。"
                )
        elif intent.intent == "school_chance":
            school_context = self._build_school_chance_context(enriched, intent)
            if school_context:
                extra_parts.append(school_context)
        elif intent.intent == "insight":
            insight_context = self._build_insight_context(enriched, intent)
            if insight_context:
                extra_parts.append(insight_context)
        else:
            strategy_context = self._build_profile_strategy_context(enriched)
            if strategy_context:
                extra_parts.append(strategy_context)

        if not extra_parts:
            extra_parts.append(
                "系统提示：未触发结构化Agent。仍必须调用DeepSeek按张雪峰表达方式回答用户原始问题；"
                "优先结合已有考生画像，不要机械要求用户重复补全画像。"
            )

        extra_parts.append(self._build_admission_score_research_policy())
        extra_parts.append(self._build_data_honesty_context())

        citations = [item.url for item in research_results]
        llm_history = self._history_for_current_question(history, enriched.question, intent)
        response = llm_client.consult(
            enriched,
            extra_context="\n\n".join(extra_parts),
            citations=citations,
            history=llm_history,
        )
        if self._is_fact_data_question(enriched.question):
            response = self._guard_fact_data_response(response, enriched, intent, citations)
        elif intent.intent != "recommend":
            response = self._guard_non_recommend_response(response, enriched, intent, citations)
        if intent.intent == "recommend":
            user = self._build_user_preferences(enriched)
            if not user:
                response = self._guard_insufficient_recommendation_response(response, enriched)
                answer_plans = []
            else:
                answer_plans = [] if self._answer_declines_recommendations(response.answer) else self._extract_recommendations_from_answer(
                    answer=response.answer,
                    user=user,
                    citations=citations,
                )
            if not recommendation_plans and len(answer_plans) > len(recommendation_plans):
                recommendation_plans = answer_plans
            if recommendation_plans and self._should_use_template_recommend_answer(response.answer, recommendation_plans):
                response.answer = self._compose_recommendation_answer(
                    enriched,
                    recommendation_plans,
                    self._research_status_text(research_results),
                )
            if recommendation_plans:
                response = answer_guard.guard_response(
                    response,
                    extra_context="\n\n".join(extra_parts),
                    citations=citations,
                    allowed_schools=[plan.school for plan in recommendation_plans],
                    known_school_names=self.school_names,
                    require_recommendation_guard=True,
                )
        response.recommendation_plans = recommendation_plans
        return response

    def _guard_insufficient_recommendation_response(
        self,
        response: ConsultResponse,
        request: ConsultRequest,
    ) -> ConsultResponse:
        response.recommendation_plans = []
        mentions_school = any(name in (response.answer or "") for name in self.school_names)
        if not mentions_school and not self._is_unrequested_recommendation_answer(response.answer, request, IntentResult("chat", [], [], False)):
            response.follow_up_questions = response.follow_up_questions or [
                "孩子是哪个省的？",
                "高考分数和位次是多少？",
                "选科和目标专业方向是什么？",
            ]
            return response

        response.answer = (
            "[核心判断]\n"
            "现在不能直接给学校名单。缺省份、分数、位次、选科或专业方向时，硬推荐学校就是瞎排，容易把孩子带沟里。\n\n"
            "[灵魂追问]\n"
            "1. 孩子是哪个省的？\n"
            "2. 高考分数和位次是多少？\n"
            "3. 选科、目标城市和专业方向分别是什么？\n\n"
            "[核验清单]\n"
            "补齐画像后，再按省考试院投档表、学校招生网专业组和招生计划做冲稳保。"
        )
        response.follow_up_questions = [
            "孩子是哪个省的？",
            "高考分数和位次是多少？",
            "选科、目标城市和专业方向分别是什么？",
        ]
        response.confidence = "low"
        return response

    def _history_for_current_question(
        self,
        history: list[dict] | None,
        question: str,
        intent: IntentResult,
    ) -> list[dict] | None:
        if not history or intent.intent == "recommend":
            return history
        filtered: list[dict] = []
        recommendation_markers = ["[院校推荐]", "【院校推荐】", "冲稳保", "冲刺", "保底", "同步方案"]
        for message in history:
            role = message.get("role") if isinstance(message, dict) else None
            content = str(message.get("content", "")) if isinstance(message, dict) else ""
            if role == "assistant" and any(marker in content for marker in recommendation_markers):
                continue
            filtered.append(message)
        return filtered[-6:]

    def _should_use_template_recommend_answer(self, answer: str, plans: list[ConsultRecommendationPlan]) -> bool:
        if not plans:
            return False
        if not (answer or "").strip():
            return True
        visible_school_count = sum(1 for plan in plans[:10] if plan.school in answer)
        has_recommendation_heading = bool(re.search(
            r"(?:^|\n)\s*(?:\[|【)?\s*(?:院校推荐|推荐院校|学校推荐|推荐学校|冲稳保推荐|具体推荐|方案推荐)\s*(?:\]|】)?",
            answer or "",
        ))
        if visible_school_count == 0 and not has_recommendation_heading:
            return True
        return False

    def _guard_non_recommend_response(
        self,
        response: ConsultResponse,
        request: ConsultRequest,
        intent: IntentResult,
        citations: list[str],
    ) -> ConsultResponse:
        """Prevent non-recommend consultations from being overwritten by recommendation templates."""
        response.recommendation_plans = []
        if not self._is_unrequested_recommendation_answer(response.answer, request, intent):
            if self._answer_conflicts_with_major_scope(response.answer, request, intent):
                if intent.intent == "school_chance":
                    response.answer = self._build_school_chance_fallback_answer(request, intent, citations)
                elif intent.intent == "insight":
                    response.answer = self._build_local_insight_answer(request, intent, citations)
                else:
                    response.answer = self._build_chat_scope_fallback_answer(request, citations)
                response.follow_up_questions = []
                response.confidence = "medium"
            return response

        if intent.intent == "school_chance":
            response.answer = self._build_school_chance_fallback_answer(request, intent, citations)
        elif intent.intent == "insight":
            response.answer = self._build_local_insight_answer(request, intent, citations)
        else:
            response.answer = self._build_chat_scope_fallback_answer(request, citations)
        response.follow_up_questions = []
        response.confidence = "medium"
        return response

    def _build_major_scope_context(self, request: ConsultRequest, intent: IntentResult) -> str:
        majors = self._active_major_scope(request, intent)
        if not majors:
            return ""
        return (
            "专业一致性要求：\n"
            f"1. 本轮允许使用的专业方向是：{'、'.join(majors)}。\n"
            "2. 如果用户没有明确说要换专业、比较专业或看调剂，不要把回答主语改成其他专业。\n"
            "3. 学校名里的“师范、医科、财经、建筑、农业、政法”等字样只是院校类型，不等于用户专业方向。"
        )

    def _active_major_scope(self, request: ConsultRequest, intent: IntentResult) -> list[str]:
        candidates: list[str] = []
        if intent.major_names:
            candidates.extend(intent.major_names)
        elif request.context and request.context.major_preference:
            candidates.extend(request.context.major_preference)
        expanded = self._expand_major_preferences(candidates)
        if expanded:
            return expanded
        normalized: list[str] = []
        for item in candidates:
            value = str(item or "").strip()
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    def _answer_conflicts_with_major_scope(
        self,
        answer: str,
        request: ConsultRequest,
        intent: IntentResult,
    ) -> bool:
        allowed = self._active_major_scope(request, intent)
        if not allowed or self._allows_major_scope_change(request.question):
            return False
        mentioned = self._extract_major_preference(answer)
        conflicting = [major for major in mentioned if major not in allowed]
        if not conflicting:
            return False

        compact = re.sub(r"\s+", "", answer or "")
        allowed_mentioned = any(self._contains_major_alias(compact, major) for major in allowed)
        if not allowed_mentioned:
            return True

        focus_markers = [
            "当前方向", "目标方向", "专业方向", "目标专业", "本轮按", "按", "报", "读",
            "选择", "考虑", "核验", "这个专业", "该专业",
        ]
        for major in conflicting:
            aliases = self._major_aliases(major)
            for alias in aliases:
                if not alias or alias not in compact:
                    continue
                if any(re.search(rf"{re.escape(marker)}.{{0,12}}{re.escape(alias)}", compact) for marker in focus_markers):
                    return True
                if re.search(rf"{re.escape(alias)}.{{0,8}}(?:方向|专业|这个专业|该专业|就业|师范|非师范)", compact):
                    return True
        return False

    def _allows_major_scope_change(self, question: str) -> bool:
        compact = re.sub(r"\s+", "", question or "")
        return any(
            marker in compact
            for marker in [
                "换专业", "转专业", "改专业", "换方向", "转方向", "改方向",
                "对比", "比较", "另一个专业", "其他专业", "其它专业", "相近专业",
                "调剂专业", "专业组里", "专业组内",
            ]
        )

    def _major_aliases(self, major: str) -> list[str]:
        aliases = [major]
        for alias, target in MAJOR_ALIASES.items():
            if target == major and alias not in aliases:
                aliases.append(alias)
        return aliases

    def _contains_major_alias(self, compact_text: str, major: str) -> bool:
        return any(alias and alias in compact_text for alias in self._major_aliases(major))

    def _is_unrequested_recommendation_answer(
        self,
        answer: str,
        request: ConsultRequest,
        intent: IntentResult,
    ) -> bool:
        compact = re.sub(r"\s+", "", answer or "")
        if not compact:
            return True
        recommend_markers = [
            "[院校推荐]", "【院校推荐】", "[推荐院校]", "【推荐院校】",
            "冲稳保方案", "院校推荐主回答", "同步方案", "待审核方案",
            "冲刺方案", "稳妥方案", "保底方案",
        ]
        if any(marker in compact for marker in recommend_markers):
            return True
        if intent.intent == "school_chance":
            target_school = intent.school_names[0] if intent.school_names else ""
            mentioned = [name for name in self.school_names if name in (answer or "")]
            other_schools = [name for name in mentioned if name != target_school]
            if len(other_schools) >= 2:
                return True
        off_topic_markers = [
            "哪个学校名字好听", "哪所学校名字好听", "先别问哪个学校",
            "别先问哪个学校", "先问这条路能不能换饭碗",
        ]
        return any(marker in compact for marker in off_topic_markers)

    def _build_school_chance_fallback_answer(
        self,
        request: ConsultRequest,
        intent: IntentResult,
        citations: list[str],
    ) -> str:
        school_name = intent.school_names[0] if intent.school_names else ""
        user = self._build_user_preferences(request, allow_partial=True)
        ctx = request.context
        target_major = self._recommend_major_focus(user.major_preference if user else (ctx.major_preference if ctx else None))
        school = self.school_by_name.get(school_name, {})
        major = self.major_by_name.get(target_major) or (self.major_by_name.get((user.major_preference or [""])[0]) if user and user.major_preference else {})

        risk_text = "本地规则没有形成稳定档位，需要回到考试院投档表和学校招生网核验。"
        if user and user.province and user.score:
            try:
                recommend = agent_engine.recommend(RecommendRequest(user=user, limit=20))
                matched_plan = next((plan for plan in recommend.plans if plan.school == school_name), None)
                if matched_plan:
                    risk_text = f"本地规则把{school_name}放在「{matched_plan.risk_level}」档，专业方向按「{matched_plan.major}」核验。"
                elif school and major:
                    risk = agent_engine._profile_risk_bucket(school, major, user)
                    risk_text = f"本地规则粗判属于「{risk}」档，但没有进入当前结构化候选，必须单独查官方位次。"
            except Exception:
                pass

        answer = (
            "[核心判断]\n"
            f"这轮只判断{school_name or '这所学校'}，不展开多校推荐。{risk_text}\n\n"
            "[分析过程]\n"
            f"1. 考生画像：{self._profile_brief(ctx)}。\n"
            f"2. 目标专业方向：{target_major}。\n"
            f"3. 学校核验入口：{(school.get('official_url') if school else None) or '未命中本地官网，需要手动查学校本科招生网'}。\n\n"
            "[核验清单]\n"
            "1. 查本省教育考试院近三年投档位次，先看这所学校对应专业组，不拿学校最低线代替专业线。\n"
            "2. 查学校本科招生网，确认招生计划、选科要求、专业组包含专业和调剂规则。\n"
            "3. 如果本轮没有官方来源支撑，只能当作本地规则粗判，不能当录取承诺。"
        )
        if citations:
            answer += "\n\n数据来源：\n" + "\n".join(f"{idx + 1}. {url}" for idx, url in enumerate(citations[:5]))
        return answer

    def _build_local_insight_answer(
        self,
        request: ConsultRequest,
        intent: IntentResult,
        citations: list[str],
    ) -> str:
        user = self._build_user_preferences(request, allow_partial=True)
        target_name = (intent.major_names or intent.school_names or [""])[0]
        if not target_name:
            return self._build_chat_scope_fallback_answer(request, citations)

        target_type = "major" if target_name in intent.major_names else "school"
        insight = agent_engine.insights(
            request=InsightRequest(target_type=target_type, target_name=target_name, user=user)
        )
        if target_type == "major":
            answer = self._format_major_fit_answer(request, target_name, insight)
        else:
            answer = self._format_school_fit_answer(request, target_name, insight)
        answer += "\n\n数据口径：以上为本地专业/院校库和公开核验入口辅助判断，不是多校名单；后续若要报志愿，需要另起一轮单独做方案核验。"
        if citations:
            answer += "\n\n数据来源：\n" + "\n".join(f"{idx + 1}. {url}" for idx, url in enumerate(citations[:5]))
        return answer

    def _build_chat_scope_fallback_answer(self, request: ConsultRequest, citations: list[str]) -> str:
        answer = (
            "[核心判断]\n"
            f"这轮问题是：{request.question}。它没有明确要求列学校，所以不能把回答改成多校名单。\n\n"
            "[分析过程]\n"
            f"已知画像：{self._profile_brief(request.context)}。\n"
            "先围绕本轮问题给判断；如果后续要做学校方案，再单独进入冲稳保推荐流程。\n\n"
            "[核验清单]\n"
            "涉及录取就查教育考试院和学校招生网；涉及就业和收入就查就业质量报告、企业招聘官网和真实岗位样本。"
        )
        if citations:
            answer += "\n\n数据来源：\n" + "\n".join(f"{idx + 1}. {url}" for idx, url in enumerate(citations[:5]))
        return answer

    def _build_pressure_test_response(
        self,
        request: ConsultRequest,
        intent: IntentResult,
        citations: list[str],
    ) -> ConsultResponse:
        target_major = self._fact_target_major(request, intent)
        if not target_major:
            return ConsultResponse(
                answer=(
                    "[核心判断]\n"
                    "可以做10年后压力测试，但当前画像里没有明确专业方向，我不能替你编一个对象。\n\n"
                    "[灵魂追问]\n"
                    "1. 要测试哪个专业或哪套学校-专业方案？\n"
                    "2. 孩子能不能接受读研、考编或转行？\n"
                    "3. 家庭能承受几年试错成本？"
                ),
                thinking_process=[ThinkingStep(step="压力测试", analysis="缺少专业对象，未生成虚构压力测试")],
                follow_up_questions=[
                    "要测试哪个专业或哪套学校-专业方案？",
                    "孩子能不能接受读研、考编或转行？",
                    "家庭能承受几年试错成本？",
                ],
                confidence="low",
                citations=citations,
                recommendation_plans=[],
            )

        user = self._build_user_preferences(request, allow_partial=True)
        insight = agent_engine.insights(
            request=InsightRequest(target_type="major", target_name=target_major, user=user)
        )
        major = self.major_by_name.get(target_major, {})
        family = (request.context.family_background if request.context and request.context.family_background else "普通家庭")
        salary_10y = int(insight.median_salary * 1.6) if insight.median_salary else None
        irreplaceability = insight.irreplaceability
        risks = insight.risk_factors or []
        needs_grad = bool(major.get("requires_grad_school"))
        subjects = user.subjects or (request.context.subjects if request.context and request.context.subjects else "未填")
        city_pref = "、".join(user.city_preference or []) or "未限定"
        risk_pref = user.risk_appetite or "均衡"
        employment_line = self._format_rate(insight.employment_rate)
        current_salary_line = self._format_salary(insight.median_salary)
        barrier_line = self._format_irreplaceability(irreplaceability)
        tags = "、".join(major.get("tags", [])[:4]) or "暂无标签"
        category = major.get("category") or "未分类"

        family_line = (
            "普通家庭的核心不是赌十年后的高薪个例，而是看普通毕业生能不能稳定落地；如果要读研、考证或换城市，这些都要提前算成本。"
            if "普通" in family
            else "家庭试错空间相对更大，可以给成长型方向更多时间，但仍要看岗位样本和毕业去向，不能只看专业名字。"
        )
        subject_line = (
            "选科偏理，若专业也偏技术或工程，长期壁垒更容易做出来；如果转向低壁垒文商科，要额外核算机会成本。"
            if any(key in subjects for key in ["物", "化", "生"])
            else "选科不是强技术底座时，更要靠学校平台、城市实习、证书、作品或考编路径补强可替代性问题。"
        )
        grad_line = (
            "这个方向要把读研或继续深造当成默认成本的一部分，本科毕业直接兑现不能按最好情况估。"
            if needs_grad
            else "这个方向可以先按本科就业核验，但仍要看普通毕业生去向，不要只看优秀样本。"
        )
        city_line = (
            f"目标地区是{city_pref}；十年后能否扛住压力，和城市产业密度、实习机会、校友网络强相关，不能把一线城市样本直接套到所有城市。"
        )
        risk_pref_line = (
            f"当前风险偏好是{risk_pref}；偏稳就要优先保留稳定出口，偏激进才适合把高成长、长培养周期的路径放到前面。"
        )

        risk_flags: list[str] = []
        if irreplaceability is not None and irreplaceability < 65:
            risk_flags.append("技术壁垒偏弱，10年后更怕被平台、AI工具或更便宜的人替代")
        if needs_grad:
            risk_flags.append("本科直接就业的确定性不够，读研预期要提前算进成本")
        if any("市场化岗位少" in item or "就业确定性较弱" in item for item in risks):
            risk_flags.append("市场化岗位少，不能只靠兴趣硬扛，要提前设计考编、考研或体制内路径")
        if not risk_flags:
            risk_flags.append("主要看学校平台、城市资源和个人持续积累，不能只用专业名判断")

        pressure_heavy = ("普通" in family and irreplaceability is not None and irreplaceability < 65) or (needs_grad and "普通" in family)
        conclusion = (
            "压力测试不算轻松，普通家庭必须先把保底、深造和稳定出口设计好。"
            if pressure_heavy
            else "能做备选，但不能无脑当第一主线。"
        )
        salary_line = (
            f"本地估算的10年后收入参考大约在{salary_10y // 1000}K上下，只能看方向，不能当官方工资。"
            if salary_10y
            else "当前缺少可用薪资估算，不展示10年后收入数字，先看路径风险。"
        )

        answer = "\n".join([
            "[分析过程]",
            f"1. 测试对象：按当前画像和本轮问题，锁定为{target_major}，不改成其他专业。",
            f"2. 家庭约束：{family}；普通家庭重点看中位数出口、深造成本和稳定路径。",
            f"3. 长期变量：{insight.trend_analysis or '暂无趋势数据'}。",
            "",
            "[核心判断]",
            f"{target_major}的10年后压力测试结论：{conclusion}",
            salary_line,
            f"当前本地估算中位数参考：{current_salary_line}；就业稳定性参考：{employment_line}；不可替代性：{barrier_line}。",
            "",
            "[画像补充分析]",
            f"1. 考生画像：{self._profile_brief(request.context)}。",
            f"2. 专业属性：{target_major}属于{category}，本地标签为{tags}。",
            f"3. 家庭视角：{family_line}",
            f"4. 选科/能力视角：当前选科{subjects}；{subject_line}",
            f"5. 深造视角：{grad_line}",
            f"6. 城市视角：{city_line}",
            f"7. 风险偏好：{risk_pref_line}",
            "",
            "[红旗风险]",
            *[f"- {item}" for item in risk_flags],
            "",
            "[应对动作]",
            "1. 保底路径：确认本科毕业能走的稳定岗位、考编/考公/国企入口，别把读研当唯一退路。",
            "2. 提升路径：把课程、证书、项目、竞赛、实习或作品集做成可展示证据，否则十年后只剩学历标签。",
            "3. 核验路径：用目标学校就业质量报告和企业招聘样本交叉看，不用单个高薪故事替代中位数判断。",
            "",
            "[核验清单]",
            "1. 查目标学校该专业就业质量报告，别只看学校总就业率。",
            "2. 查该专业普通毕业生去向：教师编、考研、公务员、企业岗位分别占多少。",
            "3. 查10年后仍能站住的能力：证书、编制、作品、项目、学历或城市资源到底是哪一个。",
            "",
            "数据口径：这轮是本地专业库和当前画像的压力测试，不是官方统计；薪资、就业稳定性和技术壁垒都只能做方向判断。",
        ])
        if citations:
            answer += "\n\n数据来源：\n" + "\n".join(f"{idx + 1}. {url}" for idx, url in enumerate(citations[:5]))

        return ConsultResponse(
            answer=answer,
            thinking_process=[
                ThinkingStep(step="画像读取", analysis=self._build_profile_context(request) or "本轮未提供完整画像"),
                ThinkingStep(step="压力测试", analysis=f"已按当前画像专业{target_major}生成10年后压力测试"),
            ],
            follow_up_questions=[],
            confidence="medium",
            citations=citations,
            recommendation_plans=[],
        )

    def _answer_declines_recommendations(self, answer: str) -> bool:
        compact = re.sub(r"\s+", "", answer or "")
        if not compact:
            return False
        decline_patterns = [
            r"(?:筛|粗筛|结果).*?0个(?:志愿|候选|方案)",
            r"(?:候选|方案|志愿)(?:为零|是0|为0|为空)",
            r"冲、稳、保三个档位一个都筛不出来",
            r"没法给你一个.*?学校名单",
            r"不建议直接给.*?学校名单",
            r"没有可同步的院校推荐",
        ]
        return any(re.search(pattern, compact) for pattern in decline_patterns)

    def _build_direct_recommend_response(self, request: ConsultRequest) -> ConsultResponse | None:
        user = self._build_user_preferences(request)
        if not user:
            return None

        recommend = agent_engine.recommend(RecommendRequest(user=user, limit=10))
        lines = [
            f"我跟你说，按你这个画像（{user.province}，{user.score}分，位次{user.rank or '未填'}，选科{user.subjects or '未填'}，{user.family_background or '普通家庭'}），先别乱冲热门。",
            "",
            recommend.summary,
            "",
            "冲稳保建议：",
        ]
        for plan in recommend.plans:
            salary = f"{plan.median_salary_5yr // 1000}K" if plan.median_salary_5yr else "暂无"
            lines.append(
                f"{plan.order}. [{plan.risk_level}] {plan.school} - {plan.major}："
                f"规则模拟概率{plan.probability}%，5年估算中位数薪资{salary}，理由：{plan.reason}"
            )

        if recommend.red_flags:
            lines.append("")
            lines.append("红旗提醒：" + "；".join(recommend.red_flags))

        lines.append("")
        lines.append("重点提醒：这是后端Agent按画像即时推荐；录取概率是规则模拟，薪资是本地估算，最终还要用当年招生计划、专业选科要求和投档位次再核验。")

        return ConsultResponse(
            answer="\n".join(lines),
            thinking_process=[
                ThinkingStep(step="画像识别", analysis=f"已读取省份、分数、位次、选科、家庭、城市和专业方向：{user.model_dump()}"),
                ThinkingStep(step="Agent推荐", analysis="已调用后端推荐Agent生成冲稳保方案"),
            ],
            follow_up_questions=[
                "要不要我按省内学校单独列一版？",
                "要不要继续核验这些学校近年投档位次？",
            ],
            confidence="medium",
            citations=[],
        )

    def _build_direct_insight_response(
        self,
        request: ConsultRequest,
        intent: IntentResult,
        research_results: list[ResearchResult],
    ) -> ConsultResponse | None:
        target_name = (intent.major_names or intent.school_names or [""])[0]
        if not target_name:
            return None

        target_type = "major" if target_name in intent.major_names else "school"
        user = self._build_user_preferences(request, allow_partial=True)
        insight = agent_engine.insights(
            InsightRequest(target_type=target_type, target_name=target_name, user=user)
        )
        if insight.overview == "未找到相关数据":
            return None

        if target_type == "major":
            answer = self._format_major_fit_answer(request, target_name, insight)
            follow_up = [
                f"要不要我按{target_name}方向重新列一版冲稳保学校？",
                f"要不要把{target_name}和你原来的专业方向做就业对比？",
            ]
        else:
            answer = self._format_school_fit_answer(request, target_name, insight)
            follow_up = [
                f"要不要继续核验{target_name}近年投档位次？",
                f"要不要把{target_name}放进当前冲稳保方案里比较？",
            ]

        if research_results:
            answer += "\n\n数据核验：已尝试联网检索，下面展示可用来源；具体招生计划仍以考试院和学校招生网为准。"
        else:
            answer += "\n\n数据核验：当前未拿到稳定联网来源，以上先按本地专业/院校库和你的考生画像判断；涉及当年投档线时必须再查考试院。"

        return ConsultResponse(
            answer=answer,
            thinking_process=[
                ThinkingStep(step="画像读取", analysis=self._build_profile_context(request) or "本轮未提供完整画像"),
                ThinkingStep(step="Agent洞察", analysis=f"已识别咨询对象：{target_type}={target_name}，并生成适配度判断"),
            ],
            follow_up_questions=follow_up,
            confidence="medium",
            citations=[item.url for item in research_results],
        )

    def _format_major_fit_answer(self, request: ConsultRequest, target_name: str, insight) -> str:
        ctx = request.context
        user = self._build_user_preferences(request, allow_partial=True)
        major = self.major_by_name.get(target_name, {})
        family = user.family_background or "普通家庭"
        subjects = user.subjects or "未填"
        current_pref = "、".join(user.major_preference or []) or "未填"
        salary = self._format_salary(insight.median_salary)
        emp = self._format_rate(insight.employment_rate)
        needs_grad = "是" if major.get("requires_grad_school") else "否"
        risks = "；".join(insight.risk_factors) if insight.risk_factors else "暂无明确风险"
        show_precise_metrics = self._is_fact_data_question(request.question)
        salary_text = salary if show_precise_metrics else "待就业质量报告核验"
        emp_text = emp if show_precise_metrics else "待就业质量报告核验"
        irreplaceability_text = f"{insight.irreplaceability or '暂无'}/100" if show_precise_metrics else "待培养方案和岗位样本核验"

        verdict = "可以考虑，但不要无脑转主线。"
        if major.get("requires_grad_school") and "普通" in family:
            verdict = "能学，但不建议作为普通家庭的第一主线，除非你能接受读研、考证和城市资源竞争。"
        elif insight.irreplaceability and insight.irreplaceability >= 80:
            verdict = "可以作为主线看，核心是选到壁垒强、就业路径清楚的学校和专业组。"
        elif insight.irreplaceability and insight.irreplaceability < 65:
            verdict = "可以当备选，但不要只冲名头，必须看城市、平台和后续深造。"

        conflict_notes = []
        if subjects and any(key in subjects for key in ["物", "化", "生"]) and major.get("category") in ["经济学", "管理学", "文学", "法学", "历史学"]:
            conflict_notes.append("你是偏理科选科，转这个方向会放弃一部分工科/技术壁垒优势。")
        if current_pref and target_name not in current_pref:
            conflict_notes.append(f"你当前画像方向是「{current_pref}」，本轮问的是「{target_name}」，后续推荐需要按新方向重算。")
        if "看背景" in major.get("tags", []) or "高度依赖人脉" in insight.risk_factors:
            conflict_notes.append("这个方向比较吃城市、平台、实习和家庭资源，普通路径分化会很明显。")
        if not conflict_notes:
            conflict_notes.append("与当前画像没有硬冲突，但仍要看学校层次和城市资源。")
        advice = self._major_fit_advice(target_name, major)

        return "\n".join([
            f"我跟你说，{target_name}不是不能学，关键看你拿它当主线还是备选。",
            "",
            f"直接判断：{verdict}",
            "",
            f"结合你的画像：{self._profile_brief(ctx)}",
            "",
            "就业倒推看本地估算数据：",
            f"- 5年后收入参考：{salary_text}",
            f"- 就业稳定性参考：{emp_text}",
            f"- 技术壁垒/被替代风险：{irreplaceability_text}",
            f"- 是否明显依赖深造：{needs_grad}",
            f"- 主要风险：{risks}",
            "",
            "和你当前情况的冲突点：",
            *[f"- {note}" for note in conflict_notes],
            "",
            f"我的建议：{advice}",
        ])

    def _major_fit_advice(self, target_name: str, major: dict) -> str:
        category = major.get("category", "")
        if "历史" in target_name or category == "历史学":
            return "如果坚持历史学，优先看师范培养、保研考研、文博档案、地方教育资源和考编口径；别只拿学校牌子赌就业，普通家庭要把深造和稳定出口提前算清楚。"
        if "汉语言" in target_name or "中文" in target_name:
            return "如果坚持中文方向，优先看师范属性、写作训练、考编岗位、媒体出版和新媒体实习资源；别把中文理解成只靠背书，输出能力和城市机会很关键。"
        if category == "法学" or "法学" in target_name:
            return "如果坚持法学，优先看法考通过、实习半径、法院律所资源和考公路径；别只看校名，普通家庭更要问毕业后靠什么拿到第一份稳定机会。"
        if category == "经济学" or "金融" in target_name or "经济" in target_name:
            return "如果坚持这个方向，优先选城市资源强、财经/综合平台强、实习机会多的学校；别为了一个金融名头，去一个平台弱、城市弱、实习少的学校。"
        if category == "工学" or any(key in target_name for key in ["计算机", "软件", "电子", "电气", "通信", "自动化"]):
            return "如果坚持这个方向，重点看课程硬度、实验项目、实习企业和行业标签；别只看专业名字热不热，要看学校能不能把工程能力喂出来。"
        return "如果坚持这个方向，优先看学校资源、城市岗位、培养方案和毕业去向；普通家庭更要看中位数路径，不要只看头部样本。"

    def _format_school_fit_answer(self, request: ConsultRequest, target_name: str, insight) -> str:
        ctx = request.context
        salary = self._format_salary(insight.median_salary)
        emp = self._format_rate(insight.employment_rate)
        risks = "；".join(insight.risk_factors) if insight.risk_factors else "暂无明确风险"
        return "\n".join([
            f"我跟你说，{target_name}能不能选，不能只看名字，要看它放在你这个分数和位次里是冲、稳还是保。",
            "",
            f"结合你的画像：{self._profile_brief(ctx)}",
            "",
            f"学校概览：{insight.overview}",
            f"估算就业率：{emp}",
            f"估算平均/中位薪资参考：{salary}",
            f"风险点：{risks}",
            "",
            "下一步应该做的是：拿这所学校近三年在你省的投档位次、专业组选科要求、可调剂专业一起核验。只看校名，志愿容易填歪。",
        ])

    def _profile_brief(self, ctx: UserProfile | None) -> str:
        if not ctx:
            return "画像未完整提供"
        parts = []
        if ctx.province:
            parts.append(str(ctx.province))
        if ctx.score:
            parts.append(f"{ctx.score}分")
        if ctx.rank:
            parts.append(f"位次{ctx.rank}")
        if ctx.subjects:
            parts.append(f"选科{ctx.subjects}")
        if ctx.family_background:
            parts.append(str(ctx.family_background))
        if ctx.city_preference:
            parts.append("目标地区" + "、".join(ctx.city_preference))
        if ctx.major_preference:
            parts.append("当前方向" + "、".join(ctx.major_preference))
        return "，".join(parts) if parts else "画像未完整提供"

    def _format_salary(self, value: int | None) -> str:
        if not value:
            return "暂无"
        return f"{value // 1000}K" if value >= 1000 else str(value)

    def _format_rate(self, value: float | None) -> str:
        if value is None:
            return "暂无"
        return f"{round(value * 100)}%"

    def _build_profile_context(self, request: ConsultRequest) -> str:
        ctx = request.context
        if not ctx:
            return ""
        fields = []
        if ctx.province:
            fields.append(f"考生省份：{ctx.province}")
        if ctx.score:
            fields.append(f"分数：{ctx.score}")
        if ctx.rank:
            fields.append(f"位次：{ctx.rank}")
        if ctx.subjects:
            fields.append(f"选科：{ctx.subjects}")
        if ctx.family_background:
            fields.append(f"家庭条件：{ctx.family_background}")
        if ctx.city_preference:
            fields.append(f"意向城市：{'、'.join(ctx.city_preference)}")
        if ctx.major_preference:
            fields.append(f"兴趣方向：{'、'.join(ctx.major_preference)}")
        if ctx.risk_appetite:
            fields.append(f"风险偏好：{ctx.risk_appetite}")
        if not fields:
            return ""
        return "当前考生画像（必须优先参考）：\n" + "\n".join(fields)

    def _enrich_request_context(self, request: ConsultRequest) -> ConsultRequest:
        question = request.question
        ctx = request.context.model_copy(deep=True) if request.context else UserProfile()

        if not ctx.province:
            province = self._extract_province(question)
            if province:
                ctx.province = province
        if not ctx.score:
            score = self._extract_score(question)
            if score:
                ctx.score = score
        if not ctx.rank:
            rank = self._extract_rank(question)
            if rank:
                ctx.rank = rank
        explicit_school_regions = self._extract_school_region_preference(question, ctx.province)
        explicit_cities = self._extract_city_preference(question)
        if explicit_school_regions:
            # 本轮明确说“山东院校/省内学校/只看本省”时，必须覆盖历史画像里的旧城市偏好。
            # 省份字段表示考生生源地，不能自动等同于院校所在地；但用户显式要求院校地区时要以本轮为准。
            ctx.city_preference = explicit_school_regions
        elif explicit_cities:
            # 本轮明确提到“青岛济南/南京苏州”等城市时，覆盖历史画像里的旧城市偏好。
            ctx.city_preference = explicit_cities
        elif self._asks_out_of_province(question):
            # “外省/省外”是新的地区范围意图，不能继续沿用上一轮的上海、山东等旧偏好。
            ctx.city_preference = None
        elif not ctx.city_preference:
            if explicit_cities:
                ctx.city_preference = explicit_cities
        question_major_pref = self._extract_major_preference(question)
        if question_major_pref and (
            self._is_explicit_current_major_recommendation(question, question_major_pref)
            or self._asks_about_major_switch(question, question_major_pref, ctx.major_preference)
        ):
            # 追问里出现新的专业方向时，以本轮问题为准，避免沿用第一次画像里的旧方向。
            ctx.major_preference = question_major_pref

        return ConsultRequest(question=request.question, context=ctx)

    def _extract_school_region_preference(self, text: str, profile_province: str | None = None) -> list[str]:
        regions: list[str] = []
        for province in PROVINCES:
            if re.search(rf"{province}(?:省|市|自治区)?(?:内|本地)?.{{0,16}}(?:院校|高校|学校|大学)", text):
                regions.append(province)
            elif re.search(rf"(?:只看|仅看|优先看|限定|限)(?:在)?{province}(?:省|市|自治区)?", text):
                regions.append(province)
            elif re.search(rf"在{province}(?:省|市|自治区)?(?:有|能|可以).{{0,8}}(?:推荐|报|上)", text):
                regions.append(province)

        if re.search(r"(?:省内|本省|本地)(?:院校|高校|学校|大学|推荐|能报|有哪些)", text):
            province = profile_province or self._extract_province(text)
            if province:
                regions.append(self._normalize_region_name(province))

        normalized = []
        for region in regions:
            region = self._normalize_region_name(region)
            if region and region not in normalized:
                normalized.append(region)
        return normalized

    def _detect_intent(self, request: ConsultRequest) -> IntentResult:
        question = request.question
        school_names = self._extract_school_names(question)
        major_names = []
        for name in self.major_names:
            if name in question and name not in major_names:
                major_names.append(name)
        for name in self._extract_major_preference(question):
            if name not in major_names:
                major_names.append(name)
        ctx = request.context
        profile_ready = bool(ctx and ctx.province and ctx.score)
        fact_data_question = self._is_fact_data_question(question)
        pressure_test_question = self._is_pressure_test_question(question)
        declines_recommendation = self._declines_school_recommendation(question)
        should_backfill_major = any(marker in question for marker in [
            "中位数", "薪资", "工资", "收入", "就业", "出路", "岗位",
            "这个专业", "该专业", "本专业", "这个方向", "当前方向", "我的方向",
            "我的专业", "壁垒", "不可替代", "压力测试", "10年后", "十年后",
        ])
        profile_major_reference = any(marker in question for marker in [
            "这个专业", "该专业", "本专业", "这个方向", "当前方向", "我的方向",
            "我的专业", "壁垒", "不可替代", "压力测试", "10年后", "十年后",
        ])
        if (fact_data_question or pressure_test_question or declines_recommendation or profile_major_reference) and should_backfill_major and not major_names and ctx and ctx.major_preference:
            for name in self._expand_major_preferences(ctx.major_preference):
                if name not in major_names:
                    major_names.append(name)
        profile_recommend_signal = profile_ready and any(
            keyword in question
            for keyword in [
                "院校推荐", "学校推荐", "推荐院校", "推荐学校", "志愿", "冲稳保", "填报",
                "能报", "能上", "怎么报", "怎么选", "该选", "报什么", "选什么",
            ]
        )

        if self._is_single_school_chance_question(question, school_names):
            intent = "school_chance"
        elif pressure_test_question:
            intent = "pressure_test"
        elif fact_data_question:
            intent = "insight"
        elif declines_recommendation:
            intent = "insight" if (school_names or major_names or any(keyword in question for keyword in INSIGHT_KEYWORDS + FACT_KEYWORDS)) else "chat"
        elif any(keyword in question for keyword in RECOMMEND_KEYWORDS) or profile_recommend_signal:
            intent = "recommend"
        elif school_names or major_names or any(keyword in question for keyword in INSIGHT_KEYWORDS):
            intent = "insight"
        else:
            intent = "chat"

        needs_research = bool(
            intent in ["recommend", "insight", "school_chance", "pressure_test"]
            or
            school_names
            or major_names
            or any(keyword in question for keyword in FACT_KEYWORDS)
        )
        return IntentResult(intent=intent, school_names=school_names, major_names=major_names, needs_research=needs_research)

    def _is_pressure_test_question(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return False
        return any(marker in compact for marker in ["10年后压力测试", "十年后压力测试", "压力测试", "10年后", "十年后"])

    def _declines_school_recommendation(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return False
        decline_patterns = [
            r"不要(?:给我)?(?:推荐|列|排).{0,8}(?:学校|院校|大学|志愿)",
            r"别(?:给我)?(?:推荐|列|排).{0,8}(?:学校|院校|大学|志愿)",
            r"不(?:要|用|想)(?:推荐|列|看).{0,8}(?:学校|院校|大学|志愿)",
            r"只(?:说|聊|看|分析).{0,8}(?:就业|出路|薪资|工资|收入|前景|专业)",
            r"(?:先|暂时)?不(?:做|要|看).{0,8}(?:冲稳保|院校推荐|学校推荐|志愿方案)",
        ]
        return any(re.search(pattern, compact) for pattern in decline_patterns)

    def _fact_target_major(self, request: ConsultRequest, intent: IntentResult | None = None) -> str:
        candidates = list(intent.major_names if intent else [])
        if not candidates:
            candidates.extend(self._extract_major_preference(request.question))
        if not candidates and request.context and request.context.major_preference:
            candidates.extend(self._expand_major_preferences(request.context.major_preference))
        if candidates:
            return candidates[0]
        if request.context and request.context.major_preference:
            return str(request.context.major_preference[0]).strip()
        return ""

    def _is_fact_data_question(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if not compact:
            return False

        fact_markers = [
            "中位数", "薪资", "工资", "收入", "就业率", "就业数据", "就业质量",
            "500强", "五百强", "世界五百强", "校招", "招聘", "招聘名单", "企业名单",
        ]
        if not any(marker in compact for marker in fact_markers):
            return False

        explicit_recommend_markers = [
            "推荐", "报考", "志愿", "冲稳保", "院校推荐", "学校推荐", "推荐院校", "推荐学校",
            "能报", "能上", "该报", "报什么", "报哪些", "怎么报", "选什么学校", "选哪些学校",
        ]
        if any(marker in compact for marker in explicit_recommend_markers):
            return False

        fact_question_markers = [
            "多少", "是多少", "有多少", "哪些", "哪几", "哪里", "去哪些", "怎么查",
            "有没有", "能进", "进不进", "容易进", "数据", "名单", "招聘", "校招", "中位数", "薪资", "工资", "收入",
        ]
        return any(marker in compact for marker in fact_question_markers)

    def _guard_fact_data_response(
        self,
        response: ConsultResponse,
        request: ConsultRequest,
        intent: IntentResult,
        citations: list[str],
    ) -> ConsultResponse:
        """Keep salary/500强 questions in the data-answer lane after LLM post-processing."""
        response.recommendation_plans = []
        response.answer = self._downgrade_unsupported_source_claims(response.answer, citations)
        answer = response.answer or ""
        if self._is_fact_answer_off_topic(answer, request) or self._answer_conflicts_with_major_scope(answer, request, intent):
            response.answer = self._build_fact_data_fallback_answer(request, intent, citations)
            response.follow_up_questions = []
            response.confidence = "medium"
        else:
            response.answer = self._append_fact_data_contextual_analysis(response.answer, request, intent)
            response.follow_up_questions = []
        return response

    def _downgrade_unsupported_source_claims(self, answer: str, citations: list[str] | None = None) -> str:
        text = str(answer or "")
        if citations:
            return text
        replacements = {
            "已经官方核验": "按本地库估算",
            "已官方核验": "按本地库估算",
            "官方真实": "本地估算",
            "真实中位数": "收入参考",
            "真实就业率": "就业稳定性参考",
            "真实数据": "本地估算数据",
            "已经核验": "待官方来源核验",
            "已核验": "待官方来源核验",
            "联网核验": "公开来源待核验",
            "官方核验": "官方来源待核验",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _is_fact_answer_off_topic(self, answer: str, request: ConsultRequest) -> bool:
        compact = re.sub(r"\s+", "", answer or "")
        if not compact:
            return True
        forbidden_markers = [
            "[灵魂追问]", "【灵魂追问】", "[院校推荐]", "【院校推荐】",
            "冲稳保", "冲刺", "保底", "志愿表",
            "哪个学校名字好听", "哪所学校名字好听", "先别问哪个学校",
        ]
        if any(marker in compact for marker in forbidden_markers):
            return True
        question = request.question or ""
        salary_like = any(marker in question for marker in ["中位数", "薪资", "工资", "收入"])
        if salary_like and not any(marker in compact for marker in ["中位数", "薪资", "工资", "收入", "K", "元", "暂无"]):
            return True
        return False

    def _build_fact_data_fallback_answer(
        self,
        request: ConsultRequest,
        intent: IntentResult,
        citations: list[str],
    ) -> str:
        question = request.question or ""
        target_major = self._fact_target_major(request, intent)
        salary_like = any(marker in question for marker in ["中位数", "薪资", "工资", "收入"])
        fortune500_like = any(marker in question for marker in ["500强", "五百强", "世界五百强"])

        if salary_like:
            if not target_major:
                return (
                    "[核心判断]\n"
                    "你问的是“这个专业”的中位数收入，但当前画像里没有识别到明确专业名，所以我不能硬编一个薪资数。\n\n"
                    "[核验清单]\n"
                    "把专业名补上后，我再按本地估算、就业质量报告口径和行业去向给你拆。"
                )
            user = self._build_user_preferences(request, allow_partial=True)
            insight = agent_engine.insights(
                request=InsightRequest(target_type="major", target_name=target_major, user=user)
            )
            if insight.median_salary:
                salary_text = self._format_salary(insight.median_salary)
            else:
                salary_text = "暂无可用估算"
            risk_text = "；".join(insight.risk_factors) if insight.risk_factors else "暂无明确风险标签"
            answer = (
                "[核心判断]\n"
                f"{target_major}普通毕业生工作几年后的收入参考：{salary_text}。这个数是本地估算，不是官方就业质量报告里的精确统计。\n\n"
                "[分析过程]\n"
                f"1. 专业对象：按当前画像和本轮问题，识别为{target_major}。\n"
                f"2. 趋势判断：{insight.trend_analysis or '暂无趋势数据'}。\n"
                f"3. 风险提醒：{risk_text}。\n\n"
                f"{self._build_salary_profile_analysis(request, target_major, insight)}\n\n"
                "[核验清单]\n"
                "1. 查目标学校就业质量报告，看该专业毕业去向和行业分布。\n"
                "2. 查招聘平台同岗位真实薪酬区间，分城市看，不要只看头部样本。\n"
                "3. 如果后续要选学校，再单独回到投档位次和专业组核验。"
            )
        elif fortune500_like:
            user = self._build_user_preferences(request, allow_partial=True)
            insight = None
            if target_major:
                insight = agent_engine.insights(
                    request=InsightRequest(target_type="major", target_name=target_major, user=user)
                )
            major_prefix = f"如果按{target_major}看，" if target_major else ""
            answer = (
                "[核心判断]\n"
                f"{major_prefix}500强校招不是一张全国统一固定名单，它通常看学校层次、专业对口度、城市产业圈和企业当年岗位需求。你这轮问的是招聘去向，不是让系统重排志愿。\n\n"
                "[分析过程]\n"
                f"1. 专业对象：{target_major or '本轮未识别到明确专业'}。\n"
                "2. 985、强211、行业特色院校和重点城市高校更容易进入500强校招池。\n"
                "3. 不同企业差异很大：电力、制造、金融、互联网、央国企看的学校和专业并不一样。\n\n"
                f"{self._build_fortune500_profile_analysis(request, target_major, insight)}\n\n"
                "[核验清单]\n"
                "1. 查企业校园招聘官网的目标院校和宣讲行程。\n"
                "2. 查学校就业质量报告里的重点单位、世界500强/中国500强去向。\n"
                "3. 查学院层面的就业去向，别只看学校总表。"
            )
        else:
            answer = (
                "[核心判断]\n"
                "这轮是数据咨询，不是院校推荐。我会按数据口径回答，不展开冲稳保学校名单。\n\n"
                "[核验清单]\n"
                "优先查官方就业质量报告、企业招聘官网和公开统计口径。"
            )

        if citations:
            answer += "\n\n数据来源：\n" + "\n".join(f"{idx + 1}. {url}" for idx, url in enumerate(citations[:5]))
        return answer

    def _append_fact_data_contextual_analysis(
        self,
        answer: str,
        request: ConsultRequest,
        intent: IntentResult,
    ) -> str:
        question = request.question or ""
        target_major = self._fact_target_major(request, intent)
        salary_like = any(marker in question for marker in ["中位数", "薪资", "工资", "收入"])
        fortune500_like = any(marker in question for marker in ["500强", "五百强", "世界五百强"])
        if not target_major and not fortune500_like:
            return answer

        user = self._build_user_preferences(request, allow_partial=True)
        insight = None
        if target_major:
            insight = agent_engine.insights(
                request=InsightRequest(target_type="major", target_name=target_major, user=user)
            )

        supplement = ""
        if salary_like and insight:
            supplement = self._build_salary_profile_analysis(request, target_major, insight)
        elif fortune500_like:
            supplement = self._build_fortune500_profile_analysis(request, target_major, insight)

        if not supplement or supplement in (answer or ""):
            return answer
        return f"{(answer or '').rstrip()}\n\n{supplement}"

    def _build_salary_profile_analysis(self, request: ConsultRequest, target_major: str, insight) -> str:
        user = self._build_user_preferences(request, allow_partial=True)
        major = self.major_by_name.get(target_major, {})
        family = user.family_background or "普通家庭"
        subjects = user.subjects or "未填"
        city_pref = "、".join(user.city_preference or []) or "未限定"
        risk_pref = user.risk_appetite or "均衡"
        needs_grad = bool(major.get("requires_grad_school"))
        barrier = insight.irreplaceability
        employment = self._format_rate(insight.employment_rate)
        tags = "、".join(major.get("tags", [])[:4]) or "暂无标签"

        family_line = (
            "普通家庭要把这个数当作“下限和稳定性检查”，不能只看头部高薪个例。"
            if "普通" in family
            else "家庭试错空间相对更大，但也要看投入周期和回报确定性。"
        )
        subject_line = (
            "当前选科偏理，若转向文史经管类，要额外评估技术壁垒损失。"
            if any(key in subjects for key in ["物", "化", "生"]) and major.get("category") in ["文学", "历史学", "法学", "经济学", "管理学"]
            else "当前选科与专业方向没有明显硬冲突，重点看学校培养资源和毕业去向。"
        )
        grad_line = (
            "这个方向明显要把读研、考证或考编放进成本表，本科毕业直接变现不能想得太满。"
            if needs_grad
            else "这个方向可以先看本科就业出口，但仍要核验普通毕业生去向。"
        )
        barrier_line = (
            "壁垒偏高，关键是把课程、项目和实习做实，否则高壁垒也落不到个人身上。"
            if barrier and barrier >= 80
            else "壁垒一般或偏弱，更要靠城市、学校平台、证书/作品/项目补足竞争力。"
        )

        return "\n".join([
            "[画像补充分析]",
            f"1. 家庭视角：{family_line}",
            f"2. 选科视角：{subject_line}",
            f"3. 深造视角：{grad_line}",
            f"4. 壁垒视角：{barrier_line}",
            f"5. 城市视角：目标地区为{city_pref}；薪资中位数必须分城市看，一线和非一线不能混成一个数。",
            f"6. 风险偏好：当前偏好是{risk_pref}，如果偏稳，就优先看稳定岗位和可验证去向；如果偏冲，再看高成长岗位。",
            f"本地参考：就业稳定性{employment}，专业标签{tags}。这些只用于方向判断，不是官方统计。",
        ])

    def _build_fortune500_profile_analysis(self, request: ConsultRequest, target_major: str, insight) -> str:
        user = self._build_user_preferences(request, allow_partial=True)
        major = self.major_by_name.get(target_major, {}) if target_major else {}
        family = user.family_background or "普通家庭"
        subjects = user.subjects or "未填"
        city_pref = "、".join(user.city_preference or []) or "未限定"
        target = target_major or "当前方向"
        category = major.get("category", "")
        barrier = insight.irreplaceability if insight else None

        if any(key in target for key in ["计算机", "软件", "电子", "电气", "通信", "自动化"]):
            position_line = "更适合看研发、测试、运维、硬件、供应链数字化、央国企技术岗等入口。"
        elif any(key in target for key in ["历史", "汉语言", "新闻", "英语"]):
            position_line = "不要只盯总部管培，重点看品牌、内容、行政、人力、公共事务、教育培训和央国企综合岗。"
        elif any(key in target for key in ["法学", "经济", "金融", "会计"]):
            position_line = "重点看法务、合规、审计、财务、风控、运营和管培，但学校平台与实习经历权重很高。"
        else:
            position_line = "先拆岗位类型，再看企业是否真的招这个专业，不要把“500强”当成一个笼统标签。"

        family_line = (
            "普通家庭别把“能进500强”理解成必然高薪稳定，先看岗位地点、培养周期、淘汰率和转正概率。"
            if "普通" in family
            else "家庭支持更强时，可以适当接受更长培养周期，但仍要核验岗位质量。"
        )
        barrier_line = (
            "专业壁垒较强时，要用项目、竞赛、实习和证书把对口能力证明出来。"
            if barrier and barrier >= 75
            else "专业壁垒不够强时，500强筛人会更看学校层次、城市实习和可迁移能力。"
        )

        return "\n".join([
            "[画像补充分析]",
            f"1. 岗位匹配：{position_line}",
            f"2. 家庭视角：{family_line}",
            f"3. 选科/能力：当前选科{subjects}，要把课程基础转成企业能看懂的项目、证书或实习。",
            f"4. 城市视角：目标地区为{city_pref}；500强校招高度受城市产业圈影响，上海、北京、深圳、广州、杭州、南京等机会密度不同。",
            f"5. 壁垒视角：{barrier_line}",
            f"6. 验证动作：查企业校招官网、目标学校就业质量报告、学院去向表，再看近两年是否有{target}对口岗位。",
            f"结论边界：这只能判断进入校招池的可能性和准备方向，不能承诺固定企业名单或录用结果。",
        ])

    def _extract_school_names(self, text: str) -> list[str]:
        compact = re.sub(r"\s+", "", text or "")
        matches: list[str] = []
        for name in sorted(self.school_names, key=len, reverse=True):
            normalized_name = re.sub(r"[()（）]", "", name)
            if name in text or normalized_name in compact:
                matches.append(name)
        for alias, target in sorted(SCHOOL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if alias in compact and target in self.school_by_name and target not in matches:
                matches.append(target)
        return matches

    def _is_single_school_chance_question(self, text: str, school_names: list[str]) -> bool:
        if len(school_names) != 1:
            return False
        compact = re.sub(r"\s+", "", text or "")
        chance_markers = [
            "有机会吗", "有没有机会", "有无机会", "能上吗", "能不能上", "能进吗", "能不能进",
            "能冲吗", "可以冲吗", "稳吗", "够吗", "够不够", "录取吗", "能录吗", "能报吗",
            "值得冲", "这个分", "这个位次", "我这分", "我这个分",
        ]
        chance_phrases = ["有机会", "能上", "能不能上", "能进", "能冲", "可以冲", "稳不稳", "能报", "够不够", "录取"]
        return any(marker in compact for marker in chance_markers) or any(phrase in compact for phrase in chance_phrases)

    def _research_if_needed(self, request: ConsultRequest, intent: IntentResult) -> list[ResearchResult]:
        if not intent.needs_research:
            return []

        queries = self._build_research_queries(request, intent)
        results = self._local_official_sources(intent)
        results.extend(self._province_official_sources(request))
        results.extend(web_research_client.research(queries, limit_per_query=3, max_results=10, max_queries=8, max_seconds=18))
        seen = set()
        unique_results = []
        for item in results:
            if item.url in seen:
                continue
            seen.add(item.url)
            unique_results.append(item)
        return unique_results[:14]

    def _research_recommendation_plans(self, request: ConsultRequest, recommend: RecommendResponse) -> list[ResearchResult]:
        ctx = request.context
        province = self._normalize_region_name(ctx.province) if ctx and ctx.province else self._extract_province(request.question)
        results: list[ResearchResult] = []
        queries: list[str] = []
        official_site = self._province_exam_site(province)
        if province:
            major_focus = self._recommend_major_focus(ctx.major_preference if ctx else None)
            if official_site:
                queries.append(f"site:{official_site} {province} 2025 本科 普通类 专业最低分 录取分数线")
                queries.append(f"site:{official_site} {province} 2025 投档最低分 专业 录取分数线")
            queries.append(f"site:gaokao.chsi.com.cn {province} 2025 {major_focus} 专业最低分 录取分数线".strip())
        for plan in recommend.plans[:10]:
            school = self.school_by_name.get(plan.school, {})
            admissions_url = self._school_admissions_entry_url(school, plan.school)
            admissions_domain = self._extract_domain(admissions_url)
            if admissions_url:
                results.append(
                    ResearchResult(
                        title=f"{plan.school}本科招生网",
                        url=admissions_url,
                        snippet="本地高校本科招生网地址库匹配，优先用于核验招生计划、专业组、选科要求、录取分数线和调剂规则。",
                    )
                )
                if admissions_domain:
                    queries.append(f"site:{admissions_domain} 2025 {plan.school} {plan.major} 专业最低分 录取分数线")
                    queries.append(f"site:{admissions_domain} 2025 {plan.school} {province or ''} {plan.major} 招生计划 专业组".strip())
            official_url = school.get("official_url")
            if official_url and official_url.rstrip("/") != (admissions_url or "").rstrip("/"):
                official_domain = self._extract_domain(official_url)
                results.append(
                    ResearchResult(
                        title=f"{plan.school}官网",
                        url=official_url,
                        snippet="本地高校官网地址库匹配，用于核验学校基本信息、院系、招生入口和就业信息入口。",
                    )
                )
                if official_domain:
                    queries.append(f"site:{official_domain} 2025 {plan.school} {plan.major} 专业最低分 录取分数线")
                    queries.append(f"site:{official_domain} 2025 {plan.school} 本科招生 {plan.major} 最低录取分")
            queries.extend(
                self._admission_score_queries(
                    province=province,
                    school_name=plan.school,
                    major_name=plan.major,
                    official_site=official_site,
                    admissions_site=admissions_domain,
                )
            )
        results.extend(
            web_research_client.research(
                self._dedupe_queries(queries),
                limit_per_query=2,
                max_results=20,
                max_queries=28,
                max_seconds=35,
            )
        )
        seen = set()
        unique = []
        for item in results:
            key = item.url.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique[:24]

    def _dedupe_queries(self, queries: list[str]) -> list[str]:
        unique: list[str] = []
        seen = set()
        for query in queries:
            normalized = re.sub(r"\s+", " ", str(query or "")).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return unique

    def _admission_score_queries(
        self,
        province: str | None,
        school_name: str | None = None,
        major_name: str | None = None,
        official_site: str | None = None,
        admissions_site: str | None = None,
    ) -> list[str]:
        """Build focused queries for 2025 admission-score verification."""
        province = self._normalize_region_name(province or "")
        school = (school_name or "").strip()
        major = (major_name or "").strip()
        subject = " ".join(part for part in [province, school, major] if part)
        if not subject:
            return []

        queries: list[str] = []
        if admissions_site:
            queries.append(f"site:{admissions_site} 2025 {subject} 专业最低分 录取分数线")
            queries.append(f"site:{admissions_site} 2025 {subject} 招生计划 专业组 调剂")
        if official_site:
            queries.append(f"site:{official_site} 2025 {subject} 专业最低分 录取分数线")
            queries.append(f"site:{official_site} 2025 {subject} 投档最低分")
        queries.append(f"site:gaokao.chsi.com.cn 2025 {subject} 专业最低分 录取分数线")
        if school:
            queries.append(f"{school} 本科招生网 2025 {province} {major} 专业最低分".strip())
        queries.append(f"2025 {subject} 最低录取分 专业录取分数线")
        return self._dedupe_queries(queries)

    def _extract_domain(self, url: str | None) -> str:
        if not url:
            return ""
        match = re.search(r"https?://([^/]+)", url)
        return match.group(1).lower() if match else ""

    def _school_admissions_record(self, school_name: str | None) -> dict:
        if not school_name:
            return {}
        return self.school_admissions_by_name.get(school_name) or {}

    def _school_admissions_entry_url(self, school: dict, school_name: str | None = None) -> str | None:
        """Return the safest local official entry for undergraduate-admission verification."""
        name = school_name or school.get("name")
        record = self._school_admissions_record(name)
        if record.get("admissions_url"):
            return record.get("admissions_url")
        return school.get("official_url") or None

    def _school_admissions_query(self, province: str | None, school_name: str, major_name: str) -> str:
        province = self._normalize_region_name(province or "")
        pieces = [school_name, "本科招生网", "2025"]
        if province:
            pieces.append(province)
        if major_name:
            pieces.append(major_name)
        pieces.extend(["专业最低分", "录取分数线"])
        return " ".join(piece for piece in pieces if piece).strip()

    def _is_live_research_result(self, item: ResearchResult) -> bool:
        snippet = item.snippet or ""
        static_markers = [
            "本地高校官网地址库匹配",
            "用于核验",
        ]
        if any(marker in snippet for marker in static_markers):
            return False
        return bool(item.url and item.title)

    def _live_research_results(self, results: list[ResearchResult]) -> list[ResearchResult]:
        return [item for item in results if self._is_live_research_result(item)]

    def _research_status_text(self, results: list[ResearchResult]) -> str:
        live_count = len(self._live_research_results(results))
        if live_count:
            return f"联网状态：本轮拿到{live_count}条搜索结果，可作为公开来源入口；具体投档位次仍以考试院原表和学校招生网为准。"
        return "联网状态：这轮没有拿到有效联网搜索结果，先按本地库粗筛；投档位次、专业组和调剂风险必须回教育考试院与学校招生网核验。"

    def _province_official_sources(self, request: ConsultRequest) -> list[ResearchResult]:
        ctx = request.context
        province = self._normalize_region_name(ctx.province) if ctx and ctx.province else self._extract_province(request.question)
        if not province:
            return []
        province_sources = {
            "山东": ("山东省教育招生考试院", "https://www.sdzk.cn/", "用于核验山东高考投档表、招生录取政策和官方公告。"),
            "上海": ("上海市教育考试院", "https://www.shmeea.edu.cn/", "用于核验上海高考招生录取、投档与考试院官方公告。"),
            "江苏": ("江苏省教育考试院", "https://www.jseea.cn/", "用于核验江苏高考投档表、招生录取政策和官方公告。"),
            "浙江": ("浙江省教育考试院", "https://www.zjzs.net/", "用于核验浙江高考投档线、招生录取政策和官方公告。"),
            "广东": ("广东省教育考试院", "https://eea.gd.gov.cn/", "用于核验广东高考投档表、招生录取政策和官方公告。"),
        }
        exam_site = self._province_exam_site(province)
        sources = [
            ResearchResult(
                title="阳光高考",
                url="https://gaokao.chsi.com.cn/",
                snippet="教育部阳光高考平台，用于核验高校招生章程、院校信息和招生政策入口。",
            )
        ]
        source = province_sources.get(province)
        if source:
            title, url, snippet = source
            sources.append(ResearchResult(title=title, url=url, snippet=snippet))
        elif exam_site:
            sources.append(
                ResearchResult(
                    title=f"{province}教育考试院",
                    url=f"https://{exam_site}/",
                    snippet=f"用于核验{province}高考投档表、招生录取政策和官方公告。",
                )
            )
        return sources

    def _province_exam_site(self, province: str | None) -> str:
        sites = {
            "北京": "www.bjeea.cn",
            "天津": "www.zhaokao.net",
            "河北": "www.hebeea.edu.cn",
            "山西": "www.sxkszx.cn",
            "内蒙古": "www.nm.zsks.cn",
            "辽宁": "www.lnzsks.com",
            "吉林": "www.jleea.edu.cn",
            "黑龙江": "www.lzk.hl.cn",
            "上海": "www.shmeea.edu.cn",
            "江苏": "www.jseea.cn",
            "浙江": "www.zjzs.net",
            "安徽": "www.ahzsks.cn",
            "福建": "www.eeafj.cn",
            "江西": "www.jxeea.cn",
            "山东": "www.sdzk.cn",
            "河南": "www.haeea.cn",
            "湖北": "www.hbea.edu.cn",
            "湖南": "www.hneeb.cn",
            "广东": "eea.gd.gov.cn",
            "广西": "www.gxeea.cn",
            "海南": "ea.hainan.gov.cn",
            "重庆": "www.cqksy.cn",
            "四川": "www.sceea.cn",
            "贵州": "zsksy.guizhou.gov.cn",
            "云南": "www.ynzs.cn",
            "西藏": "zsks.edu.xizang.gov.cn",
            "陕西": "www.sneea.cn",
            "甘肃": "www.ganseea.cn",
            "青海": "www.qhjyks.com",
            "宁夏": "www.nxjyks.cn",
            "新疆": "www.xjzk.gov.cn",
        }
        return sites.get(self._normalize_region_name(province or ""), "")

    def _local_official_sources(self, intent: IntentResult) -> list[ResearchResult]:
        official_sources = []
        by_name = {item["name"]: item for item in schools}
        for name in intent.school_names:
            school = by_name.get(name)
            admissions_url = self._school_admissions_entry_url(school or {}, name)
            if admissions_url:
                official_sources.append(
                    ResearchResult(
                        title=f"{name}本科招生网",
                        url=admissions_url,
                        snippet="本地高校本科招生网地址库匹配，用于核验招生计划、专业组、选科要求、录取分数线和调剂规则。",
                    )
                )
            official_url = school.get("official_url") if school else None
            if official_url and official_url.rstrip("/") != (admissions_url or "").rstrip("/"):
                official_sources.append(
                    ResearchResult(
                        title=f"{name}官网",
                        url=official_url,
                        snippet="本地高校官网地址库匹配，用于核验学校基本信息、院系与招生入口。",
                    )
                )
        return official_sources

    def _build_research_queries(self, request: ConsultRequest, intent: IntentResult) -> list[str]:
        question = request.question
        ctx = request.context
        province = ctx.province if ctx and ctx.province else self._extract_province(question) or ""
        official_site = self._province_exam_site(province)
        if self._is_fact_data_question(question):
            fact_queries = [question]
            salary_like = any(marker in question for marker in ["中位数", "薪资", "工资", "收入", "就业率", "就业数据", "就业质量"])
            if salary_like:
                major_targets = intent.major_names or self._expand_major_preferences(ctx.major_preference if ctx else None) or []
                for major_name in major_targets[:3]:
                    fact_queries.append(f"{major_name} 中位数薪资 就业质量报告")
                    fact_queries.append(f"{major_name} 毕业生 薪资 中位数")
            if any(marker in question for marker in ["500强", "五百强", "世界五百强"]):
                fact_queries.append("500强 校招 高校 招聘 名单")
                fact_queries.append("世界500强 校园招聘 高校 名单")
            return self._dedupe_queries(fact_queries)

        targets = intent.school_names + intent.major_names
        if not targets:
            targets = self._extract_city_preference(question) or []

        queries: list[str] = []
        if intent.school_names:
            majors = intent.major_names or self._expand_major_preferences(ctx.major_preference if ctx else None) or [""]
            for school_name in intent.school_names[:3]:
                admissions_domain = self._extract_domain(self._school_admissions_entry_url(self.school_by_name.get(school_name, {}), school_name))
                for major_name in majors[:2]:
                    queries.extend(
                        self._admission_score_queries(
                            province=province,
                            school_name=school_name,
                            major_name=major_name,
                            official_site=official_site,
                            admissions_site=admissions_domain,
                        )
                    )
        else:
            for target in targets[:3]:
                queries.extend(
                    self._admission_score_queries(
                        province=province,
                        major_name=target,
                        official_site=official_site,
                    )
                )

        if intent.intent == "school_chance" and intent.school_names:
            major_focus = self._recommend_major_focus(ctx.major_preference if ctx else intent.major_names)
            school = intent.school_names[0]
            admissions_domain = self._extract_domain(self._school_admissions_entry_url(self.school_by_name.get(school, {}), school))
            queries.extend(
                self._admission_score_queries(
                    province=province,
                    school_name=school,
                    major_name=major_focus,
                    official_site=official_site,
                    admissions_site=admissions_domain,
                )
            )

        if intent.intent == "recommend":
            major_focus = self._recommend_major_focus(ctx.major_preference if ctx else intent.major_names)
            if official_site:
                queries.append(f"site:{official_site} {province} 2025 本科 普通类 专业最低分 录取分数线")
                queries.append(f"site:{official_site} {province} 2025 投档最低分 专业 录取分数线")
            queries.append(f"site:gaokao.chsi.com.cn {province} 2025 {major_focus} 专业最低分 录取分数线".strip())

        return self._dedupe_queries(queries)

    def _build_recommend_context(self, request: ConsultRequest) -> str:
        user = self._build_user_preferences(request)
        if not user:
            return (
                "Agent推荐状态：信息不足，无法调用 /api/agent/recommend。"
                "必须追问省份、分数、位次、选科、城市偏好、专业偏好。"
            )

        recommend = agent_engine.recommend(RecommendRequest(user=user, limit=10))
        return self._format_recommend_context(recommend, user)

    def _format_recommend_context(self, recommend: RecommendResponse, user: UserPreferences | None = None) -> str:
        lines = [
            "Agent推荐结果：",
            self._sanitize_recommend_summary_for_chat(recommend.summary),
            "院校推荐主回答模板：先给总判断，再分冲稳保；每档先解释档位作用，再逐校说明“为什么能看、普通家庭防什么坑、下一步查什么”。",
            "输出顺序硬约束：必须先完整输出[分析过程]、[核心判断]、[灵魂追问]，再输出[院校推荐]；禁止在[灵魂追问]没有写完时提前列学校。",
            "主回答禁止项：不要出现具体模拟概率、具体薪资数字、薪资区间、估算中位数、不可替代性分值；这些数值只进入 recommendation_plans。",
            "推荐理由写法：用“学校行业底色 + 专业真实出口 + 城市资源 + 调剂风险 + 官方核验入口”替代数字堆砌。",
            "冲稳保方案：",
        ]
        if user and user.city_preference:
            city_text = "、".join(user.city_preference)
            lines.append(f"地区硬约束：本轮只允许推荐位于「{city_text}」的院校；禁止新增其他省市院校，候选不足时直接说明不足，不要用外地学校补位。")
        if user and user.major_preference:
            major_text = "、".join(user.major_preference)
            lines.append(f"专业硬约束：本轮只允许推荐与「{major_text}」相关的专业和院校专业组合；禁止用计算机、电子信息、法学、医学等其他方向补位，除非用户本轮明确改问该专业。")
        if user and not user.allow_military_schools:
            lines.append("特殊院校硬约束：用户未明确要求军校/部队院校，本轮禁止新增国防、军医、陆军、海军、空军、火箭军、武警、战略支援部队等军校或部队院校。")
        for plan in recommend.plans:
            school = self.school_by_name.get(plan.school, {})
            major = self.major_by_name.get(plan.major, {})
            risk_tags = plan.risk_tags or build_family_risk_profile(school, major, user.family_background if user else None, plan.risk_level)["risk_tags"]
            lines.append(
                f"{plan.order}. [{plan.risk_level}] {plan.school} - {plan.major}，"
                f"风险档位仅用于冲稳保排序，具体模拟概率和薪资只进入同步方案。"
                f"{f'替代路径：{plan.fallback_strategy}。' if plan.fallback_strategy else ''}"
                f"家庭风险标签：{'、'.join(risk_tags) if risk_tags else '暂无明显结构性风险'}。"
                f"家庭分流建议：{plan.family_strategy or build_family_risk_profile(school, major, user.family_background if user else None, plan.risk_level)['family_strategy']}。"
                f"学校差异点：{self._school_distinctive_angle(school, plan.school, plan.major)}。"
                f"专业路径：{self._school_major_path_sentence(school, plan.school, major, plan.major)}"
                f"普通家庭核验点：{self._family_warning_sentence(school, major, plan.school, plan.major)}"
                f"逐校防重复核验点：{self._unique_school_checkpoint(plan.school, plan.major)}"
            )
        if recommend.red_flags:
            lines.append("红旗提醒：" + "；".join(recommend.red_flags))
        lines.append("表达要求：回答时不要直接堆中位数、不可替代性这类指标名，要翻译成普通家庭听得懂的话：能不能进、毕业后走什么路、被替代风险高不高、下一步查哪张官方投档表。")
        lines.append("风险标签要求：逐校推荐必须点名最关键的家庭风险标签，例如调剂风险高、专业组混杂、城市就业资源弱、专业出口窄、需要读研、家庭试错成本高、文科就业不确定、医学培养周期长、工科行业波动、学校名气强但专业一般；不同家庭要给不同策略，不要一律套“普通家庭”。")
        lines.append("0候选兜底要求：如果Agent推荐结果里出现“替代路径”，回答必须明说这是原硬约束过窄后的替代方案，并按“放宽城市但保专业 / 保城市但换相近专业 / 保稳妥但降低学校层次”解释，不要再说完全没法同步学校。")
        lines.append("展开要求：每所学校都要单独写推荐理由，不允许把多所学校合并成一句；连续两所学校不得复用同一句“学校差异点/专业路径/普通家庭核验点”。如果学校同城同层次，也必须从办学背景、行业场景、学院/培养方案、校企资源、招生网核验入口里拆开。")
        lines.append("分层硬约束：冲稳保必须先看近年投档位次、院校层次、专业热度和行业辨识度，再看城市/偏好匹配。明显更强、分数位次通常更高的学校不得放在比弱校更低的档位；例如邮电/电子信息强校不应被放到普通地方工科院校之后当保底。")
        if user and user.city_preference:
            lines.append(f"城市硬约束复核：如果回答里出现不在「{'、'.join(user.city_preference)}」的学校，必须删除；不能因为学校层次高、专业强或候选不足而越过地区偏好。")
        major_text = "、".join(user.major_preference or []) if user else ""
        if self._is_digital_engineering_major(major_text):
            lines.append("普通家庭硬约束：保底不是浪费分数。高分高位次画像如果报计算机/电子信息，不要推荐行业辨识度弱的普通农林、师范、民族院校计算机作为主方案；除非官方投档位次证明它刚好贴近，并且回答中必须解释为什么不算浪费。")
        else:
            lines.append("普通家庭硬约束：保底不是浪费分数。必须围绕本轮意向专业写理由，不要把其他专业方向的就业路径、项目要求或行业出口套到当前专业上。")
        lines.append("张雪峰式核验：按就业倒推、中位数原则、社会筛子、500强测试、城市优先来写。普通家庭先问这个学校专业组合能不能过简历筛、能不能接行业岗位、能不能从普通毕业生中位数走通，而不是只看专业名字顺眼。")
        lines.append("展示限制：聊天主回答不要出现具体模拟概率和具体薪资数字；这些数值只放在 recommendation_plans 供同步方案使用。")
        lines.append("数据口径：当前Agent推荐使用本地院校/专业库；冲稳保参考是规则引擎模拟排序，不是真实录取概率；收入和就业指标为本地估算，不是官方精确统计；录取位次必须以教育考试院和学校招生网为准。")
        return "\n".join(lines)

    def _sanitize_recommend_summary_for_chat(self, summary: str) -> str:
        text = summary or ""
        text = re.sub(r"普通毕业生几年后的收入参考区间为本地估算：[^。]*。", "", text)
        text = re.sub(r"估算[^。；\n]*?(?:\d+\s*K|\d+%)[^。；\n]*[。；]?", "", text)
        text = re.sub(r"(?:约)?\d+\s*K(?:-\d+\s*K)?", "收入待就业质量报告核验", text)
        text = re.sub(r"\d+%", "粗排参考", text)
        return text.strip()

    def _build_school_chance_context(self, request: ConsultRequest, intent: IntentResult) -> str:
        school_name = intent.school_names[0] if intent.school_names else ""
        if not school_name:
            return ""

        user = self._build_user_preferences(request, allow_partial=True)
        ctx = request.context
        target_major = self._recommend_major_focus(user.major_preference if user else (ctx.major_preference if ctx else None))
        school = self.school_by_name.get(school_name, {})
        major = self.major_by_name.get(target_major) or (self.major_by_name.get((user.major_preference or [""])[0]) if user and user.major_preference else {})

        matched_plan = None
        recommend_summary = ""
        if user and user.province and user.score:
            recommend = agent_engine.recommend(RecommendRequest(user=user, limit=20))
            recommend_summary = recommend.summary
            matched_plan = next((plan for plan in recommend.plans if plan.school == school_name), None)

        risk_text = "未进入本轮结构化候选，需要以官方投档位次和专业组计划单独核验"
        if matched_plan:
            risk_text = f"后端规则把它放在「{matched_plan.risk_level}」档，专业方向按「{matched_plan.major}」核验"
        elif user and school and major:
            try:
                risk = agent_engine._profile_risk_bucket(school, major, user)
                combo = {"school": school, "major": major}
                probability = agent_engine._estimate_probability(combo, user, risk)
                risk_text = f"后端单校规则粗排为「{risk}」档；模拟概率仅供排序参考，内部值约 {probability}%"
            except Exception:
                risk_text = "本地规则没有稳定算出单校档位，需要回到考试院投档表和学校招生网核验"

        profile_line = self._profile_brief(ctx)
        school_url = school.get("official_url") or "未命中本地官网"
        distinctive = self._school_distinctive_angle(school, school_name, target_major) if school else "先按学校招生网和就业质量报告核验院系资源。"
        path = self._school_major_path_sentence(school, school_name, major or {}, target_major) if school else "先确认目标专业是否在本省招生、专业组是否满足选科。"

        lines = [
            "单校机会判断上下文：",
            f"用户本轮只问这一所学校：{school_name}。最终回答必须只围绕这所学校判断，不要展开成多所院校推荐列表。",
            f"考生画像：{profile_line}",
            f"目标专业方向：{target_major}",
            f"后端规则初筛：{risk_text}",
            f"结构化推荐摘要：{recommend_summary or '本轮未形成完整推荐摘要'}",
            f"学校差异点：{distinctive}",
            f"专业路径核验：{path}",
            f"本地官网入口：{school_url}",
            "联网/检索使用要求：优先结合教育考试院投档表、学校本科招生网、阳光高考/学信网信息；如果检索结果不足，要明说“先按本地规则粗判，最终看官方投档位次”。",
            "校区约束：如果用户目标地区包含北京，且本地学校定位为北京，本轮只讨论北京招生口径；不要主动扩展到保定校区或其他校区，除非用户明确追问。",
            "表达要求：先给结论，再说风险，不要问“哪个学校名字好听”；不要把机械问题回答成计算机、电子信息、经济、法学等无关方向；不要使用“稳稳的幸福、黄埔军校、长期霸榜、天作之合、没有之一”等未核验绝对化话术。",
        ]
        return "\n".join(lines)

    def _compose_recommendation_answer(
        self,
        request: ConsultRequest,
        plans: list[ConsultRecommendationPlan],
        research_status: str = "",
    ) -> str:
        ctx = request.context
        profile_bits = []
        if ctx:
            if ctx.province:
                profile_bits.append(ctx.province)
            if ctx.score:
                profile_bits.append(f"{ctx.score}分")
            if ctx.rank:
                profile_bits.append(f"位次{ctx.rank}")
            if ctx.subjects:
                profile_bits.append(ctx.subjects)
            if ctx.family_background:
                profile_bits.append(ctx.family_background)
            if ctx.risk_appetite:
                profile_bits.append("风险偏好" + ctx.risk_appetite)
            if ctx.major_preference:
                profile_bits.append("方向" + "、".join(ctx.major_preference[:2]))
        profile_text = "，".join(profile_bits) or "当前画像"
        major_focus = self._recommend_major_focus(ctx.major_preference if ctx else None)
        province_name = self._normalize_region_name(ctx.province) if ctx and ctx.province else "本省"
        exam_authority = self._exam_authority_name(province_name)

        risk_titles = {
            "冲": "冲刺档：",
            "稳": "稳妥档：",
            "保": "保底档：",
        }
        risk_notes = {
            "冲": "能冲，但别拿它当主心骨，后面必须有稳保接住。",
            "稳": "志愿表的主骨架，优先看学校和专业是不是同时不亏。",
            "保": "防滑档，但不能糟蹋分数，重点看专业组和调剂风险。",
        }

        lines = [
            "[分析过程]",
            f"1. 画像拆解：{profile_text}。我先看省份、位次、选科、家庭条件和专业方向，不先看学校名头。",
            f"2. 策略筛选：这轮按{major_focus}倒推就业路径，再用冲稳保分层，防止拿学校最低投档线冒充热门专业能上。",
            "3. 风险控制：普通家庭填志愿，先保证稳妥档和保底档能接住，再谈冲刺档抬天花板。",
            "",
            "[核心判断]",
            f"我跟你说，你这个画像（{profile_text}），别先问“哪个学校名字好听”，先问这条路能不能换饭碗。",
            f"具体学校我会放在后面分三档讲：按位次分层，再按{major_focus}的真实出口筛。别拿学校最低投档线当专业线，尤其别拿冷门专业最低位次骗自己能上热门专业。",
            "",
            "[灵魂追问]",
            "- 第一，目标城市是硬约束，还是为了专业和学校层次可以适当让一步？",
            "- 第二，能不能接受专业组内调剂？如果不能，必须逐个查专业组和招生计划。",
            "- 第三，家庭能不能支持考研、实习和证书投入？这决定冲刺档能不能冒险。",
            "",
            "[院校推荐]",
        ]
        grouped = self._group_chat_plans_by_risk(self._merge_school_chat_plans(plans)[:10])
        for risk in ["冲", "稳", "保"]:
            risk_plans = grouped.get(risk) or []
            if not risk_plans:
                continue
            names = "、".join(f"{plan.school}（{plan.major}）" for plan in risk_plans)
            lines.append("")
            lines.append(f"{risk_titles.get(risk, risk + '档：')}{names}。{risk_notes.get(risk, '')}")
            for plan in risk_plans:
                lines.append(self._build_chat_plan_line(plan, self._normalize_region_name(ctx.province) if ctx and ctx.province else None))

        lines.extend([
            "",
            "[红旗风险]",
            self._major_strategy_sentence(major_focus),
            self._family_strategy_sentence(ctx.family_background if ctx else None),
            "",
            "[核验清单]",
            f"1. 第一查{exam_authority}近三年投档表，看院校专业组最低位次和专业冷热差。",
            "2. 第二查学校招生网专业组、选科要求、招生计划和调剂规则。",
            "3. 第三查学院培养方案和就业质量报告，确认这个专业在这所学校有没有真实出口。",
            "4. 第三方平台只能当入口，不能当结论。",
            "",
            "[金句]",
            "普通家庭填志愿，先让稳保接住孩子，再让冲刺抬高天花板。",
            "",
            "数据口径：当前推荐使用本地院校/专业库和规则模拟排序，不是真实录取概率；最终投档位次、招生计划、专业组和调剂风险必须回官方渠道核验。",
        ])
        if research_status:
            lines.extend(["", research_status])
        return "\n".join(lines)

    def _family_strategy_sentence(self, family_background: str | None) -> str:
        family = family_background or "普通家庭"
        if "富裕" in family:
            return "资源较足的家庭可以追一点平台和长期成长，但别把热爱当免死金牌：学校过筛子、专业有出口、城市有资源，这三件事仍然要同时成立。"
        if "中产" in family:
            return "中产家庭可以留一点试错空间，但不能把四年押在空泛名头上：学校过筛子、专业有出口、城市有实习，这三件事缺一个都要谨慎。"
        return "普通家庭的孩子，别拿大学四年换一个装点门面的专业。你得换一个能装饭的碗：学校过筛子、专业有出口、城市有实习，这三件事缺一个都要谨慎。"

    def _major_strategy_sentence(self, major_focus: str) -> str:
        if "法学" in major_focus or "政治" in major_focus:
            return "专业上：法学不是背书专业，是法考、实习和城市资源专业。能进政法强校最好；进综合大学也行，但必须把法考、律所/法院检察院实习、考公路径提前设计好。"
        if "电气" in major_focus or "自动化" in major_focus:
            return "专业上：电气/自动化别只盯名字，要看它接的是电网、电力设备、工业控制、智能制造，还是学校自己凑出来的专业。能进电力和工程现场，这个专业才值钱。"
        if "电子信息" in major_focus or "通信" in major_focus:
            return "专业上：电子信息别只喊热门，要看通信、嵌入式、硬件系统、信号处理、传感器和校企实验项目。普通家庭最怕只拿一个专业名，最后没有实验项目也没有行业场景。"
        if "计算机" in major_focus or "软件" in major_focus or "人工智能" in major_focus or "数据" in major_focus:
            return "专业上：计算机别只喊互联网大厂，要看软件工程、网安、数据平台和行业系统这些具体出口。普通家庭最怕学成泛泛写代码，最后没有学校标签也没有项目作品。"
        return "专业上：别只看专业名字顺眼，要看这个专业在这所学校有没有学院资源、项目训练、实习半径和真实就业出口。"

    def _group_chat_plans_by_risk(self, plans: list[ConsultRecommendationPlan]) -> dict[str, list[ConsultRecommendationPlan]]:
        grouped: dict[str, list[ConsultRecommendationPlan]] = {"冲": [], "稳": [], "保": []}
        for plan in plans:
            grouped.setdefault(plan.risk_level, []).append(plan)
        return grouped

    def _recommend_major_focus(self, preferences: list[str] | None) -> str:
        text = "、".join(preferences or [])
        if any(key in text for key in ["法学", "政治", "公安", "马克思"]):
            return "法学/政治方向"
        if any(key in text for key in ["电子信息", "通信", "电子科学", "微电子", "集成电路"]):
            return "电子信息"
        if any(key in text for key in ["计算机", "软件", "人工智能", "数据", "信息安全"]):
            return "计算机/软件"
        if text:
            return text
        return "目标专业"

    def _exam_authority_name(self, province: str) -> str:
        names = {
            "北京": "北京教育考试院",
            "天津": "天津市教育招生考试院",
            "河北": "河北省教育考试院",
            "山西": "山西招生考试网",
            "内蒙古": "内蒙古招生考试信息网",
            "辽宁": "辽宁招生考试之窗",
            "吉林": "吉林省教育考试院",
            "黑龙江": "黑龙江省招生考试信息港",
            "上海": "上海市教育考试院",
            "山东": "山东省教育招生考试院",
            "江苏": "江苏省教育考试院",
            "浙江": "浙江省教育考试院",
            "广东": "广东省教育考试院",
            "安徽": "安徽省教育招生考试院",
            "福建": "福建省教育考试院",
            "江西": "江西省教育考试院",
            "河南": "河南省教育考试院",
            "湖北": "湖北省教育考试院",
            "湖南": "湖南省教育考试院",
            "广西": "广西招生考试院",
            "海南": "海南省考试局",
            "重庆": "重庆市教育考试院",
            "四川": "四川省教育考试院",
            "贵州": "贵州省招生考试院",
            "云南": "云南省招生考试院",
            "西藏": "西藏自治区教育考试院",
            "陕西": "陕西省教育考试院",
            "甘肃": "甘肃省教育考试院",
            "青海": "青海省教育招生考试院",
            "宁夏": "宁夏教育考试院",
            "新疆": "新疆教育考试院",
        }
        return names.get(province, f"{province}教育考试院")

    def _append_recommendation_detail(
        self,
        answer: str,
        plans: list[ConsultRecommendationPlan],
    ) -> str:
        has_detail_block = "院校逐个展开" in answer
        if has_detail_block:
            return answer

        lines = [
            answer.strip(),
            "",
            "院校逐个展开：",
        ]
        current_risk = ""
        for plan in self._merge_school_chat_plans(plans)[:10]:
            if plan.risk_level != current_risk:
                current_risk = plan.risk_level
                lines.append("")
                lines.append(f"{current_risk}档：")
            lines.append(self._build_chat_plan_line(plan))

        lines.extend([
            "",
            "同步提示：下方点击“同步到我的当前方案”，会把本轮识别出的院校、专业、推荐理由、估算概率和薪资同步到方案对比页；当前画像最多保留10所，新同步会替换旧方案。",
        ])
        return "\n".join(part for part in lines if part is not None).strip()

    def _merge_school_chat_plans(self, plans: list[ConsultRecommendationPlan]) -> list[ConsultRecommendationPlan]:
        """聊天展示按学校合并，避免同校不同专业挤占展示名额。"""
        merged: list[ConsultRecommendationPlan] = []
        by_school: dict[str, int] = {}
        for plan in plans:
            index = by_school.get(plan.school)
            if index is None:
                by_school[plan.school] = len(merged)
                merged.append(plan)
                continue
            existing = merged[index]
            majors = [item.strip() for item in existing.major.split("/") if item.strip()]
            if plan.major not in majors:
                majors.append(plan.major)
            merged[index] = existing.model_copy(update={"major": "/".join(majors[:3])})
        return [plan.model_copy(update={"order": index}) for index, plan in enumerate(merged, start=1)]

    def _build_chat_plan_line(self, plan: ConsultRecommendationPlan, applicant_province: str | None = None) -> str:
        school = self.school_by_name.get(plan.school, {})
        primary_major = plan.major.split("/")[0]
        major = self.major_by_name.get(primary_major, {})
        raw_reason = self._zxf_chat_reason(school, major, plan.school, plan.major, plan.risk_level)
        reason = self._polish_chat_reason(
            raw_reason,
            school,
            major,
            plan.school,
            plan.major,
            plan.risk_level,
            applicant_province,
        )
        admissions_entry = plan.admissions_url or self._school_admissions_entry_url(school, plan.school)
        admissions_query = plan.admissions_query or self._school_admissions_query(applicant_province, plan.school, primary_major)
        admissions_note = (
            f"学校官网入口：{admissions_entry}；本科招生网检索词：{admissions_query}。"
            if admissions_entry
            else f"本科招生网检索词：{admissions_query}。"
        )
        risk_tags = plan.risk_tags or build_family_risk_profile(
            {**school, "name": plan.school},
            {**major, "name": primary_major},
            None,
            plan.risk_level,
        )["risk_tags"]
        tag_note = f"家庭风险标签：{'、'.join(risk_tags[:3])}。" if risk_tags else ""
        family_text = (plan.family_strategy or "").strip()
        family_note = f"{family_text}{'' if family_text.endswith('。') else '。'}" if family_text else ""
        fallback_note = f"替代路径：{plan.fallback_strategy}。" if plan.fallback_strategy else ""
        return f"{plan.school}：{fallback_note}{reason} {tag_note}{family_note}{admissions_note}"

    def _chat_risk_sentence(self, risk_level: str) -> str:
        if risk_level == "冲":
            return "冲。"
        if risk_level == "保":
            return "保。"
        return "稳。"

    def _polish_chat_reason(
        self,
        reason: str,
        school: dict,
        major: dict,
        school_name: str,
        major_name: str,
        risk_level: str,
        applicant_province: str | None = None,
    ) -> str:
        """把逐校理由压成更像真人咨询的一段话，并强制贴合本轮专业。"""
        aligned = self._align_reason_to_major(reason, major_name)
        if self._is_generic_chat_reason(aligned):
            aligned = self._personalized_chat_reason(school, major, school_name, major_name, risk_level)
        aligned = self._dedupe_reason_sentences(aligned)
        aligned = self._align_reason_to_major(aligned, major_name)
        if self._is_mismatched_major_hint(aligned, major_name):
            aligned = self._personalized_chat_reason(school, major, school_name, major_name, risk_level)
        return aligned

    def _is_generic_chat_reason(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        generic_markers = [
            "工程训练和实验平台更匹配",
            "要看实验课、工程训练、校企项目和真实行业现场",
            "先看专业组位次，再看学校行业标签",
            "平台有筛选价值，但",
            "学校层次、城市和专业组需要一起核验",
            "不能只按校名冷热判断",
            "要多看项目制课程",
            "这条路核心看项目经历",
            "关键看该专业是不是放在核心学院",
        ]
        return any(marker in compact for marker in generic_markers)

    def _dedupe_reason_sentences(self, text: str) -> str:
        sentences = [item.strip() for item in re.split(r"[。；]\s*", text or "") if item.strip()]
        kept: list[str] = []
        seen_tokens: set[str] = set()
        for sentence in sentences:
            key = re.sub(r"[，,、：:\s]", "", sentence)
            if not key or key in seen_tokens:
                continue
            if any(key in old or old in key for old in seen_tokens):
                continue
            kept.append(sentence)
            seen_tokens.add(key)
            if len(kept) >= 3:
                break
        return "。".join(kept).rstrip("。") + "。" if kept else text

    def _align_reason_to_major(self, text: str, major_name: str) -> str:
        """数字类专业不能互相串台：电子信息别写成计算机，电气别写成软件。"""
        result = text or ""
        kind = self._digital_major_kind(major_name)
        if kind == "electronic":
            replacements = {
                "计电": "电子信息",
                "计算机/电子信息": "电子信息",
                "计算机和电子信息": "电子信息",
                "计算机方向": "电子信息方向",
                "学计算机": "学电子信息",
                "计算机要": "电子信息要",
                "计算机最好": "电子信息最好",
                "计算机不是": "电子信息不是",
                "计算机更": "电子信息更",
                "计算机往": "电子信息往",
                "把计算机": "把电子信息",
                "纯计算机": "纯电子信息",
                "普通计算机": "普通电子信息",
                "泛泛写代码": "只喊电子信息",
                "前端后端": "硬件调试和系统集成",
                "软件岗": "电子/通信岗位",
                "低端代码岗": "低端实施岗",
                "项目作品": "硬件实验、通信/嵌入式项目",
                "开源作品": "硬件实验、通信/嵌入式项目",
                "工业软件": "工业通信、传感器和嵌入式控制",
                "普通软件": "普通电子信息",
                "写代码": "做硬件调试和系统集成",
                "算法/开发基础": "电路、通信和嵌入式基础",
                "低端重复开发": "低端实施",
                "工程软件": "工程电子、嵌入式和系统集成",
            }
            for old, new in replacements.items():
                result = result.replace(old, new)
        elif kind == "computer":
            result = result.replace("电气/自动化", "计算机/软件")
            result = result.replace("电气自动化", "计算机/软件")
            result = result.replace("电网、电力设备、工业控制、智能制造", "软件工程、数据平台、网络安全、行业系统")
        elif kind in ["electrical", "automation"]:
            replacements = {
                "计电": major_name,
                "计算机/电子信息": major_name,
                "计算机和电子信息": major_name,
                "计算机方向": f"{major_name}方向",
                "电子信息方向": f"{major_name}方向",
                "学计算机": f"学{major_name}",
                "计算机要": f"{major_name}要",
                "电子信息要": f"{major_name}要",
                "互联网大厂": "电力、装备和工业现场",
                "纯互联网": "纯互联网",
                "泛泛写代码": "只喊专业名",
                "软件岗": "电力/控制/装备岗位",
                "前端后端": "电力系统、控制和设备调试",
                "开源作品": "实验课、控制项目和工程现场经历",
            }
            for old, new in replacements.items():
                result = result.replace(old, new)
        return result

    def _digital_major_kind(self, major_name: str) -> str:
        name = major_name or ""
        if any(key in name for key in ["电气", "电力", "供用电"]):
            return "electrical"
        if any(key in name for key in ["自动化", "控制"]):
            return "automation"
        if any(key in name for key in ["电子信息", "通信", "电子科学", "微电子", "集成电路", "光电", "物联网"]):
            return "electronic"
        if any(key in name for key in ["计算机", "软件", "人工智能", "数据科学", "信息安全", "网络空间"]):
            return "computer"
        return "digital" if self._is_digital_engineering_major(name) else "other"

    def _personalized_chat_reason(
        self,
        school: dict,
        major: dict,
        school_name: str,
        major_name: str,
        risk_level: str,
    ) -> str:
        name = school_name or school.get("name", "这所学校")
        city = school.get("city") or school.get("province") or "当地"
        level = school.get("level", "")
        school_type = school.get("type", "")
        trait = self._school_trait_phrase(school, name, major_name)
        route = self._major_route_phrase(school, major, name, major_name)
        check = self._risk_check_phrase(risk_level, level, school_type, major_name)
        return self._compact_school_reason(trait, route, check)

    def _school_trait_phrase(self, school: dict, school_name: str, major_name: str) -> str:
        name = school_name or school.get("name", "这所学校")
        city = school.get("city") or school.get("province") or "当地"
        school_type = school.get("type", "")
        level = school.get("level", "")
        if "邮电" in name:
            return f"{name}的底色是通信、网络和信息系统，报{major_name}至少有行业标签，不是泛泛综合平台"
        if "电力" in name:
            return f"{name}的电力行业标签很直，报{major_name}要往电网、新能源和电力设备场景上解释"
        if "电子" in name:
            return f"{name}的电子信息识别度更强，报{major_name}要把硬件、通信和系统项目查实"
        if "信息工程" in name or "信息科技" in name:
            return f"{name}的信息类应用场景更密，报{major_name}要看学院资源和行业项目是不是同方向"
        if "上海科技大学" in name:
            return f"{name}看的是科研训练和硬科技氛围，报{major_name}要确认实验室、导师方向和本科项目是不是接得住"
        if "上海理工大学" in name:
            return f"{name}的底色更偏光电、仪器、医疗器械和工程应用，报{major_name}要往设备系统和智能制造上靠"
        if "上海海事大学" in name:
            return f"{name}的场景在航运、港口、船舶和物流系统，报{major_name}要接受海事行业这个垂直出口"
        if "上海工程技术大学" in name:
            return f"{name}是应用型工程路线，优势不在名头，而在轨道交通、汽车制造和产教融合项目"
        if "上海立信会计金融学院" in name:
            return f"{name}的主标签是金融会计，报{major_name}必须找金融科技、数据合规或企业系统这种结合点"
        if "理工" in name or "工业" in name or "工程" in name or school_type in ["理工", "工科"]:
            return f"{name}不是靠名字好听吃饭，价值在工程底盘和{city}产业现场，适合把{major_name}落到项目里"
        if school_type == "财经政法":
            return f"{name}的优势在财经政法资源，报{major_name}必须找和本专业能接上的行业场景，不能硬套学校强项"
        if school_type == "师范":
            return f"{name}的师范和文理平台是底色，报{major_name}要确认学院资源，别把师范稳定直接套到专业上"
        if school_type == "农林海洋":
            return f"{name}的农林海洋特色明显，报{major_name}要接受垂直行业场景，别当普通综合大学理解"
        if level in ["985", "211", "双一流"]:
            return f"{name}的平台能过一层筛子，但{major_name}不能只靠校名，专业组和学院资源要单独核验"
        return f"{name}看的是{city}本地认可、专业组干净度和实习半径，别只按校名冷热判断"

    def _major_route_phrase(self, school: dict, major: dict, school_name: str, major_name: str) -> str:
        kind = self._digital_major_kind(major_name)
        name = school_name or school.get("name", "")
        city = school.get("city") or school.get("province") or "当地"
        if kind == "electronic":
            if "邮电" in name:
                return "电子信息要重点看通信网络、运营商/设备商、嵌入式和信号处理项目，岗位解释比较顺"
            if "电力" in name:
                return "电子信息可以往电力电子、智能运维、能源通信和设备监测靠，别只按普通电子专业想"
            if "海洋" in name:
                return "电子信息要和海洋观测、传感器、数据采集和设备系统结合，接受行业场景才有辨识度"
            if "海事" in name:
                return "电子信息要看导航通信、船舶电子、港口自动化和物流信息系统，别按普通硬件专业理解"
            if "上海理工" in name:
                return "电子信息要重点看光电仪器、医疗器械电子、传感器和智能制造项目，学校工程应用场景比较实"
            if "上海科技" in name:
                return "电子信息要看科研项目、芯片/器件、智能感知和硬科技实验室，适合能扛数理和项目强度的学生"
            if "工程技术" in name:
                return "电子信息要落到轨道交通电子、汽车电子、智能装备和企业工程项目，应用型路线要查实训平台"
            if "立信" in name:
                return "电子信息在这里不能硬讲电子强校，要往金融科技系统、数据治理和企业信息化找结合点"
            return f"电子信息要查电路与系统、通信、嵌入式、传感器和校企实验项目，{city}实习资源要能接上"
        if kind == "computer":
            return f"计算机要看软件工程、数据平台、网安或行业系统项目，普通家庭要靠项目和实习补筛子"
        if kind in ["electrical", "automation"]:
            if "电力" in name or "河海" in name:
                return "这条路要盯电网调度、继电保护、新能源系统和工业控制，行业越清楚越值钱"
            return f"{major_name}要看强弱电、控制、智能制造和设备调试，能不能进工程现场比专业名头更关键"
        if self._is_economics_major(major_name):
            return f"{major_name}要看计量统计、财经实习、读研去向和{city}岗位半径，不能套理工项目逻辑"
        if self._is_chemistry_major(major_name):
            return f"{major_name}要看实验平台、材料/化工/检测方向和读研去向，不能套数字专业叙事"
        return self._school_major_path_sentence(school, name, major, major_name)

    def _risk_check_phrase(self, risk_level: str, level: str, school_type: str, major_name: str) -> str:
        if risk_level == "冲":
            return f"冲它是抬上限，不是当主心骨；必须用近三年专业组位次核验{major_name}有没有真实机会"
        if risk_level == "保":
            return f"保底不是随便填，重点查专业组里有没有不能接受的调剂方向，别最后被调走"
        if level in ["985", "211", "双一流"]:
            return f"{level}标签有筛选价值，但{major_name}要看具体学院和培养方案，不能拿最低投档线糊弄自己"
        return f"稳档看的是学校、城市和专业出口能不能同时不亏，下一步查招生网和培养方案"

    def _zxf_chat_reason(self, school: dict, major: dict, school_name: str, major_name: str, risk_level: str) -> str:
        name = school_name or school.get("name", "")
        major_primary = major_name.split("/")[0]
        level = school.get("level", "")
        school_type = school.get("type", "")
        is_electrical = any(key in major_name for key in ["电气", "自动化", "控制"])

        named = [
            ("山东科技大学", "山科大计算机不是空壳，计算机科学与技术有国家级特色/一流底子，还接智慧矿山、矿山物联网、人工智能平台。报计电要往工业互联网、智慧矿山、物联网和安全系统靠。"),
            ("青岛科技大学", "青科大底子是化工、材料和装备制造，信息学院和工业信息化、化工过程装备仿真结合紧。计电方向更适合做工业软件、流程控制、医疗健康软件，不是纯互联网叙事。"),
            ("青岛理工大学", "青岛理工强在土木建筑、工程管理和城市建设场景。计电要往智慧建造、BIM、建筑电气、城市物联网靠，别把它当纯计算机强校。"),
            ("齐鲁工业大学", "齐鲁工大背后是山东省科学院，计算机学部和国家超算济南中心、算力互联网、信息安全平台关系更紧。计电适合看高性能计算、网安、云边协同和工业数字化。"),
            ("山东理工大学", "山东理工的工科场景更贴车辆、机械、电气和制造业。计电要往车联网、嵌入式、智能制造、企业系统靠，别只盯普通软件岗。"),
            ("青岛大学", "青岛大学是青岛综合平台，优势在城市、医学、企业资源和选择面。计电可以做医疗信息化、企业数字化、海洋城市数据，但专业组必须查干净。"),
            ("济南大学", "济南大学是省会综合平台，适合借济南软件园、政企信息化和本地实习。短板是计算机标签不如山科、齐鲁工大直，必须靠项目作品补筛子。"),
            ("烟台大学", "烟台大学看的是沿海城市、地方企业和综合平台。计电要往智能制造、海洋装备、企业信息化靠，适合保底，别指望校名替你筛人。"),
            ("东南大学", "东南是硬工科强校，电气/自动化冲它是抬平台上限。它的价值在电气控制、智能制造和强工程训练，冲上了也得靠项目接住。"),
            ("南京邮电大学", "南邮底子在通信、网络和电子信息，电气自动化要往通信设备、云网基础设施、智能运维靠。它不是电网传统强校，但行业识别度不差。"),
            ("南京信息工程大学", "南信大强在气象、遥感、数据平台和行业信息化。电气/自动化要接气象装备、监测系统、数据平台，不是走传统电力老路。"),
            ("苏州大学", "苏大是211综合平台，电气自动化要借苏州制造业、生物医药设备和园区企业项目。校名能过筛子，但专业出口要自己做硬。"),
            ("河海大学", "河海的底色是水利、电力和工程系统。电气/自动化要往水电站、能源调度、工业控制和智慧水利靠，这比泛泛写代码更贴学校。"),
            ("南京农业大学", "南农读电气自动化别装纯电力强校，它的结合点在农业装备、智慧农业、食品加工自动化。适合能接受行业场景的人。"),
            ("中国药科大学", "药科读电气自动化，方向要往药企自动化产线、制药设备、质量控制系统走。别拿药科牌子去讲传统电网，那就拧巴了。"),
            ("江苏科技大学", "江苏科技靠船舶、海工装备和制造业工程底盘。电气/自动化要往船舶电气、智能装备、控制系统靠，不是普通低端代码岗。"),
            ("苏州科技大学", "苏科大更贴苏州城市建设、智能建造和长三角工程企业。电气/自动化要往建筑电气、物联网监测、智慧运维走。"),
            ("江苏大学", "江苏大学电气是国家特色专业，强弱电、机电、软硬件结合是它的主线。更适合盯电气装备、车辆电控、智能制造。"),
            ("扬州大学", "扬大是综合大学，优势在省内认可和农工医交叉。电气/自动化要找农业装备、地方制造业、电力运维项目，别只图综合名头。"),
            ("常州大学", "常州大学背后是常州制造业、新能源和化工装备场景。电气/自动化要往新能源装备、过程控制、工厂自动化靠。"),
            ("南京工程学院", "南京工程学院电气是看家饭，电力系统、继电保护、输配电和供用电方向很对口。它不是来炫名头的，是奔电网和工程现场去的。"),
            ("南京工业大学", "南京工业优势在化工、材料、安全工程和制造业底盘。电气/自动化要落到流程控制、工业软件、安全生产信息化和工厂自动化。"),
            ("华南理工大学", "华工是广东硬工科头牌，电子信息冲它是抬平台上限。能不能冲成，看专业组位次；冲上了别躺平，项目和实验室要跟上。"),
            ("暨南大学", "暨南是211综合平台，电子信息不是它最强标签。它的价值在广州、综合资源和校友圈，计电出口要靠自己把项目做硬。"),
            ("南方科技大学", "南科大适合数理强、愿意卷科研和硬科技的人。电子信息能接深圳产业，但培养节奏不轻松，普通家庭别只看新校名气。"),
            ("华南农业大学", "华农报电子信息要清醒：优势不是纯电子，而是农业装备、食品安全、智慧农业的数据和硬件场景。分数高时别把它当主菜。"),
            ("广东工业大学", "广工不是虚名校，它吃的是广东制造业和电子信息产业链。电子信息看腾创班、腾讯实践课、嵌入式和工业互联网，普通家庭要盯项目而不是只看校名。"),
            ("东莞理工学院", "东莞理工的电子信息要看松山湖、华为、OPPO和先进制造生态。它不是靠牌子筛人，是靠产业半径和工程项目吃饭。"),
            ("深圳大学", "深大最大的卖点是深圳本地产业，不是传统985/211标签。电子信息要抢企业实习、硬件产品、通信和金融科技项目，城市机会多，竞争也狠。"),
            ("香港中文大学（深圳）", "港中深适合能承受学费、英语环境和升学节奏的家庭。电子信息要往AI交叉、数据科学和深圳产业链走，不是低成本保底。"),
            ("深圳北理莫斯科大学", "深北莫适合接受中外合作、数理训练和语言环境的学生。电子信息方向要看课程体系、读研路径和家庭现金流，别只看深圳两个字。"),
            ("深圳技术大学", "深技大是应用技术和产教融合路线。电子信息要看企业导师、实训平台、机器人和智能制造项目，适合想早进工程现场的人。"),
            ("广东外语外贸大学", "广外报电子信息要想清楚：学校强项是语言、外贸和国际商务。真要读计电，就得往跨境电商技术、数据合规、国际产品支持找结合点。"),
            ("广州大学", "广大是广州本地综合平台，电子信息适合接城市治理、智慧交通、物联网和本地企业数字化。别把它吹成电子强校，但保底不算离谱。"),
            ("汕头大学", "汕大有综合大学和港资办学背景，电子信息要看粤东产业、医疗信息化和智能制造场景。离广州深圳产业核心远一点，这是成本。"),
            ("广东药科大学", "广药报电子信息不是走纯硬件强校路线，要和医药数据、医疗设备、药企信息化结合。能当保底，但别拿它和广工、深大比电子底盘。"),
            ("华东师范大学", "985平台能过社会筛子，但法学不是它最硬的饭碗。冲它可以，别拿师范名气替代法考、律所实习和专业位次。"),
            ("上海财经大学", "上财的价值在财经圈筛子，法学要往金融合规、证券基金、税务风控靠。普通家庭别只喊211，要看能不能进金融法项目。"),
            ("华东政法大学", "法学在上海最该重点看的学校之一，行业标签比很多综合211更直接。你学法不是看热闹，是看法考、律所、法院检察院实习半径。"),
            ("上海政法学院", "名字不如华政响，但法学出口比很多泛综合院校更直。适合当稳保之间的务实选择，关键看法考支持和实习基地。"),
            ("上海对外经贸大学", "它的法学别按普通法学理解，要看涉外法务、贸易合规、知识产权和企业法务。上海岗位半径是优势。"),
            ("上海师范大学", "上师大法学不是冲名校，是看上海本地认可、考公考编和教育治理相关出口。普通家庭要把稳定路径算进去。"),
            ("上海立信会计金融学院", "立信的标签是会计金融，法学要往金融监管、合规、审计法务走。它适合做保底，但别幻想它给你顶级律所光环。"),
            ("上海应用技术大学", "这是保底逻辑，不是牌子逻辑。能不能报，看专业组干不干净、调剂会不会把你调到不想去的方向。"),
            ("东华大学", "211牌子有用，但法学不是它最强标签。冲它可以，别拿纺织材料的学校强项替代法学就业证据。"),
            ("上海大学", "上大是上海综合平台，法学能借城市资源和综合大学牌子。问题是热门专业位次要单独查，别用最低组位次糊弄自己。"),
            ("山东大学", "省内985门票值钱，但计算机别拿学校最低线糊弄自己；要分清本部、威海和具体专业代码。适合冲刺抬上限，稳保必须另放。"),
            ("中国海洋大学", "985牌子加青岛位置能过社会筛子，计算机不是它最硬的传统王牌，但海洋数据、遥感监测、智慧海洋有交叉出口。适合冲，不适合当保底。"),
            ("南京邮电大学", "这校别只看不是985、211，邮电底子对通信、网安、运营商、设备商很有辨识度。学计电方向，比很多泛综合大学更对口。"),
            ("南京信息工程大学", "它不是普通计算机叙事，强在气象、遥感、数据平台和行业信息化。你要接受它走行业数据路线，不是纯互联网大厂路线。"),
            ("南京航空航天大学", "211加航空航天工科底盘，简历筛子比普通一本强一截。计电要往嵌入式、工业软件、航空电子和智能制造靠。"),
            ("南京理工大学", "211工科牌子硬，电子信息、自动化、兵工背景能给项目场景。普通家庭看它，不是图浪漫，是图筛子和工程训练。"),
            ("南京工业大学", "保底可以看，但别当成捡漏名校；优势在化工材料和制造业底盘。计电要落到工业软件、流程控制、安全生产信息化。"),
            ("南京工程学院", "这是保底里看工程落地的学校，不是拿来炫名头的。能不能选，关键看专业组、实验课、校企项目和南京实习半径。"),
            ("中国石油大学", "211牌子和能源行业底盘都在，计电要往能源数字化、工业软件、油气生产系统走。接受行业场景，它就有价值。"),
        ]
        electrical_keys = {
            "东南大学", "南京邮电大学", "南京信息工程大学", "苏州大学", "河海大学",
            "南京农业大学", "中国药科大学", "江苏科技大学", "苏州科技大学",
            "江苏大学", "扬州大学", "常州大学", "南京工程学院", "南京工业大学",
        }
        for key, value in named:
            if key in electrical_keys and not is_electrical and ("电气" in value or "自动化" in value):
                continue
            if self._is_mismatched_major_hint(value, major_name):
                continue
            if key in name:
                return value

        domain = self._major_domain(major_name)
        if domain not in ["general", "digital"]:
            angle = self._school_distinctive_angle(school, name, major_name)
            path = self._school_major_path_sentence(school, name, major, major_name)
            level_note = self._school_level_filter_sentence(level, major_primary) if level in ["985", "211", "双一流"] else ""
            return self._compact_school_reason(angle, path, level_note)

        if "东南大学" in name:
            return "东南是硬工科强校，计算机/电子信息是正经硬菜；这个分数冲它是抬上限，不是稳。普通家庭要冲可以冲，但后面一定接稳。"

        if "邮电" in name or "电子" in name:
            return "计电方向最怕学校没行业标签，这类学校至少有通信、电子、网安的岗位解释。普通家庭看的是出口，不是校名好不好听。"
        if "法学" in major_name or "政治" in major_name:
            if "政法" in name:
                return "法学最怕学校没行业入口，政法类院校至少知道你往哪走。重点看法考、实习基地和公检法律所去向。"
            if level in ["985", "211"]:
                return f"{level}平台有筛子价值，但法学要单查专业组位次和法学院资源。别拿学校最低线冒充法学线。"
            if "师范" in name:
                return "师范院校学法，要把考公、教育治理、基层法律服务这些出口想清楚。别幻想它自动等于红圈律所。"
            return "法学不是背书专业，是证书和实习专业。普通家庭重点看法考通过支持、实习半径和能不能考公考编。"
        if self._is_economics_major(major_name):
            angle = self._school_distinctive_angle(school, name, major_name)
            path = self._school_major_path_sentence(school, name, major, major_name)
            level_note = self._school_level_filter_sentence(level, major_primary) if level in ["985", "211", "双一流"] else ""
            return self._compact_school_reason(angle, path, level_note)
        if self._is_digital_engineering_major(major_name):
            angle = self._school_distinctive_angle(school, name, major_name)
            path = self._school_major_path_sentence(school, name, major, major_name)
            level_note = "" if angle and path else self._school_level_filter_sentence(level, major_primary)
            return self._compact_school_reason(angle, path, level_note)
        if level in ["985", "211"]:
            angle = self._school_distinctive_angle(school, name, major_name)
            level_note = self._school_level_filter_sentence(level, major_primary)
            return self._compact_school_reason(angle, "", level_note)
        if "农业" in name or school_type == "农林海洋":
            return "高分报农林院校计电要非常谨慎，除非投档位次贴近且专业组干净。否则就是用高分买低辨识度。"
        return "先看专业组位次，再看学校行业标签，最后看项目和实习。普通家庭别被名字带节奏。"

    def _is_mismatched_major_hint(self, text: str, major_name: str) -> bool:
        """固定校名文案必须和本轮专业同方向，避免计算机问题冒出法学/师范等提示。"""
        if not text:
            return False
        is_digital = self._is_digital_engineering_major(major_name)
        is_humanities = self._is_humanities_major(major_name)
        is_law = "法学" in major_name or "政治" in major_name
        is_medical = any(key in major_name for key in ["医学", "临床", "口腔", "药学", "护理"])
        is_chemistry = self._is_chemistry_major(major_name)
        is_economics = self._is_economics_major(major_name)
        domain = self._major_domain(major_name)
        digital_kind = self._digital_major_kind(major_name)

        law_terms = ["法学", "法考", "律所", "法院", "检察院", "政法", "法律"]
        humanities_terms = law_terms + ["师范", "中文", "历史", "新闻", "传播", "考编"]
        digital_terms = ["计算机", "计电", "电子信息", "通信", "网安", "软件", "算法", "互联网", "嵌入式", "工业软件"]
        medical_terms = ["医学", "临床", "口腔", "医院", "药学", "护理", "规培"]
        economics_terms = ["经济", "金融", "财政", "税收", "贸易", "银行", "券商", "会计", "审计"]
        education_terms = ["师范", "教师", "教育", "教资", "考编", "学校"]
        agriculture_terms = ["农业", "农学", "作物", "种业", "畜牧", "兽医", "乡村"]
        art_terms = ["艺术", "设计", "美术", "音乐", "传媒", "作品集"]

        if is_digital and any(term in text for term in humanities_terms + medical_terms):
            return True
        if digital_kind == "electronic" and any(
            term in text
            for term in ["学计算机", "计算机要", "计算机最好", "纯计算机", "普通计算机", "前端后端", "泛泛写代码", "软件岗"]
        ):
            return True
        if digital_kind in ["electrical", "automation"] and any(
            term in text
            for term in ["计算机", "计电", "电子信息", "通信软件", "网安", "互联网大厂", "前端后端", "泛泛写代码", "软件岗"]
        ):
            return True
        if is_law and any(term in text for term in digital_terms + medical_terms):
            return True
        if is_humanities and not is_law and any(term in text for term in digital_terms + medical_terms):
            return True
        if is_medical and any(term in text for term in digital_terms + humanities_terms):
            return True
        if is_chemistry and any(term in text for term in digital_terms + law_terms + ["师范", "考编"]):
            return True
        if is_economics and any(term in text for term in digital_terms + medical_terms + ["法考", "律所", "法院", "检察院"]):
            return True
        if domain == "medical" and any(term in text for term in digital_terms + law_terms + economics_terms + education_terms):
            return True
        if domain == "education" and any(term in text for term in digital_terms + medical_terms + economics_terms + law_terms):
            return True
        if domain == "agriculture" and any(term in text for term in digital_terms + law_terms + medical_terms):
            return True
        if domain == "art" and any(term in text for term in digital_terms + medical_terms + economics_terms + law_terms):
            return True
        if domain == "traditional_engineering" and any(term in text for term in digital_terms + medical_terms + law_terms + art_terms + education_terms):
            return True
        if domain in ["management", "science"] and any(term in text for term in medical_terms + law_terms + art_terms + education_terms):
            return True
        return False

    def _school_level_filter_sentence(self, level: str, major_name: str) -> str:
        if level == "985":
            return f"985是筛子，但{major_name}要看专业位次和学院资源，别拿学校最低线冒充热门专业线。"
        if level == "211":
            return f"211能过简历初筛，但{major_name}必须有岗位解释，不能只靠牌子硬撑。"
        if level == "双一流":
            return f"双一流看的是学科特色，{major_name}要确认是不是在学校强项上。"
        return "普通家庭最后要落到项目、实习和调剂风险，别被校名带节奏。"

    def _compact_school_reason(self, angle: str, path: str, level_note: str) -> str:
        parts: list[str] = []
        for item in [angle, path, level_note]:
            cleaned = str(item or "").strip().rstrip("。")
            if not cleaned:
                continue
            if any(cleaned in existing or existing in cleaned for existing in parts):
                continue
            parts.append(cleaned)
        return "。".join(parts[:3]) + "。"

    def _family_warning_sentence(self, school: dict, major: dict, school_name: str, major_name: str) -> str:
        name = school_name or school.get("name", "")
        school_type = school.get("type", "")
        level = school.get("level", "")
        city = school.get("city") or school.get("province") or "当地"
        if self._is_digital_engineering_major(major_name):
            if "苏州大学" in name:
                return "普通家庭要把苏州实习半径用起来，重点盯制造业软件、生物医药数字化和园区企业项目，别只喊互联网大厂。"
            if "江南大学" in name:
                return "普通家庭要确认计算机是不是和物联网、食品安全、工业设计这些强场景结合，单纯拼纯互联网名头不如南邮那类院校直接。"
            if "中国矿业大学" in name:
                return "普通家庭要接受它的行业底色，优势是能源安全和工业系统，不是城市消费互联网岗位。"
            if "南京邮电大学" in name:
                return "普通家庭最该看通信、网安、运营商和设备商链条，这类岗位比泛泛写代码更有学校辨识度。"
            if "南京信息工程大学" in name:
                return "普通家庭要看气象、遥感、数据平台这些行业数据岗位，别把它简单当普通计算机院校。"
            if "南通大学" in name:
                return "普通家庭把它当保底时，要盯长三角制造业、医疗信息化和本地企业实习，出口要自己主动做实。"
            if "深圳北理莫斯科大学" in name:
                return "普通家庭要先算清学费、语言环境和读研路径，别只看深圳和中外合作名头。"
            if "香港中文大学（深圳）" in name:
                return "普通家庭要把学费、升学预期和家庭现金流算进来，它不是低成本稳就业路线。"
            if "深圳大学" in name:
                return "普通家庭要主动抢深圳实习和项目，城市机会多，但同城竞争也狠，不能等学校喂饭。"
            if "深圳技术大学" in name:
                return "普通家庭要重点看产教融合项目、企业导师和实训平台，优势在能不能早进工程现场。"
            if "南方科技大学" in name:
                return "普通家庭要确认孩子能不能适应强数理和科研训练，这条路收益高，但不适合只想轻松拿文凭。"
            if "广州大学" in name:
                return "普通家庭要把广州本地实习、考公考编外的企业出口和专业组调剂一起查清楚。"
            if "广东工业大学" in name:
                return "普通家庭要盯制造业数字化、嵌入式、工业互联网这些硬出口，比泛泛写代码更稳。"
            if "华南农业大学" in name:
                return "普通家庭要接受农业食品生命科学交叉底色，别把它当纯互联网院校报。"
            if "西安电子科技大学" in name:
                return "普通家庭要重点查通信、网安、计算机学院资源和校招去向，它的优势是硬科技识别度。"
            if "西安理工大学" in name:
                return "普通家庭要看控制、自动化、制造业项目和实习出口，别只按普通一本名头判断。"
            if "西安科技大学" in name:
                return "普通家庭要接受能源安全和工矿场景，优势不是互联网热闹，而是行业系统落地。"
            if "西安邮电大学" in name:
                return "普通家庭要盯运营商、通信设备、网安和云网融合岗位，别只看计算机四个字。"
            if "陕西科技大学" in name:
                return "普通家庭要核验轻工制造、材料食品和企业数字化项目，确认出口不是空泛软件岗。"
            if "长安大学" in name:
                return "普通家庭要把智能交通、车路协同和工程系统作为主线，不要只拿211标签冲。"
            if "长沙理工大学" in name:
                return "普通家庭要重点看电力、交通、水利行业项目，这类行业入口比泛互联网更稳。"
            if "湖南师范大学" in name:
                return "普通家庭要分清教育技术、考编路径和企业出口，别把师范稳定误读成计算机稳定。"
            if "湖南大学" in name:
                return "普通家庭要核验车辆、电气、智能制造相关交叉项目，确认能借到长沙产业资源。"
            if "中南大学" in name:
                return "普通家庭要看轨道交通、材料、医学数据这些交叉资源，确认孩子能扛985强度。"
            if school_type in ["理工", "工科"]:
                return f"普通家庭要多看{city}校企项目、实验课和实习入口，工程项目比口号更能换饭碗。"
            if school_type == "综合":
                return f"普通家庭可以借{city}综合资源做交叉，但必须自己补项目、竞赛和实习，别等学校替你安排明白。"
        if self._is_humanities_major(major_name):
            if "历史" in major_name:
                if school_type == "师范":
                    return "普通家庭要把教师编、考研、文博档案和地方教育岗位分开看，别只听师范两个字就觉得稳。"
                if level in ["985", "211"]:
                    return "普通家庭看重平台筛选可以理解，但历史学更要核验保研率、师资方向和考编考研出口。"
                return "普通家庭要提前想清楚考研、考编、文博档案和教培之外的出口，别把兴趣当成天然饭碗。"
            if "法学" in major_name:
                return "普通家庭要把法考、考公、律所实习和读研压力提前算清楚，别只看专业听起来体面。"
            if "汉语言" in major_name or "中文" in major_name:
                return "普通家庭要看师范属性、考编岗位、写作能力和新媒体实习，别只把中文理解成背书。"
            return "普通家庭要把考研、考编、实习作品和城市岗位一起核验，文史社科不能只靠学校名头。"
        if level == "211":
            return "普通家庭拿211标签有筛选价值，但要防止专业组里被调到自己完全不接受的方向。"
        if school_type == "师范":
            return "普通家庭要把教师、考编、读研和非教师岗位分清楚，别把稳定两个字想得太简单。"
        return "普通家庭要核对培养方案、实习出口和调剂专业，别只看名字顺眼。"

    def _unique_school_checkpoint(self, school_name: str, major_name: str) -> str:
        """给未覆盖到的学校一个稳定且不同的核验重点，避免逐校理由同质化。"""
        if self._is_humanities_major(major_name):
            humanities_checkpoints = [
                f"这所学校优先核验{major_name}所在学院、师资方向和培养方案，别只看学校牌子。",
                f"这所学校优先核验保研率、考研去向和升学支持，判断是否适合继续深造。",
                f"这所学校优先核验教师编、文博档案、公务员等出口，看路径是不是清楚。",
                f"这所学校优先核验教育实习、博物馆/档案馆实践和城市公共文化资源。",
                f"这所学校优先核验专业组内可调剂专业，防止被调到完全不想读的方向。",
                f"这所学校优先核验本科招生网的课程设置和选科要求，确认不是名字像、培养不对口。",
            ]
            index = sum(ord(ch) for ch in f"{school_name}-{major_name}") % len(humanities_checkpoints)
            return humanities_checkpoints[index]
        checkpoints = [
            f"这所学校优先核验{major_name}所在学院和核心课程，别只看招生大类名字。",
            f"这所学校优先核验实验室、竞赛和项目制课程，看学生能不能做出可展示作品。",
            f"这所学校优先核验校企合作和实习城市半径，判断毕业前能不能接触真实岗位。",
            f"这所学校优先核验近两年就业质量报告里的行业去向，看是不是和{major_name}对口。",
            f"这所学校优先核验专业组内可调剂专业，防止冲进去后被调到完全不想读的方向。",
            f"这所学校优先核验本科招生网的培养方案和选科要求，确认不是名字像、课程不对口。",
        ]
        index = sum(ord(ch) for ch in f"{school_name}-{major_name}") % len(checkpoints)
        return checkpoints[index]

    def _probability_basis_text(self, school: dict, school_name: str, major_name: str, probability: int, applicant_province: str | None = None) -> str:
        province = applicant_province or school.get("province", "")
        query = f"{province} 2025 {school_name} {major_name} 专业最低分 录取分数线".strip()
        return (
            f"模拟估计值{probability}%：由当前分数/位次画像、冲稳保规则和学校层次粗排生成；"
            f"联网核验优先搜“{query}”，按当前专业最低录取分口径判断，再看省考试院投档表和学校招生网。"
        )

    def _salary_basis_text(self, school: dict, school_name: str, major_name: str, salary: int | None) -> str:
        salary_text = self._format_salary(salary)
        query = f"{school_name} {major_name} 就业质量报告 薪资 毕业生去向"
        return (
            f"模拟估计值{salary_text}：由本地专业薪资库、学校就业质量报告检索口径和城市行业机会综合估算；"
            f"联网核验优先搜“{query}”。"
        )

    def _estimate_plan_salary(self, school: dict, major: dict, fallback: int | None = None) -> int | None:
        major_salary = fallback or major.get("salary_median_5yr")
        school_salary = school.get("average_salary")
        if major_salary and school_salary:
            # 专业薪资决定主线，学校所在城市/就业质量做校正，避免同专业所有学校显示完全一致。
            return int(major_salary * 0.72 + school_salary * 0.28)
        return major_salary or school_salary

    def _school_distinctive_angle(self, school: dict, school_name: str, major_name: str) -> str:
        name = school_name or school.get("name", "这所学校")
        school_type = school.get("type", "")
        level = school.get("level", "")
        city = school.get("city") or school.get("province") or "所在城市"

        if self._is_humanities_major(major_name):
            return self._humanities_school_angle(school, name, major_name)
        if self._is_chemistry_major(major_name):
            return self._chemistry_school_angle(school, name, major_name)
        if self._is_economics_major(major_name):
            return self._economics_school_angle(school, name, major_name)

        domain = self._major_domain(major_name)
        if domain not in ["general", "digital"]:
            return self._domain_school_angle(school, name, major_name, domain)

        if self._is_digital_engineering_major(major_name):
            named_angles = [
            ("北京交通大学", "交通运输、轨道交通、系统工程和信息通信底色很强，计算机要往智慧交通、轨道信号、交通大数据和调度系统靠"),
            ("北京化工大学", "化工、材料、过程装备和安全工程是主场，计算机要落到流程控制、工业软件、材料数据和生产安全信息化"),
            ("北京工业大学", "北京地方工科平台，优势在城市建设、智能制造、软件工程和本地政企项目，计算机要借首都产业半径做实习"),
            ("华北电力大学", "电力系统、新能源、电网调度和能源互联网标签很直，计算机要往电力信息化、调度算法、工业控制和能源数据平台走"),
            ("中国矿业大学（北京）", "矿业安全、能源系统和地下工程底色突出，计算机要看矿山智能化、安全生产系统和工业互联网"),
            ("中国地质大学（北京）", "地质、资源、测绘遥感和地学数据是强项，计算机要往地理信息、遥感解译、资源勘探数据平台靠"),
            ("北京林业大学", "林业生态、园林、环境和碳汇监测是特色，计算机要看生态数据、遥感监测、智慧林业和城市绿地信息化"),
            ("北方工业大学", "北京应用型工科底色，优势在自动化、电子信息、智能制造和城市工程项目，计算机要靠项目课和本地实习撑出口"),
            ("北京信息科技大学", "信息、通信、自动化和仪器类应用底色更明显，计算机要看网络安全、智能感知、工业软件和北京政企信息化"),
            ("中国社会科学院大学", "社科平台强，计算机不是主标签；如果报数字方向，要往数据治理、社会计算、政务数据和政策研究工具靠"),
            ("北京物资学院", "物流、供应链、采购和流通经济标签更直，计算机要往供应链系统、物流算法、仓储数字化和企业信息系统走"),
            ("北京建筑大学", "建筑、土木、测绘和城市更新是底色，计算机要看智慧建造、BIM、城市空间数据和工程管理平台"),
            ("华东理工大学", "化工、材料和工程产业底色很强，计算机最好往工业软件、过程控制、智能制造和科研数据平台上靠"),
            ("东华大学", "纺织材料、设计制造和供应链场景更突出，计算机方向适合往智能制造、服装供应链、材料数据化上找交叉出口"),
            ("上海大学", "上海综合平台和211标签更好解释，优势在城市资源、跨学院选择面和本地企业实习半径"),
            ("上海海洋大学", "双一流不是211，海洋、水产、食品和环境特色更明显，计算机要看海洋信息、数据监测、智慧海洋这类交叉方向"),
            ("上海理工大学", "工科和工程应用底色强，计算机更适合往智能制造、医疗器械信息化、企业系统和工程软件靠"),
            ("上海工程技术大学", "应用型工程和产教融合色彩更重，适合把计算机落到制造业信息化、轨道交通和现场工程系统"),
            ("中国农业大学", "农学、食品、生物和农业工程资源强，计算机要往智慧农业、食品安全数据、农业装备智能化上解释"),
            ("中央民族大学", "综合平台加民族事务、公共治理和数据管理场景更突出，计算机要看政务信息化、数据治理和跨文化服务场景"),
            ("青海大学", "西部211和高原能源、生态、医学资源绑定更深，计算机适合看能源数据、生态监测和区域公共服务数字化"),
            ("宁夏大学", "西部211平台和区域产业结合度高，计算机要往能源化工、农业水利、地方政务和企业数字化上找落点"),
            ("新疆大学", "区域中心平台和能源、材料、信息安全场景更明显，计算机方向适合看能源数字化、网络安全和边疆治理信息化"),
            ("石河子大学", "兵团背景和农林医工交叉特色强，计算机要往农业信息化、公共卫生数据和区域企业系统上靠"),
            ("湖北大学", "湖大是湖北省属综合平台，不是硬工科王牌；计算机要借教育信息化、传媒数据、政企数字化和武汉本地实习补项目"),
            ("武汉理工大学", "材料、交通、汽车和船海工程底色很强，计算机要往智能网联汽车、交通系统、工业软件和材料数据平台靠"),
            ("中国地质大学（武汉）", "地质、资源、遥感和地理信息标签鲜明，计算机要看GIS、遥感智能解译、资源数据平台和地学AI"),
            ("华中农业大学", "农业、生物、食品和生命科学资源强，计算机要落到智慧农业、生物数据、食品安全追溯和科研数据平台"),
            ("湖北工业大学", "工程应用和制造业数字化底色更明显，计算机要看工业软件、智能制造、嵌入式和湖北本地企业项目"),
            ("武汉工程大学", "化工、材料、过程装备和安全工程是底盘，计算机要往流程控制、工业互联网、生产系统和安全信息化靠"),
            ("中南民族大学", "民族事务、公共治理和综合平台色彩更重，计算机要看政务数据、公共服务平台、数据治理和武汉实习半径"),
            ("三峡大学", "电力、水利、三峡工程和区域能源场景更直，计算机要往电力信息化、水利调度、工程运维系统上找出口"),
            ("长江大学", "石油地质、农学和地方工科底色更明显，计算机要看油气数据、农业信息化、地理信息和企业系统"),
            ("江汉大学", "武汉本地综合平台，优势在城市产业半径和应用型项目，计算机要主动接软件园、政企信息化和本地企业实习"),
            ("武汉纺织大学", "纺织服装、材料设计和供应链场景是底色，计算机要往智能制造、服装供应链、工业视觉和电商系统靠"),
            ("武汉轻工大学", "食品、粮油、轻工制造和生命健康场景更清楚，计算机要看食品安全追溯、流程控制、企业数字化和质量管理系统"),
            ("湖北医药学院", "医学和医院场景是主线，计算机要往医疗信息化、影像数据、智慧医院和健康数据平台靠，不是纯互联网路线"),
            ("武汉科技大学", "钢铁冶金、材料和制造业场景突出，计算机最好往工业互联网、智能制造、企业系统和生产数据分析靠"),
            ("山东科技大学", "传统工科和资源安全类底色更重，学计算机要往矿山智能化、工业互联网、生产系统数字化上找落点"),
            ("青岛科技大学", "化工、材料和制造业背景更明显，计算机方向更适合切到工业软件、流程控制、企业信息化"),
            ("山东理工大学", "车辆、机械、电气这类工科场景更近，计算机最好往智能制造、车联网、嵌入式和企业系统靠"),
            ("山东建筑大学", "建筑土木和城市建设标签强，计算机要看智慧建造、BIM、城市数据、工程管理系统这些交叉出口"),
            ("中国石油大学", "能源、油气和传统工科行业联系更强，计算机要盯工业软件、能源数字化、自动化系统这类落地场景"),
            ("石油", "能源、油气和传统工科行业联系更强，学计算机要盯工业软件、能源数字化、自动化系统这类落地场景"),
            ("西安电子科技大学", "电子信息、通信、网络安全和计算机底色很强，优势是行业认可度和硬科技岗位，不是泛泛综合平台"),
            ("西安理工大学", "水利水电、装备制造、自动化和控制工程底色更明显，计算机要往工业控制、智能制造和工程软件上落"),
            ("西安科技大学", "矿业安全、应急技术和工科应用场景突出，计算机要看安全生产信息化、工业互联网和能源系统"),
            ("西安邮电大学", "通信、电子信息、网络工程和运营商生态更贴近，适合盯通信软件、网安、云网融合和ICT岗位"),
            ("陕西科技大学", "轻工、材料、食品和包装工程底色明显，计算机要往智能制造、工业软件、流程控制和质量追溯上靠"),
            ("长安大学", "交通运输、车辆、道路桥梁和工程管理底色强，计算机要看智能交通、车路协同、交通大数据和工程系统"),
            ("长沙理工大学", "交通、电力、水利和土木工程底盘更实，计算机/电子信息要往电力信息化、智慧交通和工程系统上靠"),
            ("湖南师范大学", "师范和文理平台强，计算机方向要看教育技术、教育数据、师范资源之外的企业出口"),
            ("湖南大学", "车辆工程、电气、土木和工商管理底色突出，计算机要借长沙智能制造、车联网和工程管理场景"),
            ("中南大学", "冶金、材料、轨道交通和医学资源强，计算机要往工业智能、医学数据、轨道交通系统和材料计算交叉"),
            ("科技大学", "工科氛围更浓，适合把计算机往工程项目、制造业数字化、校企实践上靠"),
            ("理工", "工科训练和实验实践更实，适合不追虚名、愿意靠项目能力吃饭的路线"),
            ("建筑", "土木建筑和城市建设底色明显，计算机方向要优先看智慧建造、BIM、城市数据等交叉出口"),
            ("交通", "交通运输、车辆、物流和工程系统场景更集中，计算机最好往交通智能化、运维系统靠"),
            ("海洋", "海洋、水产和环境特色明显，计算机要看海洋信息、数据监测、智慧海洋这类交叉方向"),
            ("财经", "财经金融资源更集中，计算机方向更适合往金融科技、数据分析、风控系统走"),
            ("会计金融", "金融会计标签强，计算机不是传统王牌时，要把金融科技和数据岗位作为解释路径"),
            ("师范", "师范资源和稳定岗位路径更明显，如果读计算机，要看教育技术、信息化和考编之外的企业出口"),
            ("青岛大学", "青岛城市平台和综合大学资源更占优势，适合看本地互联网、制造业信息化和医疗教育等综合场景"),
            ("济南大学", "省会综合大学的优势是稳定和选择面，适合把省会实习、考公考编和本地企业就业一起算"),
            ("苏州大学", "强点不只是211牌子，而是苏州城市产业半径：生物医药、纳米材料、先进制造、软件服务都能给计算机找交叉场景"),
            ("江南大学", "食品、轻工、设计和物联网底色很重，计算机要往工业软件、食品安全追溯、智能制造和物联网工程上解释出口"),
            ("中国矿业大学", "矿业、安全、能源和地下空间工程底色强，计算机适合往矿山智能化、工业互联网、能源数据和安全生产系统上靠"),
            ("南京邮电大学", "通信、电子信息和网络安全标签鲜明，计算机方向更适合看通信软件、网络安全、云网融合和运营商/设备商生态"),
            ("南京信息工程大学", "气象、遥感、地理信息和数据科学特色突出，计算机要看气象大数据、遥感智能解译、城市安全和行业数据平台"),
            ("南京工业大学", "化工、材料、安全工程和制造业底盘强，计算机最好落到工业软件、流程控制、智能制造和安全生产信息化"),
            ("江苏大学", "车辆、机械、电气、农业装备这类工科场景更近，计算机要往智能制造、车联网、嵌入式和企业数字化上靠"),
            ("扬州大学", "综合性和师范农学底色都有，优势是省内认可度和稳定出口，计算机要主动找教育信息化、农业数据和地方企业数字化场景"),
            ("南京师范大学", "师范和文科平台强，计算机如果不是学校最强标签，就要重点看教育技术、数据治理、考编外企业出口和南京实习资源"),
            ("南通大学", "地方综合大学叠加医学、师范和长三角制造业场景，计算机要靠本地产业实习、医疗信息化和企业系统落地"),
            ("深圳北理莫斯科大学", "中外合作和理工交叉色彩更重，适合能接受英文/俄式数理训练、想走科研数据、交叉工程和国际化路径的学生"),
            ("香港中文大学（深圳）", "港中深的强项在国际化培养、商科数据、AI交叉和深圳产业链接，学费和培养节奏要单独算清楚"),
            ("深圳大学", "深圳本地综合平台和城市产业资源强，优势在互联网、电子信息、金融科技和本地实习半径，不是靠传统985/211标签筛人"),
            ("深圳技术大学", "应用技术和产教融合标签更明显，计算机要往智能制造、工业软件、机器人和企业工程项目上落"),
            ("南方科技大学", "新型研究型大学底色强，数理基础、科研训练和深圳硬科技资源突出，更适合能扛强度、愿意走科研或硬科技路线的学生"),
            ("广州大学", "广州综合平台和城市公共服务、建筑土木、教育资源更明显，计算机要往城市治理、智慧建造、本地企业数字化上找出口"),
            ("广东工业大学", "广东制造业工程底盘强，计算机和电子信息最好接智能制造、工业互联网、嵌入式和校企工程项目"),
            ("华南农业大学", "农业、生命科学和食品资源强，计算机要落到智慧农业、食品安全追溯、生物数据和装备智能化"),
            ("广州医科大学", "医学场景突出，计算机和电子信息要看医疗信息化、影像数据、智慧医院和健康数据平台"),
            ("西安电子科技大学", "电子信息、通信、网络安全和计算机底色很强，优势是行业认可度和硬科技岗位，不是泛泛综合平台"),
            ("西安理工大学", "水利水电、装备制造、自动化和控制工程底色更明显，计算机要往工业控制、智能制造和工程软件上落"),
            ("西安科技大学", "矿业安全、应急技术和工科应用场景突出，计算机要看安全生产信息化、工业互联网和能源系统"),
            ("西安邮电大学", "通信、电子信息、网络工程和运营商生态更贴近，适合盯通信软件、网安、云网融合和ICT岗位"),
            ("陕西科技大学", "轻工、材料、食品和包装工程底色明显，计算机要往智能制造、工业软件、流程控制和质量追溯上靠"),
            ("长安大学", "交通运输、车辆、道路桥梁和工程管理底色强，计算机要看智能交通、车路协同、交通大数据和工程系统"),
            ("长沙理工大学", "交通、电力、水利和土木工程底盘更实，计算机/电子信息要往电力信息化、智慧交通和工程系统上靠"),
            ("湖南师范大学", "师范和文理平台强，计算机方向要看教育技术、教育数据、师范资源之外的企业出口"),
            ("湖南大学", "车辆工程、电气、土木和工商管理底色突出，计算机要借长沙智能制造、车联网和工程管理场景"),
            ("中南大学", "冶金、材料、轨道交通和医学资源强，计算机要往工业智能、医学数据、轨道交通系统和材料计算交叉"),
            ]
            for key, value in named_angles:
                if key in name:
                    return value

            if "农业" in name:
                return "农林食品和生命科学场景更集中，计算机要靠智慧农业、数据平台和装备智能化讲清出口"
            if "民族" in name:
                return "公共治理和多元文化服务场景更明显，计算机要往政务数据、信息系统和公共服务数字化解释"
            if "科技" in name:
                return "工科应用场景更密，适合把计算机落到工程项目、制造业数字化和企业系统里"

        if "理工" in name:
            return f"理工训练更看实践，{major_name}要重点核验实验课、专业平台和行业去向，不能只靠校名判断"
        if level == "985":
            return f"平台筛选和校友资源强，但必须确认{major_name}是不是该校有资源支撑的方向"
        if level == "211":
            return f"{city}的211标签对简历初筛有用，但要看这个专业能不能接上当地产业和实习资源"
        if level == "双一流":
            return "双一流价值在学科特色，不等同于211，必须核对该校强势学科和招生专业组"
        if school_type in ["工科", "理工"]:
            return f"{city}的工科院校底盘能支撑项目训练，关键看该专业是不是放在核心学院"
        if school_type == "综合":
            return self._generic_comprehensive_angle(name, city, major_name)
        if school_type:
            return f"{school_type}类院校的行业色彩明显，适合把专业和行业出口绑在一起看"
        return "学校层次、城市和专业组需要一起核验，不能只按校名冷热判断"

    def _major_path_sentence(self, major: dict, major_name: str) -> str:
        tags = major.get("tags", [])
        category = major.get("category", "")
        if self._is_chemistry_major(major_name):
            return f"{major_name}要看实验平台、试剂分析、材料/化工/环境方向和读研去向，不能套用计算机项目作品逻辑。"
        if self._is_economics_major(major_name):
            return f"{major_name}要看微观宏观、计量经济、统计工具、实习半径和读研/考公/银行券商等路径，不能套用理工项目逻辑。"
        domain = self._major_domain(major_name)
        if domain != "general":
            return self._domain_major_path_sentence(major_name, domain)
        if self._is_digital_engineering_major(major_name) and ("技术壁垒" in tags or category == "工学"):
            return f"{major_name}未来不会缺人，但缺的是能写项目、懂业务、能进工程现场的人，低端重复开发会越来越卷。"
        if category == "工学":
            return f"{major_name}更看实验、工程训练、行业场景和安全规范，关键是确认培养方案能不能接真实岗位。"
        if major.get("requires_grad_school"):
            return f"{major_name}本科直接就业的确定性不够，读研和继续深造要提前放进成本表。"
        if "看背景" in tags:
            return f"{major_name}分化比较大，普通家庭要靠实习、证书和城市资源补背景。"
        return f"{major_name}要看课程方向和实习出口，别只看专业名字热不热。"

    def _generic_comprehensive_angle(self, school_name: str, city: str, major_name: str) -> str:
        templates = [
            f"{city}综合大学的价值不在单点王牌，而在城市资源和转向空间；{major_name}要靠项目、实习和学院资源把出口做实",
            f"{school_name}这种综合平台适合做交叉，但普通家庭别把“选择面宽”当成就业确定性，关键看{major_name}有没有真实项目入口",
            f"{city}本地资源能给{major_name}提供实习半径，但学校不是替你安排饭碗的，得看学院课程、企业合作和毕业去向",
            f"{school_name}的优势是平台弹性，短板是专业标签不一定够硬；报{major_name}要把竞赛、作品和实习提前压上去",
        ]
        index = sum(ord(ch) for ch in f"{school_name}-{major_name}-{city}") % len(templates)
        return templates[index]

    def _generic_comprehensive_path(self, school_name: str, city: str, major_name: str) -> str:
        templates = [
            f"这条路要借{city}本地企业、政企信息化和校内交叉资源，别只等课堂喂饭，项目作品才是普通家庭的筛子。",
            f"这条路适合把{major_name}和学校强势学院做交叉，能落到真实业务系统就值，落不到就容易变成泛泛学代码。",
            f"这条路要优先查学院培养方案、实验课和就业质量报告，看普通毕业生是进企业系统、考研，还是被迫转行。",
            f"这条路不是靠综合大学四个字吃饭，得用竞赛、实习、开源作品或行业项目把{major_name}讲清楚。",
        ]
        index = sum(ord(ch) for ch in f"{city}-{school_name}-{major_name}") % len(templates)
        return templates[index]

    def _school_major_path_sentence(self, school: dict, school_name: str, major: dict, major_name: str) -> str:
        name = school_name or school.get("name", "")
        city = school.get("city") or school.get("province") or "当地"
        if self._is_chemistry_major(major_name):
            return self._chemistry_major_path_sentence(school, name, major_name)
        if self._is_economics_major(major_name):
            return self._economics_major_path_sentence(school, name, major_name)
        domain = self._major_domain(major_name)
        if domain not in ["general", "digital"]:
            return self._domain_school_major_path_sentence(school, name, major_name, domain)
        if self._is_digital_engineering_major(major_name):
            if "北京交通大学" in name:
                return "这条路要盯轨道交通信号、智慧交通平台、调度算法和交通大数据，学校标签和岗位解释是连着的。"
            if "北京化工大学" in name:
                return "这条路别按纯互联网理解，要接化工流程控制、工业软件、材料计算和安全生产系统，场景越工业越对口。"
            if "北京工业大学" in name:
                return "这条路要用北京本地软件、智能制造、城市治理和政企信息化项目，地方工科平台靠实习半径吃饭。"
            if "华北电力大学" in name:
                return "这条路要往电网调度、新能源系统、能源互联网和工业控制靠，电力行业标签比泛泛写代码更值钱。"
            if "中国矿业大学（北京）" in name:
                return "这条路要接矿山智能化、安全生产、能源数据和工业互联网，接受行业场景就有辨识度。"
            if "中国地质大学（北京）" in name:
                return "这条路要看GIS、遥感解译、资源勘探数据和地学AI，别把它当普通计算机强校。"
            if "北京林业大学" in name:
                return "这条路要和生态监测、遥感数据、智慧林业、碳汇管理结合；如果只想互联网大厂，它不是最顺手的牌。"
            if "北方工业大学" in name:
                return "这条路要靠自动化、电子信息、智能制造和北京企业实习做出口，别只看学校名气。"
            if "北京信息科技大学" in name:
                return "这条路要往网络安全、智能感知、工业软件和政企信息化靠，优势是信息类应用场景比较直。"
            if "中国社会科学院大学" in name:
                return "这条路更像数据治理、社会计算、政策研究工具和政务数据平台，不是传统工程师培养主线。"
            if "北京物资学院" in name:
                return "这条路要切供应链系统、物流算法、仓储数字化和企业ERP，学校行业标签窄，但窄也能讲清饭碗。"
            if "北京建筑大学" in name:
                return "这条路要往BIM、智慧建造、城市空间数据和工程管理系统靠，建筑土木场景才是它的抓手。"
            if "苏州大学" in name:
                return "这条路要借苏州工业园区和长三角企业密度，把课程项目往软件工程、制造业数字化、生物医药数据平台上压。"
            if "江南大学" in name:
                return "这条路不要只和纯互联网比，要看物联网、食品质量追溯、工业设计软件和智能制造系统这些交叉出口。"
            if "中国矿业大学" in name:
                return "这条路更适合工业现场和能源安全系统，矿山智能化、生产调度、工业互联网比泛泛做前端后端更有辨识度。"
            if "南京邮电大学" in name:
                return "这条路要顺着通信和电子信息底盘走，网络安全、通信软件、云网融合和运营商设备商链条是它的优势。"
            if "南京信息工程大学" in name:
                return "这条路要和气象、遥感、地理信息、城市安全数据结合，行业数据平台比普通互联网叙事更贴学校。"
            if "南通大学" in name:
                return "这条路要靠长三角制造业、医疗信息化和地方企业系统做落点，适合把保底做成可就业的保底。"
            if "深圳北理莫斯科大学" in name:
                return "这条路要看数理基础、国际化课程和科研数据训练，适合把计算机做成交叉工程能力，不是普通应用开发叙事。"
            if "香港中文大学（深圳）" in name:
                return "这条路要借港中深的数据科学、AI交叉和深圳产业链接，重点看升学、科研项目和高质量实习。"
            if "深圳大学" in name:
                return "这条路要吃深圳城市红利，重点看互联网产品、金融科技、电子信息企业和本地实习转化。"
            if "深圳技术大学" in name:
                return "这条路要落到应用工程，智能制造、机器人、工业软件和企业项目比理论名头更重要。"
            if "南方科技大学" in name:
                return "这条路要靠强数理、科研训练和硬科技项目拉开差距，适合往AI、芯片软件、科研平台走。"
            if "广州大学" in name:
                return "这条路要和广州城市治理、智慧建造、教育信息化和本地企业数字化结合，不能只讲综合大学选择面。"
            if "广东工业大学" in name:
                return "这条路要顺着广东制造业走，工业互联网、嵌入式、自动化系统和工程软件是更清楚的饭碗。"
            if "华南农业大学" in name:
                return "这条路别只盯大厂，要看智慧农业、食品安全追溯、生物数据平台和农业装备控制。"
            if "西安电子科技大学" in name:
                return "这条路要顺着电子信息和网安底盘走，通信软件、网络安全、芯片软件和军工电子生态是辨识度。"
            if "西安理工大学" in name:
                return "这条路要落到控制工程、工业软件、智能装备和制造业数字化，项目经历比专业名更关键。"
            if "西安科技大学" in name:
                return "这条路要看矿山安全、应急管理、能源生产系统和工业互联网，行业场景比互联网叙事更清楚。"
            if "西安邮电大学" in name:
                return "这条路要盯通信网络、运营商、云网融合和网络安全，ICT链条比普通软件外包更值得看。"
            if "陕西科技大学" in name:
                return "这条路要和轻工制造、材料食品、质量追溯和流程控制结合，别只按通用计算机理解。"
            if "长安大学" in name:
                return "这条路要看智能交通、车路协同、交通大数据和工程管理系统，交通行业是它的解释路径。"
            if "长沙理工大学" in name:
                return "这条路要往电力信息化、智慧交通、水利工程系统和工程软件靠，行业项目是核心卖点。"
            if "湖南师范大学" in name:
                return "这条路要和教育技术、教育数据平台、师范资源和长沙企业实习结合，不能只讲综合选择面。"
            if "湖南大学" in name:
                return "这条路要接车辆、电气、智能制造和车联网场景，长沙产业链比单纯互联网更关键。"
            if "中南大学" in name:
                return "这条路要借轨道交通、材料计算、医学数据和工业智能资源，适合走硬科技交叉。"
            if "武汉理工大学" in name:
                return "这条路要接智能网联汽车、交通运输系统、工业软件和材料数据平台，别只按普通计算机院校理解。"
            if "中国地质大学（武汉）" in name:
                return "这条路要看GIS、遥感解译、资源勘探数据和地学AI，地学场景就是它的岗位解释。"
            if "华中农业大学" in name:
                return "这条路要和智慧农业、生物信息、食品安全追溯、农业装备数据结合，高分报农校必须讲清行业场景。"
            if "湖北工业大学" in name:
                return "这条路要往工业软件、智能制造、嵌入式和本地企业项目靠，能进工程现场比泛泛写代码重要。"
            if "武汉工程大学" in name:
                return "这条路要接化工流程控制、生产数据平台、安全工程信息化和工业互联网，场景越落地越值。"
            if "中南民族大学" in name:
                return "这条路要避免只喊计算机，重点看政务信息系统、公共服务平台、数据治理和武汉实习资源。"
            if "三峡大学" in name:
                return "这条路要盯水利电力调度、能源数据平台、工程运维和电网相关信息系统，行业越清楚越有价值。"
            if "长江大学" in name:
                return "这条路要接油气勘探数据、地理信息、农业信息化和地方企业系统，别按纯大厂软件岗理解。"
            if "江汉大学" in name:
                return "这条路要吃武汉本地软件、政企数字化和企业实习半径，保底也得保出项目和实习。"
            if "武汉纺织大学" in name:
                return "这条路要接智能纺织、工业视觉、服装供应链和电商系统，行业场景窄一点，但讲得清。"
            if "武汉轻工大学" in name:
                return "这条路要看食品安全追溯、粮油加工数字化、流程控制和质量管理系统，别只喊互联网。"
            if "湖北医药学院" in name:
                return "这条路要往医院信息系统、医学影像数据、健康管理平台和药械信息化靠，接受医疗场景才顺。"
            if "农业" in name:
                return "这条路别只盯互联网大厂，更应该看智慧农业、食品安全追溯、农业装备控制和科研数据平台。"
            if "民族" in name:
                return "这条路要避免只喊技术名词，重点看政务信息系统、公共服务平台、数据治理和考公考编外的企业出口。"
            if "石油" in name or "能源" in name:
                return "这条路适合往能源数字化、油气生产系统、工业软件和自动化运维上靠，行业场景比纯互联网更重要。"
            if "海洋" in name:
                return "这条路要看海洋观测、环境监测、智慧渔业和数据平台，优势是交叉场景，不是传统互联网名头。"
            if "理工" in name or "工程" in name or "科技" in name:
                return "这条路要多看项目制课程、实验室、校企合作和制造业数字化岗位，别只看专业名字。"
            if "大学" in name and school.get("type") == "综合":
                return self._generic_comprehensive_path(name, city, major_name)
            return "这条路核心看项目经历、算法/开发基础和行业场景，低端重复开发会越来越卷。"
        return self._major_path_sentence(major, major_name)

    def _economics_major_path_sentence(self, school: dict, school_name: str, major_name: str) -> str:
        name = school_name or school.get("name", "")
        city = school.get("city") or school.get("province") or "当地"
        if "财经" in name or school.get("type") == "财经政法":
            return "这条路要盯经管学院资源、金融财政统计课程、校友实习半径和读研去向，财经标签有用但不能替代岗位能力。"
        if "农业" in name or school.get("type") == "农林海洋":
            return "这条路更适合看农业经济、食品产业链、农村发展和产业政策研究，别把它当成泛金融热门专业。"
        if "理工" in name or school.get("type") in ["工科", "理工"]:
            return "这条路要借产业经济、供应链、企业管理和区域发展场景解释出口，不能只靠学校工科牌子。"
        if school.get("type") == "师范":
            return "这条路要把读研、公共经济、教育经济、考公考编和非教师岗位分开看，别把师范稳定简单套到经济学上。"
        return f"这条路要用{city}实习资源、统计计量训练、读研去向和财经类岗位入口来验证，普通家庭尤其要看中位数路径。"

    def _major_domain(self, major_name: str) -> str:
        primary = (major_name or "").split("/")[0]
        major = self.major_by_name.get(primary, {})
        category = major.get("category", "")
        if any(key in primary for key in ["机械", "电气", "土木", "建筑", "车辆", "航空", "能源", "材料", "集成电路", "生物医学工程"]):
            return "traditional_engineering"
        if self._is_digital_engineering_major(primary):
            return "digital"
        if self._is_chemistry_major(primary):
            return "chemistry"
        if self._is_economics_major(primary):
            return "economics"
        if any(key in primary for key in ["临床", "口腔", "医学", "药学", "护理", "中医"]):
            return "medical"
        if any(key in primary for key in ["法学", "知识产权", "社会学", "政治"]):
            return "law_social"
        if any(key in primary for key in ["教育", "学前", "小学", "师范"]):
            return "education"
        if any(key in primary for key in ["农学", "动物医学", "园艺", "植物", "种子"]):
            return "agriculture"
        if any(key in primary for key in ["会计", "工商管理", "工程管理", "人力资源", "电子商务", "物流"]):
            return "management"
        if any(key in primary for key in ["数学", "物理", "生物", "地理", "统计", "心理"]):
            return "science"
        if any(key in primary for key in ["自动化"]):
            return "traditional_engineering"
        if any(key in primary for key in ["艺术", "设计", "美术", "音乐", "数字媒体"]):
            return "art"
        if category == "医学":
            return "medical"
        if category == "法学":
            return "law_social"
        if category == "教育学":
            return "education"
        if category == "农学":
            return "agriculture"
        if category == "管理学":
            return "management"
        if category == "理学":
            return "science"
        if category == "艺术学":
            return "art"
        if category == "工学":
            return "traditional_engineering"
        return "general"

    def _domain_school_angle(self, school: dict, school_name: str, major_name: str, domain: str) -> str:
        name = school_name or school.get("name", "这所学校")
        school_type = school.get("type", "")
        level = school.get("level", "")
        city = school.get("city") or school.get("province") or "所在城市"
        if domain == "medical":
            if school_type == "医药" or any(key in name for key in ["医科", "医学", "中医"]):
                return f"{name}的医学资源更集中，{major_name}要看附属医院、临床/药学实践平台、规培或考证路径。"
            return f"{name}不是典型医药院校，报{major_name}必须核验医学院、附属医院和培养资质，不能只看学校层次。"
        if domain == "law_social":
            if "政法" in name or school_type == "财经政法":
                return f"{name}的政法资源更直接，{major_name}要看法考支持、实务课程、法院律所和公检法实习半径。"
            return f"{name}报{major_name}要借平台和城市资源，但核心仍是法考/实习/读研去向，不能只拿校名过筛。"
        if domain == "education":
            if school_type == "师范" or "师范" in name:
                return f"{name}的师范培养和地方教育系统认可度更明确，{major_name}要看教资、实习学校、考编和升学路径。"
            return f"{name}报{major_name}要确认教育学院资源和实习基地，非师范平台不能自动等同教师出口。"
        if domain == "agriculture":
            if school_type == "农林海洋" or "农业" in name:
                return f"{name}的农林生命和产业链资源更集中，{major_name}要看实验基地、农业企业、基层岗位和读研方向。"
            return f"{name}报{major_name}要核验农学院资源和实践基地，不能把农学简单理解成普通理科。"
        if domain == "management":
            if school_type == "财经政法" or "财经" in name:
                return f"{name}的财经管理资源更集中，{major_name}要看实习城市、行业证书、校友网络和企业招聘入口。"
            if school_type in ["工科", "理工"]:
                return f"{name}的工科产业背景能支撑{major_name}，但要落到工程管理、供应链、制造业管理或企业运营场景。"
            return f"{name}报{major_name}要看商学院资源、实习半径、证书路径和本地企业认可度。"
        if domain == "science":
            return f"{name}报{major_name}重点看学科平台、实验/数理训练、保研考研和科研方向，不能只按就业热度判断。"
        if domain == "traditional_engineering":
            if school_type in ["工科", "理工"] or any(key in name for key in ["理工", "工业", "工程", "科技"]):
                return f"{name}的工程训练和实验平台更匹配{major_name}，要看实验课、校企项目、行业场景和安全规范。"
            return f"{name}报{major_name}要确认工程学院资源、实验条件和实习企业，综合平台不能替代硬训练。"
        if domain == "art":
            if school_type == "语言艺术" or any(key in name for key in ["艺术", "美术", "音乐", "传媒"]):
                return f"{name}的艺术/传媒资源更直接，{major_name}要看作品集训练、行业实习、展演平台和城市机会。"
            return f"{name}报{major_name}要核验学院资源、作品集训练和就业场景，不能只看综合大学名头。"
        if level in ["985", "211", "双一流"]:
            return f"{name}的平台有筛选价值，但{major_name}必须单独核验学院资源、专业组和就业去向。"
        return f"{city}平台和学校类型需要结合{major_name}的真实出口一起看，不能只按校名冷热判断。"

    def _domain_major_path_sentence(self, major_name: str, domain: str) -> str:
        templates = {
            "medical": f"{major_name}要看培养资质、附属医院、实习轮转、规培/执业资格和读研压力，普通家庭要把周期成本算清楚。",
            "law_social": f"{major_name}要看法考或实务训练、律所/法院/基层治理实习、考公和读研去向，不是背书就能换饭碗。",
            "education": f"{major_name}要看教资、教育实习、地方考编政策、读研和非教师岗位，不能只听“稳定”两个字。",
            "agriculture": f"{major_name}要看实验实践、农业企业、基层技术服务、读研和产业链岗位，接受行业场景才有出口。",
            "management": f"{major_name}要看实习半径、证书路径、企业运营场景和校友网络，管理类最怕只学概念没有岗位证据。",
            "science": f"{major_name}要看数理/实验训练、保研考研、科研方向和跨行业应用，本科直接就业要谨慎核验。",
            "traditional_engineering": f"{major_name}要看实验课、工程训练、行业规范、校企项目和实习现场，工程类靠真训练吃饭。",
            "art": f"{major_name}要看作品集、展演/项目机会、城市行业资源和实习入口，天赋和投入周期都要提前评估。",
        }
        return templates.get(domain, f"{major_name}要看课程方向、实习出口和培养方案，别只看名字热不热。")

    def _domain_school_major_path_sentence(self, school: dict, school_name: str, major_name: str, domain: str) -> str:
        name = school_name or school.get("name", "")
        city = school.get("city") or school.get("province") or "当地"
        if domain == "medical":
            return f"这条路要优先核验{name}的附属医院、临床/药学实践、规培去向和执业资格通过支持。"
        if domain == "law_social":
            return f"这条路要用{city}的法院律所、公共部门、基层治理和读研资源解释出口，别只看学校牌子。"
        if domain == "education":
            return f"这条路要把{city}教育实习、教资、考编、读研和非教师岗位都算进去，稳定不是自动发放的。"
        if domain == "agriculture":
            return f"这条路要看{name}的实验基地、农林企业、产业链实践和读研去向，接受行业场景才顺。"
        if domain == "management":
            return f"这条路要盯{city}实习、企业项目、证书路径和校友招聘入口，管理类不能只靠课堂概念。"
        if domain == "science":
            return f"这条路要看数理/实验平台、导师方向、保研考研和跨行业应用，先把深造预期说清楚。"
        if domain == "traditional_engineering":
            return f"这条路要看实验课、工程训练、校企项目和真实行业现场，能不能动手比专业名头更关键。"
        if domain == "art":
            return f"这条路要看作品集训练、展演项目、城市行业资源和实习机会，投入和回报要提前算。"
        return self._domain_major_path_sentence(major_name, domain)

    def _is_digital_engineering_major(self, major_name: str) -> bool:
        return any(key in major_name for key in ["计算机", "软件", "人工智能", "电子信息", "通信", "自动化", "数据科学", "物联网", "信息安全"])

    def _is_chemistry_major(self, major_name: str) -> bool:
        return any(key in major_name for key in ["化学", "化工", "应用化学", "化学工程"])

    def _is_economics_major(self, major_name: str) -> bool:
        return any(key in major_name for key in ["经济", "金融", "财政", "税收", "贸易", "投资", "保险"])

    def _is_humanities_major(self, major_name: str) -> bool:
        return any(key in major_name for key in ["历史", "汉语言", "中文", "法学", "哲学", "社会学", "新闻", "传播", "政治", "教育学"])

    def _economics_school_angle(self, school: dict, school_name: str, major_name: str) -> str:
        name = school_name or school.get("name", "这所学校")
        school_type = school.get("type", "")
        level = school.get("level", "")
        city = school.get("city") or school.get("province") or "所在城市"
        if "中南财经政法大学" in name:
            return "财经政法资源集中，经济方向要重点看应用经济学、财政金融、统计计量和实习资源，别只看学校名字好听"
        if "武汉大学" in name:
            return "综合985平台筛选价值强，经济方向要核验经管学院资源、保研去向和武汉本地实习半径"
        if "华中科技大学" in name:
            return "综合985加产业资源有上限，经济方向更适合往产业经济、区域经济、企业管理和继续深造上核验"
        if "华中师范大学" in name:
            return "师范文科平台稳定，经济方向要看公共经济、教育经济、读研和体制内路径，不能只拿211牌子硬撑"
        if "武汉理工大学" in name:
            return "理工平台和交通材料产业资源强，经济方向要看产业经济、供应链管理、企业管理和武汉实习资源"
        if "湖北大学" in name:
            return "省属综合平台更看本地认可度，经济方向要靠课程、实习、考研和区域经济资源把出口做实"
        if "农业" in name or school_type == "农林海洋":
            return f"{name}的优势不在泛财经，而在农业经济、产业链、食品贸易和乡村发展等场景，报{major_name}要把行业出口想清楚"
        if "民族" in name or school_type == "民族":
            return f"{name}适合把{major_name}和公共治理、区域发展、民族地区经济服务结合起来看，重点核验培养方案和就业去向"
        if school_type == "财经政法":
            return f"{name}的财经政法标签对{major_name}更对口，但也要看实习门槛、读研比例和具体学院资源"
        if school_type == "师范":
            return f"{name}的文科和师范资源能支撑{major_name}，但要重点看是否偏公共经济、教育经济、考研和体制内路径"
        if school_type in ["工科", "理工"]:
            return f"{city}的理工平台可以借产业资源，但{major_name}必须落到产业经济、区域经济、企业经营或供应链场景，别按纯工科逻辑理解"
        if level in ["985", "211", "双一流"]:
            return f"{name}的平台有筛选价值，但{major_name}要单独核验经管学院资源、实习半径、读研比例和专业组调剂风险"
        return f"{name}报{major_name}要看课程结构、实习城市、考研去向和财经类岗位入口，不能只看专业名字热不热"

    def _chemistry_school_angle(self, school: dict, school_name: str, major_name: str) -> str:
        name = school_name or school.get("name", "这所学校")
        school_type = school.get("type", "")
        level = school.get("level", "")
        city = school.get("city") or school.get("province") or "所在城市"
        if "武汉理工大学" in name:
            return "武汉理工的材料、交通、汽车和船海工程底色强，化学方向要重点看材料化学、应用化学、化工安全和实验平台，不要套用其他专业的就业叙事"
        if "中国地质大学" in name:
            return "中国地质大学的地质、资源、环境和材料分析场景更突出，化学方向要看地球化学、环境检测、材料测试和实验室资源"
        if "华中农业大学" in name:
            return "华中农业大学的农业、生物、食品和生命科学资源强，化学方向要看农药、食品检测、生物化学和分析测试平台"
        if "武汉工程大学" in name:
            return "武汉工程大学化工底盘更直接，化学/化工要重点看化工工艺、材料化工、安全工程和产业实习"
        if "理工" in name or school_type in ["工科", "理工"]:
            return f"{name}的工科实验训练更重要，{major_name}要核验实验室、化工/材料平台、安全规范和就业质量报告"
        if "农业" in name or school_type == "农林海洋":
            return f"{name}的农林生命科学资源更集中，{major_name}要往食品检测、农化、生物化学和环境分析方向核验"
        if level in ["985", "211", "双一流"]:
            return f"{name}的平台有筛选价值，但{major_name}必须单独核验学院资源、实验条件、保研率和专业组调剂风险"
        return f"{name}报{major_name}要看实验课程、导师方向、读研去向和{city}化工/材料/检测行业岗位半径"

    def _chemistry_major_path_sentence(self, school: dict, school_name: str, major_name: str) -> str:
        name = school_name or school.get("name", "")
        if "化学工程" in major_name or "化工" in major_name:
            return "这条路要看化工原理、反应工程、分离工程、过程安全和企业实习，核心是实验能力、工程化能力和安全规范。"
        if "历史" in major_name:
            return self._major_path_sentence(school, major_name)
        if "农业" in name or "华中农业" in name:
            return "这条路要往食品安全检测、农化分析、生物化学和实验平台靠，适合能接受实验室与读研路径的学生。"
        if "地质" in name:
            return "这条路要看地球化学、环境监测、资源材料分析和实验室训练，别把学校的地学标签误解成计算机路线。"
        return "这条路要优先核验实验平台、课程体系、读研比例和化工/材料/检测行业去向，普通家庭不能只看学校牌子。"

    def _humanities_school_angle(self, school: dict, school_name: str, major_name: str) -> str:
        name = school_name or school.get("name", "这所学校")
        school_type = school.get("type", "")
        level = school.get("level", "")
        city = school.get("city") or school.get("province") or "所在城市"

        if "历史" in major_name:
            if "南京师范大学" in name:
                return "南京师范大学的优势在师范文科平台、南京教育资源和省内教师岗位认可度，历史学要重点看师范方向、保研和考编出口"
            if "江苏师范大学" in name:
                return "江苏师范大学更适合把历史学和中学教师、地方教育系统、考研考编路径绑定起来看，不能只按城市热度判断"
            if "河海大学" in name:
                return "河海大学的强项不在历史学本身，优势更多是211平台筛选；报历史要确认学院资源和转向读研的可行性"
            if "苏州大学" in name:
                return "苏州大学的文科平台和苏州城市资源有加成，历史学要看文博、地方文化机构、师范/考研通道是否清楚"
            if "南京大学" in name:
                return "南京大学历史学平台强，但对分数和学术能力要求高，更适合能接受深造和学术训练的学生"
            if "师范" in name or school_type == "师范":
                return f"{name}的历史学要优先看师范培养、教师编制、教育实习和考研去向，稳定来自路径设计，不是来自专业名字"
            if level in ["985", "211"]:
                return f"{name}的主要价值是平台和筛选，历史学要额外核验学科实力、保研率、文博档案和考编出口"
            return f"{name}报历史学要看{city}本地教育、文博、档案和公务员岗位半径，学校名气只是第一层"
        if "法学" in major_name:
            return f"{name}报法学要看法学学科资源、法考通过支持、律所实习和{city}政法机关岗位，别只看学校层次"
        if "汉语言" in major_name or "中文" in major_name:
            return f"{name}报中文要看师范属性、写作训练、考编岗位和{city}媒体出版/教育资源，出口要提前设计"
        if "新闻" in major_name or "传播" in major_name:
            return f"{name}报新闻传播要看城市媒体资源、实习作品和数据传播能力，不能只靠课堂内容找工作"
        return f"{name}报{major_name}要看学科平台、考研考编、实习作品和{city}岗位半径，文史社科更吃路径规划"

    def _risk_sentence(self, risk_level: str, probability: int) -> str:
        if risk_level == "冲":
            return f"冲档，粗排参考{probability}%，它的任务是抬上限，不是当主心骨。"
        if risk_level == "保":
            return f"保档，粗排参考{probability}%，它的任务是防滑档，别嫌它名字没那么响。"
        return f"稳档，粗排参考{probability}%，它的任务是守住学校、城市、专业三件事的平衡。"

    def _score_verify_sentence(self, school: dict, school_name: str, major_name: str, applicant_province: str | None = None) -> str:
        province = applicant_province or school.get("province", "")
        query_hint = f"{province} 2025 {school_name} {major_name} 专业最低分 录取分数线".strip()
        admissions = self._school_admissions_entry_url(school, school_name)
        if admissions:
            return f"核验分数时优先搜“{query_hint}”，按当前专业最低分判断，再进本科招生网（{admissions}）或省考试院看专业组。"
        return f"核验分数时优先搜“{query_hint}”，按当前专业最低分判断，不要只看第三方榜单。"

    def _shorten_for_chat(self, text: str, limit: int) -> str:
        text = re.sub(r"\s+", " ", text or "").strip("；。 ")
        if len(text) <= limit:
            return text
        cut = text[:limit].rstrip("；，, ")
        return f"{cut}……"

    def _risk_role_label(self, risk_level: str) -> tuple[str, str]:
        if risk_level == "冲":
            return "冲刺", "低"
        if risk_level == "保":
            return "保底", "高"
        return "稳妥", "中"

    def _build_sync_reason(
        self,
        school: dict,
        major: dict,
        user: UserPreferences | None,
        risk_level: str,
        fallback_strategy: str = "",
        fallback_reason: str = "",
    ) -> str:
        """Generate a short, readable reason for the sync/audit desk."""
        school_name = school.get("name", "这所学校")
        major_name = major.get("name", "该专业")
        province = self._normalize_region_name(user.province) if user and user.province else "本省"
        family = user.family_background if user and user.family_background else "当前家庭"
        city = school.get("city") or school.get("province") or "目标城市"
        school_level = school.get("level") or "待核验层次"
        risk_role, chance_label = self._risk_role_label(risk_level)

        if fallback_strategy:
            lead = (
                f"这是原始条件过窄后的「{fallback_strategy}」方案，先把它作为{risk_role}候选同步，"
                f"用于审核是否比“0候选”更可执行。"
            )
        else:
            lead = (
                f"同步这所学校，是因为它在当前画像下属于{risk_role}候选，"
                f"录取把握先按「{chance_label}」档审核。"
            )

        body = (
            f"{school_name}位于{city}，本地库标记为{school_level}；本轮重点看{major_name}是否有明确招生计划、"
            f"近三年在{province}的专业最低位次，以及专业组里是否混入不想去的专业。"
        )
        family_guard = (
            f"对{family}来说，这张卡不是最终结论，而是提醒你先查官方表，再决定是否放进正式志愿表。"
        )
        if fallback_reason:
            family_guard = f"{family_guard} 替代原因：{fallback_reason}"
        return " ".join([lead, body, family_guard])

    def _official_verification_links(
        self,
        school: dict,
        school_name: str,
        province: str | None,
    ) -> list[dict]:
        links: list[dict] = []
        exam_site = self._province_exam_site(province)
        if province and exam_site:
            links.append(
                {
                    "label": f"{province}教育考试院",
                    "type": "省考试院",
                    "url": f"https://{exam_site}/",
                    "desc": "查近三年投档表、招生录取政策和官方公告。",
                }
            )
        links.append(
            {
                "label": "阳光高考",
                "type": "教育部平台",
                "url": "https://gaokao.chsi.com.cn/",
                "desc": "核对招生章程、选科要求、专业组和院校基本信息。",
            }
        )
        admissions_url = self._school_admissions_entry_url(school, school_name)
        if admissions_url:
            links.append(
                {
                    "label": f"{school_name}本科招生网",
                    "type": "学校招生网",
                    "url": admissions_url,
                    "desc": "查招生计划、专业最低分、专业组和调剂规则。",
                }
            )
        official_url = school.get("official_url")
        if official_url and official_url.rstrip("/") != (admissions_url or "").rstrip("/"):
            links.append(
                {
                    "label": f"{school_name}官网",
                    "type": "学校官网",
                    "url": official_url,
                    "desc": "查学院设置、培养方案、就业质量报告入口。",
                }
            )
        return links[:4]

    def _required_verification_tables(
        self,
        user: UserPreferences | None,
        school_name: str,
        major_name: str,
    ) -> list[str]:
        province = self._normalize_region_name(user.province) if user and user.province else "本省"
        subject = user.subjects if user and user.subjects else "对应科类"
        return [
            f"{province}教育考试院：近三年本科批/普通批投档表，先看{subject}位次。",
            f"{school_name}本科招生网：当年招生计划，确认{major_name}是否在目标省份招生。",
            f"{school_name}本科招生网：近三年{province}{major_name}专业最低分和最低位次。",
            "阳光高考或招生章程：专业组包含专业、选科要求、调剂规则和转专业规则。",
        ]

    def _missing_key_data(
        self,
        school: dict,
        user: UserPreferences | None,
        school_name: str,
        major_name: str,
        citations: list[str],
    ) -> list[str]:
        province = self._normalize_region_name(user.province) if user and user.province else "本省"
        missing: list[str] = []
        if not (user and user.rank):
            missing.append("考生位次未确认，冲稳保档位只能粗排。")
        if not self._school_admissions_entry_url(school, school_name):
            missing.append(f"缺{school_name}本科招生网入口，需要人工核验官网。")
        missing.extend(
            [
                f"缺近三年{province}{major_name}专业最低分/最低位次的结构化记录。",
                "缺当年招生计划人数、专业组包含专业和调剂规则的官方确认。",
                f"缺{major_name}所在学院培养方案和就业质量报告的逐项核对。",
            ]
        )
        if not citations:
            missing.append("本轮未拿到联网检索结果，当前只按本地规则和入口提示展示。")
        return missing[:5]

    def _build_sync_audit_bundle(
        self,
        school: dict,
        major: dict,
        user: UserPreferences | None,
        risk_level: str,
        citations: list[str],
    ) -> dict:
        school_name = school.get("name", "该校")
        major_name = major.get("name", "该专业")
        province = self._normalize_region_name(user.province) if user and user.province else None
        official_links = self._official_verification_links(school, school_name, province)
        has_school_admissions = any(item.get("type") == "学校招生网" for item in official_links)
        missing = self._missing_key_data(school, user, school_name, major_name, citations)
        risk_role, chance_label = self._risk_role_label(risk_level)
        return {
            "evidence_status": "入口已匹配，关键表待核验" if has_school_admissions else "缺学校招生网入口",
            "official_verification": official_links,
            "required_tables": self._required_verification_tables(user, school_name, major_name),
            "missing_key_data": missing,
            "audit_notes": [
                f"录取概率只展示为{chance_label}，对应当前{risk_role}档，不代表官方录取概率。",
                "模拟中位薪资只用于比较专业出口，不能替代学校就业质量报告。",
                "先查官方表，再决定这所学校是保留、降档还是删除。",
            ],
        }

    def _build_structured_recommendations(
        self,
        recommend: RecommendResponse,
        user: UserPreferences,
        citations: list[str],
    ) -> list[ConsultRecommendationPlan]:
        structured: list[ConsultRecommendationPlan] = []
        for plan in recommend.plans[:10]:
            school = self.school_by_name.get(plan.school, {})
            major = self.major_by_name.get(plan.major, {})
            school = {**school, "name": plan.school}
            major = {**major, "name": plan.major}
            salary = self._estimate_plan_salary(school, major, plan.median_salary_5yr)
            school_level = school.get("level", "院校")
            city = school.get("city") or school.get("province") or "目标城市"
            tier = school.get("tier", "")
            school_type = school.get("type", "")
            major_category = major.get("category", "")
            requires_grad = "需要重点考虑考研/深造路径" if major.get("requires_grad_school") else "本科就业出口相对更直接"
            irreplaceability = plan.irreplaceability or major.get("irreplaceability")
            barrier_text = (
                f"技术壁垒/被替代风险估算{irreplaceability}/100"
                if irreplaceability is not None
                else "技术壁垒需要结合培养方案再核验"
            )
            overview = (
                f"{plan.school}是{school_level}层次，位于{tier}{city}，"
                f"本轮推荐报考专业为{plan.major}"
                f"{f'（{major_category}）' if major_category else ''}。"
                f"{requires_grad}，{barrier_text}。"
            )
            reason = self._build_plan_specific_reason(
                school=school,
                major=major,
                user=user,
                risk_level=plan.risk_level,
                probability=plan.probability,
                fallback=plan.reason,
            )
            if plan.fallback_strategy:
                reason = f"替代路径【{plan.fallback_strategy}】：{plan.fallback_reason}；{reason}"
            family_risk = build_family_risk_profile(school, major, user.family_background if user else None, plan.risk_level)
            if plan.risk_tags:
                family_risk["risk_tags"] = plan.risk_tags
            if plan.family_strategy:
                family_risk["family_strategy"] = plan.family_strategy
            if plan.family_risk_summary:
                family_risk["family_risk_summary"] = plan.family_risk_summary
            applicant_province = self._normalize_region_name(user.province) if user and user.province else None
            sync_reason = self._build_sync_reason(
                school=school,
                major=major,
                user=user,
                risk_level=plan.risk_level,
                fallback_strategy=plan.fallback_strategy,
                fallback_reason=plan.fallback_reason,
            )
            audit_bundle = self._build_sync_audit_bundle(
                school=school,
                major=major,
                user=user,
                risk_level=plan.risk_level,
                citations=citations,
            )
            structured.append(
                ConsultRecommendationPlan(
                    order=plan.order,
                    risk_level=plan.risk_level,
                    school=plan.school,
                    major=plan.major,
                    school_level=school_level,
                    overview=overview,
                    recommendation_reason=reason,
                    probability=plan.probability,
                    median_salary_5yr=salary,
                    median_salary_display=self._format_salary(salary),
                    irreplaceability=irreplaceability,
                    probability_basis=self._probability_basis_text(
                        school,
                        plan.school,
                        plan.major,
                        plan.probability,
                        applicant_province=applicant_province,
                    ),
                    salary_basis=self._salary_basis_text(school, plan.school, plan.major, salary),
                    data_basis=(
                        "录取概率和中位数薪资均为模拟估计值；联网检索只按2025年录取分数线、当前专业最低分口径核验。"
                        "最终以省教育考试院、阳光高考和学校本科招生网为准。"
                    ),
                    admissions_url=self._school_admissions_entry_url(school),
                    admissions_query=self._school_admissions_query(applicant_province, plan.school, plan.major),
                    risk_tags=family_risk["risk_tags"],
                    family_strategy=family_risk["family_strategy"],
                    family_risk_summary=family_risk["family_risk_summary"],
                    fallback_strategy=plan.fallback_strategy,
                    fallback_reason=plan.fallback_reason,
                    sync_reason=sync_reason,
                    evidence_status=audit_bundle["evidence_status"],
                    official_verification=audit_bundle["official_verification"],
                    required_tables=audit_bundle["required_tables"],
                    missing_key_data=audit_bundle["missing_key_data"],
                    audit_notes=audit_bundle["audit_notes"],
                    citations=citations[:5],
                )
            )
        return structured

    def _build_plan_specific_reason(
        self,
        school: dict,
        major: dict,
        user: UserPreferences | None,
        risk_level: str,
        probability: int,
        fallback: str = "",
    ) -> str:
        school_name = school.get("name", "该校")
        major_name = major.get("name", "该专业")
        school_type = school.get("type", "")
        city = school.get("city") or school.get("province") or "目标城市"
        tier = school.get("tier", "")
        salary = major.get("salary_median_5yr")
        irreplaceability = major.get("irreplaceability")
        family_risk = build_family_risk_profile(
            school,
            major,
            user.family_background if user else None,
            risk_level,
        )
        risk_tags = family_risk.get("risk_tags") or []

        parts = [
            self._risk_sentence(risk_level, probability),
            f"学校差异点：{self._school_distinctive_angle(school, school_name, major_name)}",
            f"普通家庭落点：{self._family_warning_sentence(school, major, school_name, major_name)}",
            f"家庭风险标签：{'、'.join(risk_tags[:4]) if risk_tags else '暂无明显结构性风险'}",
            f"家庭分流建议：{family_risk.get('family_strategy', '')}",
            f"逐校核验点：{self._unique_school_checkpoint(school_name, major_name)}",
            self._specific_major_value(major, user),
        ]

        if salary:
            parts.append(f"{major_name}普通毕业生几年后收入按本地专业库估算约{salary // 1000}K，不能当官方工资")
        if irreplaceability is not None:
            parts.append(f"技术壁垒/被替代风险估算{irreplaceability}/100，用来看方向风险，不替代实习和培养方案核验")
        if user and user.family_background:
            parts.append(f"已按{user.family_background}的试错成本处理，避免只看学校名头忽略落地就业")
        if user and user.city_preference:
            city_hit = city in user.city_preference or school.get("province") in user.city_preference or tier in user.city_preference
            parts.append("城市偏好匹配，后续实习和就业半径更顺" if city_hit else f"{city}不完全等于目标城市，需确认是否能接受地域机会成本")
        if school_type and school_type != "综合":
            parts.append(f"{school_type}类院校要核对{major_name}是不是该校强项，不要只按校名报")
        parts.append(self._future_trend_text(major, school))
        applicant_province = self._normalize_region_name(user.province) if user and user.province else None
        parts.append(self._score_verify_sentence(school, school_name, major_name, applicant_province=applicant_province))

        return "；".join(part for part in parts if part)

    def _specific_school_value(self, school: dict, major: dict) -> str:
        level = school.get("level", "")
        school_type = school.get("type", "")
        city = school.get("city") or school.get("province") or "所在城市"
        major_category = major.get("category", "")
        if level == "985":
            text = "985平台更适合拿学历上限和校友资源"
        elif level == "211":
            text = "211平台对简历初筛有现实帮助"
        elif level == "双一流":
            text = "双一流不是211，价值要看学科和行业特色，不能再按211口径误判"
        elif level == "普通一本":
            text = "普通一本要靠专业强度、城市机会和录取确定性取胜"
        else:
            text = f"{level or '院校'}层次需要结合当年专业组位次核验"

        if school_type in ["理工", "工科"] and major_category == "工学":
            text += "，理工底色与工科专业匹配度更高"
        elif school_type in ["财经", "财经政法"] and major_category in ["经济学", "管理学", "法学"]:
            text += "，财经政法资源对口但实习门槛也更明显"
        elif school_type == "师范":
            text += "，师范资源更适合教师、考编和稳定岗位路径"
        elif school_type in ["农林", "海洋"]:
            text += "，行业特色明显，适合接受垂直赛道的人"
        elif school_type == "医药":
            text += "，医药路径要把资格证、规培和深造周期算进去"

        return f"{text}；{city}的城市资源会影响实习、校招和就业半径"

    def _specific_major_value(self, major: dict, user: UserPreferences | None) -> str:
        major_name = major.get("name", "该专业")
        tags = major.get("tags", [])
        if major.get("requires_grad_school"):
            return f"{major_name}本科出口不一定够硬，适合愿意读研或继续深造的考生"
        if "看背景" in tags:
            return f"{major_name}就业分化较大，普通家庭要看实习、证书和城市资源能不能补上"
        if "天坑" in tags:
            return f"{major_name}行业周期和转化路径要谨慎，不适合只凭兴趣硬上"
        if "技术壁垒" in tags or (major.get("irreplaceability") or 0) >= 75:
            return f"{major_name}更看项目能力、实验/工程训练和持续学习，学深了才有壁垒"
        if (major.get("employment_rate") or 0) >= 0.88:
            return f"{major_name}出口相对直接，适合先追求就业确定性"
        return f"{major_name}需要看具体培养方向和课程结构，不能只按专业大类想象就业"

    def _future_trend_text(self, major: dict, school: dict) -> str:
        major_name = major.get("name", "该专业")
        category = major.get("category", "")
        tags = major.get("tags", [])
        school_type = school.get("type", "")
        if category == "工学" or "技术壁垒" in tags:
            return f"未来趋势：{major_name}低端重复岗位会被自动化压缩，工程实践、项目经验和复合能力会越来越值钱"
        if category in ["经济学", "管理学", "法学"]:
            return f"未来趋势：{major_name}会更依赖城市平台、实习质量和证书，资源差距会放大"
        if category == "医学" or school_type == "医药":
            return "未来趋势：医疗健康需求长期存在，但培养周期和准入门槛不会降低"
        if category == "教育学" or school_type == "师范":
            return "未来趋势：教师岗位更看地区人口、学科缺口和编制供给，稳定但竞争更细"
        return f"未来趋势：{major_name}要看行业周期和个人能力积累，不建议只按当下冷热做决定"

    def _extract_recommendations_from_answer(
        self,
        answer: str,
        user: UserPreferences | None,
        citations: list[str],
    ) -> list[ConsultRecommendationPlan]:
        if not answer:
            return []

        school_names = sorted(self.school_names, key=len, reverse=True)
        structured: list[ConsultRecommendationPlan] = []
        seen: set[str] = set()
        current_risk = ""

        for raw_line in answer.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            current_risk = self._update_risk_context(line, current_risk)

            if not any(name in line for name in school_names):
                continue
            if line.startswith(("数据口径", "本次联网来源", "继续追问", "红旗风险", "你下一步")):
                continue

            fragments = re.split(r"[，、；;。]|\s+和\s+|和(?=[\u4e00-\u9fa5A-Za-z（）()·]{2,}(?:大学|学院))", line)
            for fragment in fragments:
                fragment = fragment.strip(" -·\t")
                if not fragment:
                    continue
                current_risk = self._update_risk_context(fragment, current_risk)
                school_name = next((name for name in school_names if name in fragment), "")
                if not school_name:
                    continue
                key = school_name
                if key in seen:
                    continue
                seen.add(key)

                risk_level = self._infer_risk_level(fragment, current_risk)
                major_name = self._infer_recommended_major(fragment, school_name, user)
                probability = self._infer_probability(fragment, risk_level)
                plan = self._build_answer_recommendation_plan(
                    order=len(structured) + 1,
                    risk_level=risk_level,
                    school_name=school_name,
                    major_name=major_name,
                    probability=probability,
                    user=user,
                    citations=citations,
                )
                structured.append(plan)
                if len(structured) >= 10:
                    return structured

        return structured

    def _update_risk_context(self, text: str, current_risk: str) -> str:
        if re.search(r"(第[一1]层|冲|冲一冲|冲刺|冲档)", text):
            return "冲"
        if re.search(r"(第[二2]层|稳|稳阵|稳住|稳妥|稳档)", text):
            return "稳"
        if re.search(r"(第[三3]层|保|保底|保档|兜底)", text):
            return "保"
        return current_risk

    def _infer_risk_level(self, text: str, current_risk: str) -> str:
        if re.search(r"(冲|冲一冲|冲刺|冲档)", text):
            return "冲"
        if re.search(r"(保|保底|保档|安全垫)", text):
            return "保"
        if re.search(r"(稳|稳阵|稳住|稳妥|稳档)", text):
            return "稳"
        return current_risk or "稳"

    def _infer_probability(self, text: str, risk_level: str) -> int:
        match = re.search(r"(\d{2})\s*%", text)
        if match:
            value = int(match.group(1))
            return max(35, min(98, value))
        return {"冲": 66, "稳": 88, "保": 96}.get(risk_level, 84)

    def _infer_recommended_major(self, fragment: str, school_name: str, user: UserPreferences | None) -> str:
        after_school = fragment.split(school_name, 1)[-1]
        for alias, major in MAJOR_ALIASES.items():
            if alias in after_school or (alias in fragment and alias not in school_name):
                return major
        for major in self.major_names:
            if major in after_school:
                return major
        if user and user.major_preference:
            return user.major_preference[0]
        return "专业待核验"

    def _build_answer_recommendation_plan(
        self,
        order: int,
        risk_level: str,
        school_name: str,
        major_name: str,
        probability: int,
        user: UserPreferences | None,
        citations: list[str],
    ) -> ConsultRecommendationPlan:
        school = self.school_by_name.get(school_name, {})
        major = self.major_by_name.get(major_name, {})
        school = {**school, "name": school_name}
        major = {**major, "name": major_name}
        salary = self._estimate_plan_salary(school, major, major.get("salary_median_5yr"))
        irreplaceability = major.get("irreplaceability")
        school_level = school.get("level", "院校")
        city = school.get("city") or school.get("province") or "目标城市"
        tier = school.get("tier", "")
        major_category = major.get("category", "")
        overview = (
            f"{school_name}是{school_level}层次，位于{tier}{city}；"
            f"本轮推荐报考专业为{major_name}"
            f"{f'（{major_category}）' if major_category else ''}。"
        )
        reason = self._build_plan_specific_reason(
            school=school,
            major=major,
            user=user,
            risk_level=risk_level,
            probability=probability,
        )
        reason = f"{reason}；最终需核验当年招生计划、专业组和投档位次"
        family_risk = build_family_risk_profile(
            school,
            major,
            user.family_background if user else None,
            risk_level,
        )
        sync_reason = self._build_sync_reason(
            school=school,
            major=major,
            user=user,
            risk_level=risk_level,
        )
        audit_bundle = self._build_sync_audit_bundle(
            school=school,
            major=major,
            user=user,
            risk_level=risk_level,
            citations=citations,
        )
        return ConsultRecommendationPlan(
            order=order,
            risk_level=risk_level,
            school=school_name,
            major=major_name,
            school_level=school_level,
            overview=overview,
            recommendation_reason=reason,
            probability=probability,
            median_salary_5yr=salary,
            median_salary_display=self._format_salary(salary),
            irreplaceability=irreplaceability,
            probability_basis=self._probability_basis_text(school, school_name, major_name, probability),
            salary_basis=self._salary_basis_text(school, school_name, major_name, salary),
            data_basis="从最终对话回答逐校提取后结构化；概率和中位数薪资为模拟估计值，已按2025年录取分数线、当前专业最低分口径提示核验。",
            admissions_url=self._school_admissions_entry_url(school),
            admissions_query=self._school_admissions_query(user.province if user else None, school_name, major_name),
            risk_tags=family_risk["risk_tags"],
            family_strategy=family_risk["family_strategy"],
            family_risk_summary=family_risk["family_risk_summary"],
            sync_reason=sync_reason,
            evidence_status=audit_bundle["evidence_status"],
            official_verification=audit_bundle["official_verification"],
            required_tables=audit_bundle["required_tables"],
            missing_key_data=audit_bundle["missing_key_data"],
            audit_notes=audit_bundle["audit_notes"],
            citations=citations[:5],
        )

    def _build_insight_context(self, request: ConsultRequest, intent: IntentResult) -> str:
        user = self._build_user_preferences(request, allow_partial=True)
        target_name = (intent.major_names or intent.school_names or [""])[0]
        should_backfill_major = any(marker in request.question for marker in [
            "中位数", "薪资", "工资", "收入",
            "这个专业", "该专业", "本专业", "这个方向", "当前方向", "我的方向",
        ])
        if not target_name and self._is_fact_data_question(request.question) and should_backfill_major:
            target_name = self._fact_target_major(request, intent)
        if not target_name:
            return ""

        target_type = "major" if target_name in intent.major_names else "school"
        if self._is_fact_data_question(request.question) and target_name not in intent.school_names:
            target_type = "major"
        insight = agent_engine.insights(
            request=InsightRequest(
                target_type=target_type,
                target_name=target_name,
                user=user,
            )
        )
        return (
            "Agent洞察结果：\n"
            f"对象：{insight.target}\n"
            f"概览：{insight.overview}\n"
            f"普通毕业生几年后的收入参考：{insight.median_salary or '暂无'}（估算值，仅供方向判断）\n"
            f"就业稳定性参考：{insight.employment_rate or '暂无'}（估算值，仅供方向判断）\n"
            f"技术壁垒/被替代风险：{self._format_irreplaceability(insight.irreplaceability)}\n"
            f"趋势：{insight.trend_analysis}\n"
            f"风险：{'；'.join(insight.risk_factors) if insight.risk_factors else '暂无'}\n"
            "数据口径：以上薪资和就业数据均为本地估算值，非官方精确统计；只能用于方向判断，不能当作真实统计。具体录取分数线和位次请以教育考试院发布的官方数据为准。"
        )

    def _format_irreplaceability(self, value: int | None) -> str:
        if value is None:
            return "暂无"
        if value >= 85:
            return f"{value}/100，壁垒高，越学越值钱，被轻易替代的风险低"
        if value >= 70:
            return f"{value}/100，壁垒中上，得靠项目、证书或学校平台拉开差距"
        if value >= 55:
            return f"{value}/100，壁垒一般，普通家庭要谨慎看就业出口"
        return f"{value}/100，壁垒偏低，容易卷成纯体力竞争"

    def _build_admission_score_research_policy(self) -> str:
        return (
            "联网检索口径（必须在回答中遵守）：\n"
            "1. 分数核验只围绕2025年高考录取分数线，不使用2024年旧分数线替代当前判断。\n"
            "2. 涉及具体院校和专业时，按当前专业的最低录取分/专业最低分口径判断，不拿学校最低投档线冒充热门专业能上。\n"
            "3. 来源优先级为省教育考试院、阳光高考、学校本科招生网；第三方平台只能作为入口，不能作为结论。\n"
            "4. 如果本轮没有查到2025年当前专业最低分，必须明确说待官方核验，不能编造分数。"
        )

    def _build_data_honesty_context(self) -> str:
        return (
            "数据真实性边界（必须在回答中遵守）：\n"
            "1. 本地 majors.json 的 salary_median_5yr、employment_rate、irreplaceability 均标记为 estimate，是经验估算，不是官方真实统计。\n"
            "2. 本地 schools.json 的 average_salary、employment_rate 均标记为 estimate，是经验估算，不是学校官方就业质量报告数据。\n"
            "3. Agent输出的录取概率是规则引擎模拟概率，只能用于冲稳保排序参考，不是真实录取概率。\n"
            "4. 只有 citations 中明确给出的联网来源，才可以称为联网核验或公开来源；没有 citations 时必须说“本地估算/模拟”。\n"
            "5. 回答时不要把 18K、92%、98% 这类数字说成精确真实数据；必须加“估算、左右、模拟、仅供排序参考”。"
        )

    def _build_profile_strategy_context(self, request: ConsultRequest) -> str:
        ctx = request.context
        if not ctx:
            return ""
        notes = []
        family = ctx.family_background or "普通家庭"
        subjects = ctx.subjects or ""
        major_pref = "、".join(ctx.major_preference or [])
        city_pref = "、".join(ctx.city_preference or [])

        notes.append("画像策略上下文：")
        notes.append(f"家庭条件：{family}；风险偏好：{ctx.risk_appetite or '均衡'}。")
        if "普通" in family:
            notes.append("普通家庭优先确定性：保底要足、专业壁垒要清楚、不要为了校名牺牲就业路径。")
        elif "中产" in family:
            notes.append("中产家庭可以有一部分试错，但不能把全部志愿压在高波动专业上。")
        else:
            notes.append("资源较充足时可以更重视热爱和平台，但也要看长期路径。")
        if any(key in subjects for key in ["物", "化", "生"]):
            notes.append("物化生画像适合考虑工科、医药、计算机、电子信息等有技术壁垒方向；跨到低壁垒文商科要说明机会成本。")
        if major_pref:
            notes.append(f"当前专业方向：{major_pref}。回答追问时要说明是否偏离当前方向，以及是否需要重新计算方案。")
        if city_pref:
            notes.append(f"目标地区：{city_pref}。城市资源判断要优先参考这个偏好。")
        notes.append("回答用户开放式问题时，也要给明确判断：该做什么、避开什么、下一步查什么。")
        return "\n".join(notes)

    def _build_user_preferences(self, request: ConsultRequest, allow_partial: bool = False) -> Optional[UserPreferences]:
        ctx = request.context
        province = (ctx.province if ctx else None) or self._extract_province(request.question)
        score = (ctx.score if ctx else None) or self._extract_score(request.question)

        if not province or not score:
            if not allow_partial:
                return None
            province = province or "山东"
            score = score or 600

        question_major_pref = self._extract_major_preference(request.question)
        profile_major_pref = ctx.major_preference if ctx and ctx.major_preference else None
        if question_major_pref and (
            self._is_explicit_current_major_recommendation(request.question, question_major_pref)
            or self._asks_about_major_switch(request.question, question_major_pref, profile_major_pref)
        ):
            selected_major_pref = question_major_pref
        else:
            selected_major_pref = profile_major_pref or question_major_pref
        major_pref = self._expand_major_preferences(selected_major_pref)
        raw_cities = (ctx.city_preference if ctx else None) or self._extract_city_preference(request.question)
        cities = self._expand_region_preferences(raw_cities) if raw_cities else []

        return UserPreferences(
            province=self._normalize_region_name(province),
            score=score,
            rank=(ctx.rank if ctx else None) or self._extract_rank(request.question),
            subjects=ctx.subjects if ctx else None,
            family_background=(ctx.family_background if ctx else None) or "普通家庭",
            city_preference=cities or None,
            major_preference=major_pref or None,
            risk_appetite=(ctx.risk_appetite if ctx else None) or "均衡",
            willing_grad_school=ctx.willing_grad_school if ctx else None,
            allow_military_schools=self._asks_for_military_school(request.question),
        )

    def _extract_province(self, text: str) -> Optional[str]:
        for province in PROVINCES:
            patterns = [
                rf"(?:在|来自|我是|孩子在|考生在){province}(?:省|市|自治区)?",
                rf"{province}(?:省|市|自治区)?考生",
                rf"{province}(?:省|市|自治区)?\s*\d{{3}}\s*分",
                rf"{province}(?:省|市|自治区)?\s*(?:位次|排名|排位)",
            ]
            if any(re.search(pattern, text) for pattern in patterns):
                return province
        for province in PROVINCES:
            if province in text:
                return province
        return None

    def _extract_score(self, text: str) -> Optional[int]:
        match = re.search(r"(\d{3})\s*分", text)
        if match:
            score = int(match.group(1))
            if 100 <= score <= 750:
                return score
        return None

    def _extract_rank(self, text: str) -> Optional[int]:
        match = re.search(r"(?:位次|排名|排位)\D{0,5}([\d,，]{3,})", text)
        if match:
            return int(match.group(1).replace(",", "").replace("，", ""))
        return None

    def _extract_city_preference(self, text: str) -> list[str]:
        cities = []
        for province in PROVINCES:
            if f"去{province}" in text or f"到{province}" in text or f"{province}上学" in text or f"{province}读" in text:
                cities.append(province)
        compact = re.sub(r"\s+", "", text or "")
        for alias in REGION_GROUP_ALIASES:
            if alias in compact and alias not in cities:
                cities.append(alias)
        for city in COMMON_CITY_NAMES:
            if city in compact and city not in cities:
                cities.append(city)
        return cities

    def _asks_out_of_province(self, text: str) -> bool:
        return any(key in text for key in ["外省", "省外", "外地学校", "外地院校", "外地高校"])

    def _asks_about_major_switch(
        self,
        text: str,
        question_major_pref: list[str] | None,
        profile_major_pref: list[str] | None,
    ) -> bool:
        if not question_major_pref:
            return False
        expanded_question = self._expand_major_preferences(question_major_pref)
        expanded_profile = self._expand_major_preferences(profile_major_pref)
        if expanded_profile and set(expanded_question).issubset(set(expanded_profile)):
            return False

        compact = re.sub(r"\s+", "", text or "")
        switch_markers = [
            "改", "换", "转", "另", "其他", "其它", "对比", "比较",
            "电子信息", "计算机", "软件", "人工智能", "通信", "金融", "法学", "医学", "临床", "口腔",
        ]
        recommend_markers = ["推荐", "报", "选", "能上", "怎么样", "好吗", "值不值", "适合", "分析", "看看", "了解", "咨询"]
        major_mentioned = self._major_preference_mentioned(compact, question_major_pref + expanded_question)
        return major_mentioned and (
            any(marker in compact for marker in switch_markers)
            or any(marker in compact for marker in recommend_markers)
        )

    def _major_preference_mentioned(self, compact_text: str, preferences: list[str]) -> bool:
        if any(pref and pref in compact_text for pref in preferences):
            return True
        target_set = set(preferences)
        return any(alias in compact_text and major in target_set for alias, major in MAJOR_ALIASES.items())

    def _is_explicit_current_major_recommendation(self, text: str, question_major_pref: list[str] | None) -> bool:
        if not question_major_pref:
            return False
        compact = re.sub(r"\s+", "", text or "")
        if not self._major_preference_mentioned(compact, question_major_pref + self._expand_major_preferences(question_major_pref)):
            return False
        recommendation_intent = any(
            marker in compact
            for marker in [
                "院校推荐", "学校推荐", "推荐院校", "推荐学校", "有什么院校", "有什么学校",
                "能报", "能上", "报什么", "选什么", "怎么报", "去学", "学", "读", "这个专业",
            ]
        )
        question_shape = any(marker in compact for marker in ["什么", "哪些", "哪", "推荐", "怎么样", "好不好", "适合"])
        return recommendation_intent or question_shape

    def _asks_for_military_school(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        return any(
            key in compact
            for key in ["军校", "军队院校", "部队院校", "国防科技", "国防类", "军医", "陆军", "海军", "空军", "火箭军", "武警"]
        )

    def _extract_major_preference(self, text: str) -> list[str]:
        search_text = self._text_without_school_names(text)
        positioned: list[tuple[int, int, str]] = []
        for major in self.major_names:
            index = search_text.find(major)
            if index >= 0:
                positioned.append((index, -len(major), major))
        for alias, major in MAJOR_ALIASES.items():
            index = search_text.find(alias)
            if index >= 0:
                positioned.append((index, -len(alias), major))
        if "地理历史" in search_text or ("地理" in search_text and "历史" in search_text):
            for major in ["地理科学", "历史学"]:
                index = search_text.find(major[:2])
                positioned.append((index if index >= 0 else len(search_text), -len(major), major))
        matches: list[str] = []
        for _, _, major in sorted(positioned):
            if major not in matches:
                matches.append(major)
        return matches

    def _text_without_school_names(self, text: str) -> str:
        masked = str(text or "")
        compact = re.sub(r"\s+", "", masked)
        for name in sorted(self.school_names, key=len, reverse=True):
            normalized_name = re.sub(r"[()（）]", "", name)
            masked = masked.replace(name, " ")
            if normalized_name and normalized_name in compact:
                masked = masked.replace(normalized_name, " ")
        for alias in sorted(SCHOOL_ALIASES.keys(), key=len, reverse=True):
            masked = masked.replace(alias, " ")
        return masked

    def _expand_major_preferences(self, preferences: Optional[list[str]]) -> list[str]:
        if not preferences:
            return []
        expanded: list[str] = []
        for preference in preferences:
            value = str(preference).strip()
            if not value:
                continue
            if value in self.major_names and value not in expanded:
                expanded.append(value)
                continue
            for item in self._extract_major_preference(value):
                if item not in expanded:
                    expanded.append(item)
        return expanded

    def _normalize_region_name(self, value: str) -> str:
        return re.sub(r"(省|市|自治区|壮族自治区|回族自治区|维吾尔自治区)$", "", str(value).strip())

    def _expand_region_preferences(self, values: list[str] | None) -> list[str]:
        if not values:
            return []
        known_regions = sorted(set(PROVINCES + COMMON_CITY_NAMES), key=len, reverse=True)
        expanded: list[str] = []

        def add_region(region: str) -> None:
            normalized_region = self._normalize_region_name(region)
            if normalized_region and normalized_region not in expanded:
                expanded.append(normalized_region)

        for raw in values:
            value = self._normalize_region_name(raw)
            if not value:
                continue
            pieces = [piece for piece in re.split(r"[、,，\s/]+", value) if piece]
            for piece in pieces:
                normalized_piece = self._normalize_region_name(piece)
                alias_hits = [
                    (normalized_piece.find(alias), alias)
                    for alias in REGION_GROUP_ALIASES
                    if alias in normalized_piece
                ]
                if alias_hits:
                    for _, alias in sorted(alias_hits, key=lambda item: item[0]):
                        for region in REGION_GROUP_ALIASES[alias]:
                            add_region(region)
                    continue
                hits = [
                    (normalized_piece.find(region), region)
                    for region in known_regions
                    if region in normalized_piece and normalized_piece != region
                ]
                if hits:
                    for _, region in sorted(hits, key=lambda item: item[0]):
                        add_region(region)
                    continue
                add_region(normalized_piece)
        return expanded


consult_orchestrator = ConsultOrchestrator()
