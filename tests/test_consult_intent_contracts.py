import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from core import consult_orchestrator as orchestrator_module
from core.consult_orchestrator import consult_orchestrator
from core.models import ConsultRequest, UserProfile
from core.session_manager import session_manager
from main import app


PROFILE = UserProfile(
    province="河南",
    score=542,
    rank=102542,
    subjects="政史地",
    family_background="普通家庭",
    major_preference=["历史政治"],
    risk_appetite="均衡",
)

SHANGHAI_HISTORY_PROFILE = UserProfile(
    province="上海",
    score=534,
    rank=15744,
    subjects="政史地",
    family_background="普通家庭",
    city_preference=["上海"],
    major_preference=["历史"],
    risk_appetite="均衡",
)


OFF_TOPIC_RECOMMENDATION = (
    "[院校推荐]\n"
    "1. A大学：先做冲稳保。\n"
    "2. B大学：再看保底。\n\n"
    "[灵魂追问]\n"
    "- 能不能接受保底？"
)

OFF_TOPIC_SCHOOL_NAME_PROMPT = (
    "[分析过程]\n"
    "1. 画像拆解：河南，542分，政史地。\n\n"
    "[核心判断]\n"
    "别先问哪个学校名字好听，先问这条路能不能换饭碗。"
)


def make_request(question: str, context=PROFILE) -> ConsultRequest:
    return ConsultRequest(question=question, context=context)


def routed(question: str):
    request = make_request(question)
    enriched = consult_orchestrator._enrich_request_context(request)
    return consult_orchestrator._detect_intent(enriched)


class ConsultIntentContractsTest(unittest.TestCase):
    def test_intent_router_keeps_common_consults_in_their_lanes(self):
        cases = [
            {
                "question": "这个专业中位数收入多少",
                "intent": "insight",
                "major": "历史学",
                "school": None,
            },
            {
                "question": "500强去哪些学校招聘",
                "intent": "insight",
                "major": None,
                "school": None,
            },
            {
                "question": "这个专业能进500强吗",
                "intent": "insight",
                "major": "历史学",
                "school": None,
            },
            {
                "question": "华北电力大学能上吗",
                "intent": "school_chance",
                "major": None,
                "school": "华北电力大学",
            },
            {
                "question": "历史学怎么样",
                "intent": "insight",
                "major": "历史学",
                "school": None,
            },
            {
                "question": "河南542分历史学推荐哪些学校",
                "intent": "recommend",
                "major": "历史学",
                "school": None,
            },
            {
                "question": "刚才第一个学校详细说说",
                "intent": "chat",
                "major": None,
                "school": None,
            },
            {
                "question": "不要推荐学校，只说就业",
                "intent": "insight",
                "major": "历史学",
                "school": None,
            },
        ]

        for case in cases:
            with self.subTest(case["question"]):
                intent = routed(case["question"])
                self.assertEqual(case["intent"], intent.intent)
                if case["major"]:
                    self.assertIn(case["major"], intent.major_names)
                if case["school"]:
                    self.assertIn(case["school"], intent.school_names)

    def test_research_queries_match_the_consult_type(self):
        cases = [
            ("这个专业中位数收入多少", ["历史学 中位数薪资"], ["专业最低分", "冲稳保"]),
            ("500强去哪些学校招聘", ["500强 校招 高校 招聘 名单"], ["专业最低分"]),
            ("华北电力大学能上吗", ["华北电力大学"], ["500强 校招"]),
            ("河南542分历史学推荐哪些学校", ["历史学", "专业最低分"], ["500强 校招"]),
        ]

        for question, expected_fragments, forbidden_fragments in cases:
            with self.subTest(question):
                request = make_request(question)
                enriched = consult_orchestrator._enrich_request_context(request)
                intent = consult_orchestrator._detect_intent(enriched)
                queries = consult_orchestrator._build_research_queries(enriched, intent)
                joined = "\n".join(queries)
                for fragment in expected_fragments:
                    self.assertIn(fragment, joined)
                for fragment in forbidden_fragments:
                    self.assertNotIn(fragment, joined)

    def test_non_recommend_consults_do_not_return_recommendation_blocks_when_llm_drifts(self):
        cases = [
            {
                "question": "这个专业中位数收入多少",
                "llm_answer": OFF_TOPIC_SCHOOL_NAME_PROMPT,
                "must_contain": ["历史学", "收入参考"],
            },
            {
                "question": "500强去哪些学校招聘",
                "llm_answer": OFF_TOPIC_RECOMMENDATION,
                "must_contain": ["500强", "校招"],
            },
            {
                "question": "这个专业能进500强吗",
                "llm_answer": OFF_TOPIC_RECOMMENDATION,
                "must_contain": ["500强"],
            },
            {
                "question": "华北电力大学能上吗",
                "llm_answer": OFF_TOPIC_RECOMMENDATION,
                "must_contain": ["华北电力大学", "这轮只判断"],
            },
            {
                "question": "历史学怎么样",
                "llm_answer": OFF_TOPIC_RECOMMENDATION,
                "must_contain": ["历史学", "就业倒推"],
            },
            {
                "question": "刚才第一个学校详细说说",
                "llm_answer": OFF_TOPIC_RECOMMENDATION,
                "must_contain": ["没有明确要求列学校"],
            },
            {
                "question": "不要推荐学校，只说就业",
                "llm_answer": OFF_TOPIC_RECOMMENDATION,
                "must_contain": ["历史学", "就业倒推"],
            },
        ]

        for case in cases:
            with self.subTest(case["question"]):
                response = self._consult_with_stubbed_llm(case["question"], case["llm_answer"])
                self.assertEqual([], response.recommendation_plans)
                self._assert_no_unrequested_recommendation(response.answer)
                for fragment in case["must_contain"]:
                    self.assertIn(fragment, response.answer)

    def test_explicit_recommendation_is_still_allowed(self):
        response = self._consult_with_stubbed_llm(
            "河南542分历史学推荐哪些学校",
            "[院校推荐]\n河南大学可以看，但要核验专业组。\n\n[红旗风险]\n注意调剂。",
        )

        self.assertGreater(len(response.recommendation_plans), 0)
        self.assertIn("河南大学", response.answer)
        self.assertIn("院校推荐", response.answer)

    def test_recommendation_history_is_filtered_for_non_recommend_followups(self):
        history = [
            {"role": "user", "content": "河南542分历史学推荐哪些学校"},
            {"role": "assistant", "content": "[院校推荐]\n1. 河南大学\n2. 河南师范大学\n冲稳保方案"},
            {"role": "user", "content": "这个专业中位数收入多少"},
        ]

        intent = routed("这个专业中位数收入多少")
        filtered = consult_orchestrator._history_for_current_question(
            history,
            "这个专业中位数收入多少",
            intent,
        )

        self.assertEqual(2, len(filtered))
        self.assertTrue(all("[院校推荐]" not in item["content"] for item in filtered))

    def test_stream_fact_question_final_does_not_override_good_delta_with_recommendation(self):
        events = self._stream_with_stubbed_llm(
            question="这个专业中位数收入多少",
            streamed_answer="[核心判断]\n历史学中位数收入参考约8K，本地估算，仅供方向判断。",
            returned_answer=OFF_TOPIC_SCHOOL_NAME_PROMPT,
        )

        delta_text = self._joined_delta_text(events)
        final_answer = self._final_answer(events)
        final_plans = self._final_plans(events)

        self.assertIn("8K", delta_text)
        self.assertIn("历史学", final_answer)
        self.assertIn("收入参考", final_answer)
        self.assertEqual([], final_plans)
        self._assert_no_unrequested_recommendation(final_answer)

    def test_stream_single_school_final_does_not_expand_to_multi_school(self):
        events = self._stream_with_stubbed_llm(
            question="华北电力大学能上吗",
            streamed_answer="[核心判断]\n只看华北电力大学，先按偏冲判断，最终查专业组位次。",
            returned_answer=OFF_TOPIC_RECOMMENDATION,
        )

        final_answer = self._final_answer(events)
        final_plans = self._final_plans(events)

        self.assertIn("华北电力大学", final_answer)
        self.assertIn("这轮只判断", final_answer)
        self.assertEqual([], final_plans)
        self._assert_no_unrequested_recommendation(final_answer)

    def test_stream_insight_question_final_does_not_become_recommendation(self):
        events = self._stream_with_stubbed_llm(
            question="历史学怎么样",
            streamed_answer="[核心判断]\n历史学能学，但普通家庭要看考编、读研和文博档案出口。",
            returned_answer=OFF_TOPIC_RECOMMENDATION,
        )

        final_answer = self._final_answer(events)
        final_plans = self._final_plans(events)

        self.assertIn("历史学", final_answer)
        self.assertIn("就业倒推", final_answer)
        self.assertEqual([], final_plans)
        self._assert_no_unrequested_recommendation(final_answer)

    def test_stream_negative_recommendation_request_stays_employment_focused(self):
        events = self._stream_with_stubbed_llm(
            question="不要推荐学校，只说就业",
            streamed_answer="[核心判断]\n只说历史学就业：教师编、考研、文博档案是主线。",
            returned_answer=OFF_TOPIC_RECOMMENDATION,
        )

        final_answer = self._final_answer(events)
        final_plans = self._final_plans(events)

        self.assertIn("历史学", final_answer)
        self.assertIn("就业倒推", final_answer)
        self.assertEqual([], final_plans)
        self._assert_no_unrequested_recommendation(final_answer)

    def test_stream_explicit_recommendation_can_still_finalize_with_plans(self):
        events = self._stream_with_stubbed_llm(
            question="河南542分历史学推荐哪些学校",
            streamed_answer="[院校推荐]\n河南大学可以看，但要核验专业组。",
            returned_answer="[院校推荐]\n河南大学可以看，但要核验专业组。\n\n[红旗风险]\n注意调剂。",
        )

        final_answer = self._final_answer(events)
        final_plans = self._final_plans(events)

        self.assertIn("院校推荐", final_answer)
        self.assertIn("河南大学", final_answer)
        self.assertGreater(len(final_plans), 0)

    def test_stream_short_final_does_not_overwrite_good_delta_or_session_history(self):
        session = session_manager.create_session(title="stream reconcile", user_profile=PROFILE)
        streamed = (
            "[核心判断]\n"
            "历史学能学，但普通家庭不能只看兴趣，要把考研、教师编、文博档案和公务员路径提前算清楚。\n\n"
            "[分析过程]\n"
            "这个方向中位数不靠头部样本撑场面，关键看学校平台、城市教育资源和孩子是否接受深造。"
        )
        try:
            events = self._stream_with_stubbed_llm(
                question="帮我简单说说当前画像",
                streamed_answer=streamed,
                returned_answer="收到。",
                session_id=session.id,
            )
            final_answer = self._final_answer(events)
            history = session_manager.get_history_messages(session.id, limit=10)

            self.assertIn("历史学能学", final_answer)
            self.assertIn("考研", final_answer)
            self.assertNotEqual("收到。", final_answer.strip())
            self.assertEqual("assistant", history[-1]["role"])
            self.assertIn("历史学能学", history[-1]["content"])
            self.assertNotEqual("收到。", history[-1]["content"].strip())
        finally:
            session_manager.delete_session(session.id)

    def test_recommendation_without_profile_does_not_hardcode_school_list(self):
        response = self._consult_with_stubbed_llm(
            "推荐哪些学校",
            "河南大学、河南师范大学可以先看，按冲稳保排一下。",
            context=None,
        )

        self.assertEqual([], response.recommendation_plans)
        self.assertIn("不能直接给学校名单", response.answer)
        self.assertIn("高考分数和位次", response.answer)
        self.assertNotIn("河南大学", response.answer)
        self.assertNotIn("河南师范大学", response.answer)
        self._assert_no_unrequested_recommendation(response.answer)

    def test_visible_answer_replaces_internal_technical_terms(self):
        response = self._consult_with_stubbed_llm(
            "历史学怎么样",
            "[核心判断]\nAgent推荐结果由后端模型根据提示词和上下文生成。历史学要看就业出口。",
        )

        self.assertIn("历史学", response.answer)
        self._assert_no_visible_technical_terms(response.answer)

    def test_school_name_alias_words_do_not_override_profile_major(self):
        request = make_request("我这个分数考上海师范大学怎么样", context=SHANGHAI_HISTORY_PROFILE)
        enriched = consult_orchestrator._enrich_request_context(request)
        intent = consult_orchestrator._detect_intent(enriched)
        user = consult_orchestrator._build_user_preferences(enriched, allow_partial=True)
        school_context = consult_orchestrator._build_school_chance_context(enriched, intent)

        self.assertEqual(["历史"], enriched.context.major_preference)
        self.assertEqual("school_chance", intent.intent)
        self.assertEqual(["上海师范大学"], intent.school_names)
        self.assertNotIn("汉语言文学", intent.major_names)
        self.assertEqual(["历史学"], user.major_preference)
        self.assertIn("目标专业方向：历史学", school_context)
        self.assertNotIn("目标专业方向：汉语言文学", school_context)

        self.assertEqual([], consult_orchestrator._extract_major_preference("北京建筑大学怎么样"))
        self.assertEqual(["汉语言文学"], consult_orchestrator._extract_major_preference("上海师范大学汉语言文学怎么样"))

        explicit_request = make_request("上海师范大学汉语言文学怎么样", context=SHANGHAI_HISTORY_PROFILE)
        explicit_enriched = consult_orchestrator._enrich_request_context(explicit_request)
        self.assertEqual(["汉语言文学"], explicit_enriched.context.major_preference)

    def test_school_chance_answer_keeps_profile_major_when_llm_switches_major(self):
        response = self._consult_with_stubbed_llm(
            "我这个分数考上海师范大学怎么样",
            "[核心判断]\n目标专业方向：汉语言文学。上海师范大学汉语言文学比较稳。",
            context=SHANGHAI_HISTORY_PROFILE,
        )

        self.assertEqual([], response.recommendation_plans)
        self.assertIn("上海师范大学", response.answer)
        self.assertIn("历史学", response.answer)
        self.assertNotIn("汉语言文学", response.answer)

    def test_current_major_pronoun_backfills_profile_major(self):
        request = make_request("这个专业怎么样", context=SHANGHAI_HISTORY_PROFILE)
        enriched = consult_orchestrator._enrich_request_context(request)
        intent = consult_orchestrator._detect_intent(enriched)

        self.assertEqual("insight", intent.intent)
        self.assertIn("历史学", intent.major_names)

        response = self._consult_with_stubbed_llm(
            "这个专业怎么样",
            "[核心判断]\n汉语言文学这个专业更适合走中文师范。",
            context=SHANGHAI_HISTORY_PROFILE,
        )
        self.assertIn("历史学", response.answer)
        self.assertNotIn("汉语言文学", response.answer)
        self.assertNotIn("金融名头", response.answer)

    def test_short_pressure_and_barrier_questions_bind_profile_major(self):
        pressure_request = make_request("做10年后压力测试", context=SHANGHAI_HISTORY_PROFILE)
        pressure_enriched = consult_orchestrator._enrich_request_context(pressure_request)
        pressure_intent = consult_orchestrator._detect_intent(pressure_enriched)

        self.assertEqual("pressure_test", pressure_intent.intent)
        self.assertIn("历史学", pressure_intent.major_names)

        pressure_response = self._consult_with_stubbed_llm(
            "做10年后压力测试",
            "[核心判断]\n经济学10年后要看宏观和金融能力。",
            context=SHANGHAI_HISTORY_PROFILE,
        )
        self.assertIn("历史学", pressure_response.answer)
        self.assertIn("10年后压力测试", pressure_response.answer)
        self.assertNotIn("经济学", pressure_response.answer)

        barrier_response = self._consult_with_stubbed_llm(
            "我的专业壁垒够不够高？",
            "[核心判断]\n经济学的专业壁垒主要是宏观思维。",
            context=SHANGHAI_HISTORY_PROFILE,
        )
        self.assertIn("历史学", barrier_response.answer)
        self.assertNotIn("经济学", barrier_response.answer)

    def test_profile_consistency_contract_matrix(self):
        cases = [
            {
                "name": "学校名里的师范不能改画像方向",
                "question": "我这个分数考上海师范大学怎么样",
                "context": SHANGHAI_HISTORY_PROFILE,
                "llm_answer": "[核心判断]\n目标专业方向：汉语言文学。上海师范大学汉语言文学比较稳。",
                "must_contain": ["上海师范大学", "历史学"],
                "forbidden": ["汉语言文学"],
            },
            {
                "name": "当前专业代词回填历史方向",
                "question": "这个专业能进500强吗",
                "context": SHANGHAI_HISTORY_PROFILE,
                "llm_answer": "[核心判断]\n汉语言文学进500强要看文案和新媒体岗位。",
                "must_contain": ["历史学", "500强"],
                "forbidden": ["汉语言文学", "[院校推荐]"],
            },
            {
                "name": "只问就业不能被旧推荐历史带成学校名单",
                "question": "不要推荐学校，只说就业",
                "context": PROFILE,
                "llm_answer": OFF_TOPIC_RECOMMENDATION,
                "must_contain": ["历史学", "就业倒推"],
                "forbidden": ["[院校推荐]", "冲稳保方案"],
            },
            {
                "name": "明确换到汉语言时允许切换方向",
                "question": "换成汉语言文学怎么样",
                "context": SHANGHAI_HISTORY_PROFILE,
                "llm_answer": "[核心判断]\n汉语言文学可以看，但要比较它和历史学的就业路径。",
                "must_contain": ["汉语言文学"],
                "forbidden": [],
            },
        ]

        for case in cases:
            with self.subTest(case["name"]):
                response = self._consult_with_stubbed_llm(
                    case["question"],
                    case["llm_answer"],
                    context=case["context"],
                )
                for fragment in case["must_contain"]:
                    self.assertIn(fragment, response.answer)
                for fragment in case["forbidden"]:
                    self.assertNotIn(fragment, response.answer)

    def test_data_truthfulness_contract_matrix(self):
        cases = [
            {
                "name": "无来源专业洞察不得声称官方真实已核验",
                "question": "历史学怎么样",
                "llm_answer": "[核心判断]\n我已经官方核验，历史学真实中位数8K，真实就业率92%。",
                "forbidden": ["已经官方核验", "真实中位数", "真实就业率", "8K", "92%"],
                "must_contain": ["数据口径"],
            },
            {
                "name": "无来源薪资问题可给估算但不得说官方真实",
                "question": "这个专业中位数收入多少",
                "llm_answer": "[核心判断]\n历史学官方真实中位数收入8K，已经核验。",
                "forbidden": ["官方真实", "已经核验"],
                "must_contain": ["8K", "本地估算"],
            },
            {
                "name": "推荐主回答不得泄露模拟概率和具体薪资数字",
                "question": "河南542分历史学推荐哪些学校",
                "llm_answer": "[院校推荐]\n河南大学模拟概率88%，薪资18K，可以作为稳妥方案。\n\n[核验清单]\n查考试院。",
                "forbidden": ["88%", "18K", "模拟概率88%", "薪资18K"],
                "must_contain": ["河南大学", "核验"],
            },
        ]

        for case in cases:
            with self.subTest(case["name"]):
                response = self._consult_with_stubbed_llm(case["question"], case["llm_answer"])
                for fragment in case["forbidden"]:
                    self.assertNotIn(fragment, response.answer)
                for fragment in case["must_contain"]:
                    self.assertIn(fragment, response.answer)

    def _consult_with_stubbed_llm(self, question: str, llm_answer: str, context=PROFILE):
        request = make_request(question, context=context)
        with patch.object(orchestrator_module.llm_client, "is_available", return_value=True), \
            patch.object(orchestrator_module.llm_client, "_complete_with_retry", return_value=llm_answer), \
            patch.object(consult_orchestrator, "_research_if_needed", return_value=[]), \
            patch.object(consult_orchestrator, "_research_recommendation_plans", return_value=[]):
            return consult_orchestrator.consult(request, history=[])

    def _stream_with_stubbed_llm(self, question: str, streamed_answer: str, returned_answer: str, context=PROFILE, session_id: str | None = None):
        def fake_complete(_messages):
            callback = orchestrator_module.llm_client._stream_callback()
            if callback:
                midpoint = max(1, len(streamed_answer) // 2)
                callback(streamed_answer[:midpoint])
                callback(streamed_answer[midpoint:])
            return returned_answer

        payload = {"question": question, "context": context.model_dump(exclude_none=True) if context else None}
        if session_id:
            payload["session_id"] = session_id
        with patch.object(orchestrator_module.llm_client, "is_available", return_value=True), \
            patch.object(orchestrator_module.llm_client, "_complete_with_retry", side_effect=fake_complete), \
            patch.object(consult_orchestrator, "_research_if_needed", return_value=[]), \
            patch.object(consult_orchestrator, "_research_recommendation_plans", return_value=[]):
            client = TestClient(app)
            response = client.post("/api/consult/stream", json=payload)

        self.assertEqual(200, response.status_code)
        return self._parse_sse(response.text)

    def _parse_sse(self, text: str):
        events = []
        for block in text.strip().split("\n\n"):
            event = None
            data_lines = []
            for line in block.splitlines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())
            if event:
                data = json.loads("\n".join(data_lines)) if data_lines else {}
                events.append((event, data))
        return events

    def _joined_delta_text(self, events):
        return "".join(data.get("text", "") for event, data in events if event == "delta")

    def _final_answer(self, events):
        finals = [data for event, data in events if event == "final"]
        self.assertEqual(1, len(finals), events)
        return finals[0]["answer"]

    def _final_plans(self, events):
        finals = [data for event, data in events if event == "final"]
        self.assertEqual(1, len(finals), events)
        return finals[0].get("recommendation_plans") or []

    def _assert_no_unrequested_recommendation(self, answer: str):
        forbidden = [
            "[院校推荐]",
            "【院校推荐】",
            "冲稳保方案",
            "冲刺方案",
            "稳妥方案",
            "保底方案",
            "哪个学校名字好听",
            "先别问哪个学校",
        ]
        for fragment in forbidden:
            self.assertNotIn(fragment, answer)

    def _assert_no_visible_technical_terms(self, answer: str):
        forbidden = [
            "Agent",
            "后端模型",
            "API模型",
            "API 模型",
            "模型回答",
            "模型生成",
            "提示词",
            "上下文",
        ]
        for fragment in forbidden:
            self.assertNotIn(fragment, answer)


if __name__ == "__main__":
    unittest.main()
