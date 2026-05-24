import logging
import json
import re
import time
import threading
from contextlib import contextmanager
import urllib.error
import urllib.request

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from core.config import settings
from core.answer_guard import answer_guard
from core.models import ConsultRequest, ConsultResponse, ThinkingStep

logger = logging.getLogger("app.llm")


class ZXFLLMClient:
    """LLM API 封装：张雪峰角色咨询（含超时、重试、降级）"""

    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self.anthropic_api_key = settings.anthropic_api_key
        self.deepseek_api_key = settings.deepseek_api_key
        self.deepseek_base_url = settings.deepseek_base_url
        self.model = settings.deepseek_model if self.provider == "deepseek" else settings.anthropic_model
        self.timeout = settings.llm_timeout
        self.system_prompt = self._load_system_prompt()
        self.client = None
        self._stream_local = threading.local()
        if self.provider == "anthropic" and HAS_ANTHROPIC and self.anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)

    @contextmanager
    def stream_deltas_to(self, callback):
        previous = getattr(self._stream_local, "callback", None)
        self._stream_local.callback = callback
        try:
            yield
        finally:
            self._stream_local.callback = previous

    def _stream_callback(self):
        return getattr(self._stream_local, "callback", None)

    def _load_system_prompt(self) -> str:
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "prompts", "system_prompt.txt")
        path = os.path.abspath(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"System prompt not found: {path}")
            return ""

    def is_available(self) -> bool:
        if self.provider == "deepseek":
            return bool(self.deepseek_api_key)
        return self.client is not None

    def consult(
        self,
        request: ConsultRequest,
        extra_context: str = "",
        citations: list[str] | None = None,
        history: list[dict] | None = None,
    ) -> ConsultResponse:
        if not self.is_available():
            logger.warning("LLM not available, returning fallback")
            return self._fallback_response(request)

        context_str = ""
        if request.context:
            ctx = request.context.model_dump(exclude_none=True)
            if ctx:
                context_str = f"\n\n用户背景信息：{json.dumps(ctx, ensure_ascii=False)}"
        fact_data_mode = "本轮识别为数据/事实咨询" in (extra_context or "")
        fact_data_instruction = (
            "【最高优先级】本轮是数据/事实咨询，不是院校推荐。"
            "必须直接回答用户问的中位数、薪资、收入、就业或500强招聘问题；"
            "禁止输出[院校推荐]、[灵魂追问]、冲稳保、保底、投档位次、专业组推荐；"
            "如果本地估算里有薪资数字，可以展示，但必须标注“本地估算/非官方统计/仅供方向判断”。"
            "如果没有数据，就说暂无，不要改写成学校推荐。"
        ) if fact_data_mode else ""

        # 构建当前用户消息
        current_message = {
            "role": "user",
            "content": (
                f"用户原始问题：{request.question}"
                f"{context_str}\n\n"
                f"{extra_context}\n\n"
                "你必须调用张雪峰式表达方式回答本轮问题，不管用户问学校、专业、行业、闲聊还是追问，都要直接回应。"
                "请严格围绕用户原始问题回答，不要擅自改变目标省份、目标城市、分数、位次、选科或家庭背景。"
                "如果提供了联网研究摘要或Agent推荐结果，必须优先使用这些上下文，但要区分数据性质。"
                "当上下文包含“Agent推荐结果”时，推荐学校、专业、冲稳保层级只能来自Agent推荐结果，禁止自行新增学校。"
                "推荐院校的聊天回复页面不要生成具体模拟概率、具体薪资数字、薪资区间、估算中位数或不可替代性分值；这些数值只进入结构化同步方案，不写进主回答。"
                "如果你认为Agent推荐不合理，可以在回答中指出需要核验，但不能替换成上下文之外的学校名单。"
                "当上下文包含“Agent洞察结果”时，专业/学校判断必须围绕该对象，不要跑题到其他对象。"
                "当上下文包含“单校机会判断上下文”时，用户是在问某一所学校有没有机会，必须只回答这所学校，不要扩展成院校推荐列表。"
                "单校机会判断只能说“有机会/偏稳/偏冲/风险较大”这类可核验结论，禁止说“稳稳的幸福、黄埔军校、长期霸榜、天作之合、没有之一”等绝对化或宣传化话术；用户没有问其他校区时，不要主动扩展校区。"
                "凡是标记为本地估算、规则模拟、estimate/simulated 的数据，必须说清楚是估算/模拟/仅供排序参考，不能说成真实官方统计。"
                "不要把“中位数、不可替代性、模拟概率”这些指标名原样堆给用户；必须翻译成人话：中位数=普通毕业生几年后的饭碗水平，不可替代性=技术壁垒/被替代风险，模拟概率=冲稳保粗排参考。"
                "主回答只说冲稳保相对档位、专业出口和核验方向，不出现类似66%、88%、14K、18K、12K-16K这样的具体模拟值。"
                "回答推荐学校时必须按“总判断→画像策略→冲稳保分层→逐校理由→红旗风险→下一步核验清单”组织，不要写成散文，不要只罗列指标。"
                "总判断里用张雪峰式表达：先说这孩子该用什么策略，再说为什么；不要先报数字、概率和收入。"
                "冲稳保分层里，每一所学校都要单独展开，至少写2句话：第一句讲这所学校和推荐专业为什么放在这个档位，第二句讲毕业路径、未来趋势或普通家庭要防的坑。"
                "如果Agent推荐结果提供了家庭风险标签，逐校推荐必须引用其中1-3个最关键标签，并按普通/中产/富裕家庭给不同策略；不要把所有家庭都写成普通家庭。"
                "如果Agent推荐结果提供了“替代路径”，必须说明这是0候选兜底方案，并分别解释放宽城市但保专业、保城市但换相近专业、保稳妥但降低学校层次的取舍。"
                "不同学校的理由必须不同，不能用同一套话复制粘贴；同一行如果出现多所学校，也要逐所分开说明。"
                "如果几所学校层次、城市、专业都相似，也必须从校名背后的行业底色、院校类型、学校差异点、招生网核验路径里找差别；禁止连续两所学校出现高度相似的推荐语。"
                "回答学校分数线时优先引用教育考试院、阳光高考、学校本科招生网、学校官网；不能把第三方商业榜单说成官方数据。"
                "每个学校或专业后面都要说明现实含义：毕业后大概走什么路径、需要不要考研/考证、普通家庭风险在哪里。"
                "最终回答不要出现“Agent、后端、模型、提示词、上下文”这类技术词，要像真人咨询一样直接给判断；只有系统出错摘要可以留在thinking_process里，不要写进主回答。"
                "只有 citations 中有来源的内容，才可以说已联网核验。没有 citations 时，不要说真实数据或官方数据。"
                "如果上下文出现“联网状态：这轮没有拿到有效联网搜索结果”，主回答末尾必须原样保留这个意思：本轮只能按本地库粗筛，投档位次、专业组和调剂风险必须回教育考试院与学校招生网核验。"
                "如果只有考试院/学校官网入口但没有真实搜索摘要，不能说“已经核验”，只能说“给出核验入口”。"
                "如果缺少关键信息，先基于已知画像给初步判断，再在[灵魂追问]列最多3个必须补充的问题；不要机械重复要求用户补全已经提供的画像。"
                "推荐院校时必须按固定顺序输出：[分析过程]→[核心判断]→[灵魂追问]→[院校推荐]→[红旗风险]→[核验清单]→[金句]。"
                "[灵魂追问]必须完整写完后，才能开始[院校推荐]；不要在追问段落里夹带学校名单，也不要先列学校再补追问。"
                f"{fact_data_instruction}"
            )
        }

        # 合并历史消息和当前消息
        messages = []
        if history:
            messages.extend(history)
        messages.append(current_message)

        try:
            answer = self._complete_with_retry(messages)
            response = self._parse_response(answer)
            response = self._enforce_data_notice(response, extra_context, citations or [])
            if fact_data_mode:
                response.citations = citations or []
                return response
            response = answer_guard.guard_response(
                response,
                extra_context=extra_context,
                citations=citations or [],
                require_recommendation_guard="Agent推荐结果" in extra_context,
            )
            response.citations = citations or []
            return response
        except Exception as e:
            logger.error(f"LLM final failure, using local fallback: {e}")
            return self._fallback_from_context(request, extra_context, citations or [], str(e))

    def _complete_with_retry(self, messages: list[dict], max_retries: int = 1) -> str:
        """带重试的LLM调用"""
        for attempt in range(max_retries + 1):
            try:
                if self.provider == "deepseek":
                    return self._complete_deepseek(messages)
                return self._complete_anthropic(messages)
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    wait = 2 ** attempt  # 指数退避
                    logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"LLM call failed after {max_retries + 1} attempts: {e}")
                    raise

    def _complete_anthropic(self, messages: list[dict]) -> str:
        callback = self._stream_callback()
        if callback:
            pieces: list[str] = []
            with self.client.messages.stream(
                model=self.model,
                max_tokens=3200,
                system=self.system_prompt,
                messages=messages,
                timeout=self.timeout,
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        pieces.append(text)
                        callback(text)
            return "".join(pieces)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3200,
            system=self.system_prompt,
            messages=messages,
            timeout=self.timeout,
        )
        return response.content[0].text if response.content else ""

    def _complete_deepseek(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *messages,
            ],
            "temperature": 0.7,
            "max_tokens": 3200,
        }
        callback = self._stream_callback()
        if callback:
            payload["stream"] = True
            return self._complete_deepseek_stream(payload, callback)

        req = urllib.request.Request(
            self.deepseek_base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"DeepSeek API 请求失败：{e.reason}") from e
        except TimeoutError:
            raise RuntimeError(f"DeepSeek API 请求超时（>{self.timeout}秒）") from None

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"DeepSeek API 未返回有效 choices：{data}")
        return choices[0].get("message", {}).get("content", "")

    def _complete_deepseek_stream(self, payload: dict, callback) -> str:
        req = urllib.request.Request(
            self.deepseek_base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        pieces: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content") or ""
                    if delta:
                        pieces.append(delta)
                        callback(delta)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"DeepSeek API request failed: {e.reason}") from e
        except TimeoutError:
            raise RuntimeError(f"DeepSeek API request timed out after {self.timeout}s") from None

        return "".join(pieces)
    def _parse_response(self, text: str) -> ConsultResponse:
        """解析LLM返回的结构化文本"""
        thinking = []
        follow_up = []
        confidence = "medium"

        sections = {
            "分析过程": "",
            "核心判断": "",
            "灵魂追问": "",
            "院校推荐": "",
            "红旗风险": "",
            "核验清单": "",
            "金句": "",
        }
        section_aliases = {
            "分析过程": ["分析过程", "分析拆解", "分析"],
            "核心判断": ["核心判断", "总判断", "判断结论", "核心结论", "结论"],
            "灵魂追问": ["灵魂追问", "继续追问", "关键追问", "必须追问"],
            "院校推荐": ["院校推荐", "推荐院校", "学校推荐", "推荐学校", "冲稳保推荐", "具体推荐", "方案推荐"],
            "红旗风险": ["红旗风险", "风险提醒", "风险提示", "注意事项"],
            "核验清单": ["核验清单", "下一步核验清单", "数据核验", "核验入口", "下一步"],
            "金句": ["金句", "一句话总结", "总结"],
        }

        current_section = None
        lines = text.split("\n")
        parsed_lines = []

        for line in lines:
            stripped = line.strip()
            matched_section = None
            section_remainder = ""
            normalized_heading = re.sub(r"^(?:#{1,6}\s*|[-•]\s*)+", "", stripped).strip()
            for sec_name, aliases in section_aliases.items():
                alias_pattern = "|".join(re.escape(alias) for alias in aliases)
                bracket_match = re.match(
                    rf"^(?:\[\s*(?:{alias_pattern})\s*\]|【\s*(?:{alias_pattern})\s*】)\s*(.*)$",
                    normalized_heading,
                )
                plain_match = re.match(
                    rf"^(?:{alias_pattern})\s*(?:(?:[：:]\s*(.*))|$)",
                    normalized_heading,
                )
                heading_match = bracket_match or plain_match
                if heading_match:
                    matched_section = sec_name
                    section_remainder = next((group.strip() for group in heading_match.groups() if group), "")
                    break
            if matched_section:
                current_section = matched_section
                if section_remainder:
                    sections[current_section] += section_remainder + "\n"
                continue
            else:
                if current_section:
                    sections[current_section] += line + "\n"
                else:
                    parsed_lines.append(line)

        analysis_text = sections.get("分析过程", "")
        for line in analysis_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                content = line.lstrip("-• ").lstrip("0123456789.)")
                if "：" in content or ":" in content:
                    parts = content.replace(":", "：", 1).split("：", 1)
                    thinking.append(ThinkingStep(step=parts[0].strip(), analysis=parts[1].strip()))
                else:
                    thinking.append(ThinkingStep(step="分析", analysis=content))

        soul_text = sections.get("灵魂追问", "")
        for line in soul_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                follow_up.append(line.lstrip("-• ").lstrip("0123456789.) "))

        answer_parts = []
        for section_name in ["分析过程", "核心判断", "灵魂追问", "院校推荐", "红旗风险", "核验清单", "金句"]:
            section_body = sections.get(section_name, "").strip()
            if section_body:
                answer_parts.append(f"[{section_name}]\n{section_body}")
        if not answer_parts and parsed_lines:
            answer_parts.append("\n".join(parsed_lines).strip())

        answer = "\n\n".join(answer_parts) if answer_parts else text.strip()

        if "我还真不太了解" in answer or "不太清楚" in answer:
            confidence = "low"
        elif len(follow_up) > 2:
            confidence = "low"
        elif not follow_up:
            confidence = "high"

        if not thinking:
            thinking.append(ThinkingStep(step="综合分析", analysis="基于张雪峰思维框架进行分析"))

        answer = self._humanize_decision_terms(answer)
        answer = self._clean_display_text(answer)
        follow_up = [self._clean_display_text(item) for item in follow_up]

        return ConsultResponse(
            answer=answer,
            thinking_process=thinking,
            follow_up_questions=follow_up,
            confidence=confidence,
        )

    def _fallback_response(self, request: ConsultRequest) -> ConsultResponse:
        """当LLM API不可用时返回fallback回答"""
        provider_hint = (
            "请配置 DEEPSEEK_API_KEY 环境变量以启用AI咨询功能。"
            if self.provider == "deepseek"
            else "请配置 ANTHROPIC_API_KEY 环境变量以启用AI咨询功能。"
        )
        return ConsultResponse(
            answer=f"我跟你说，'{request.question}'这个问题——",
            thinking_process=[
                ThinkingStep(
                    step="系统提示",
                    analysis=f"LLM服务当前不可用。{provider_hint}当前为规则引擎fallback模式。"
                )
            ],
            follow_up_questions=[
                "孩子高考多少分？哪个省的？",
                "家里经济条件怎么样？",
                "能接受去哪些城市？",
            ],
            confidence="low",
            citations=[],
        )

    def _fallback_from_context(
        self,
        request: ConsultRequest,
        extra_context: str,
        citations: list[str],
        error: str,
    ) -> ConsultResponse:
        """LLM超时/失败时，直接用Agent和本地数据生成可读兜底，避免前端暴露技术错误。"""
        agent_text = ""
        if "Agent推荐结果：" in extra_context:
            agent_text = extra_context.split("Agent推荐结果：", 1)[1]
            for marker in ["数据口径：", "注意：", "数据真实性边界"]:
                if marker in agent_text:
                    agent_text = agent_text.split(marker, 1)[0].strip()
                    break
            agent_text = agent_text.strip()
        elif "Agent洞察结果：" in extra_context:
            agent_text = extra_context.split("Agent洞察结果：", 1)[1]
            for marker in ["数据口径：", "注意：", "数据真实性边界"]:
                if marker in agent_text:
                    agent_text = agent_text.split(marker, 1)[0].strip()
                    break
            agent_text = agent_text.strip()

        if agent_text:
            if "Agent推荐结果：" in extra_context:
                answer = self._build_recommendation_fallback(request, agent_text, citations)
            else:
                answer = self._build_insight_fallback(request, agent_text, citations)
            confidence = "medium"
        else:
            profile_hint = ""
            if request.context:
                ctx = request.context.model_dump(exclude_none=True)
                if ctx:
                    labels = {
                        "province": "省份",
                        "score": "分数",
                        "rank": "位次",
                        "subjects": "选科",
                        "family_background": "家庭条件",
                        "city_preference": "目标地区",
                        "major_preference": "专业方向",
                        "risk_appetite": "风险偏好",
                    }
                    readable = []
                    for key, label in labels.items():
                        value = ctx.get(key)
                        if value is None:
                            continue
                        if isinstance(value, list):
                            value = "、".join(map(str, value))
                        readable.append(f"{label}{value}")
                    profile_hint = f"\n\n我已经看到你的画像：{'，'.join(readable)}。"
            answer = (
                "我跟你说，先按你已经给的画像做一个原则判断。\n\n"
                f"你这轮问的是：{request.question}。{profile_hint}\n\n"
                "先给你一个原则判断：普通家庭填志愿，别只听专业名字好不好听，要看学历筛选、城市资源、专业壁垒和普通毕业生的现实出路。"
                "如果你问的是具体专业，就拿有来源的数据，或明确标注为估算的就业稳定性、收入参考、深造要求和被替代风险去压；如果你问的是学校，就拿近三年投档位次和专业组风险去压。"
            )
            confidence = "low"

        response = ConsultResponse(
            answer=self._clean_display_text(self._humanize_decision_terms(self._soften_simulated_certainty(answer))),
            thinking_process=[
                ThinkingStep(step="Agent兜底", analysis="LLM响应超时，已优先返回后端Agent/本地数据结果"),
                ThinkingStep(step="错误摘要", analysis=self._clean_display_text(error)[:180]),
            ],
            follow_up_questions=[
                "是否要按冲稳保重新列一版？",
                "是否要只看省内学校，还是接受外省？",
            ],
            confidence=confidence,
            citations=citations,
        )
        if "本轮识别为数据/事实咨询" in (extra_context or ""):
            return response
        return answer_guard.guard_response(
            response,
            extra_context=extra_context,
            citations=citations,
            require_recommendation_guard="Agent推荐结果" in extra_context,
        )

    def _build_recommendation_fallback(
        self,
        request: ConsultRequest,
        agent_text: str,
        citations: list[str],
    ) -> str:
        """把Agent原始推荐翻译成用户能听懂的张雪峰式兜底回答。"""
        summary, plans, red_flags = self._parse_agent_recommendation(agent_text)
        profile = self._format_profile(request)
        major_hint = self._format_major_hint(request)

        lines = [
            "我跟你说，先按你的考生画像把学校和专业粗排出来。先看判断，别先盯数字。",
            "",
            f"总判断：{profile}，这类孩子填志愿不能奔着名字好听去，要奔着“能不能进、毕业后能不能吃饭、普通家庭扛不扛得住风险”去。",
        ]
        if major_hint:
            lines.append(f"专业方向：{major_hint}。如果方向是工科，就优先看电气、自动化、通信、计算机这类有技术门槛的路；如果方向是文史社科，就必须同步看城市平台、考编考研路径和岗位出口。")
        if summary:
            lines.append(f"画像粗筛结论：{summary}")

        grouped = {"冲": [], "稳": [], "保": []}
        for plan in plans:
            grouped.setdefault(plan["risk"], []).append(plan)

        section_meta = {
            "冲": ("冲一冲", "这档不是让你赌命，是拿来抬上限。普通家庭可以冲，但后面必须有稳和保托住。"),
            "稳": ("稳住主线", "这档才是志愿表的骨架。学校、城市、专业三件事别全都要，能拿住两个就很不错。"),
            "保": ("保底别嫌弃", "保底不是丢人，是防止滑档。普通家庭最怕的不是学校少响亮一点，是最后没学上、专业还被调剂烂。"),
        }
        for risk in ["冲", "稳", "保"]:
            if not grouped.get(risk):
                continue
            title, note = section_meta[risk]
            lines.extend(["", f"{title}：{note}"])
            for plan in grouped[risk][:2]:
                lines.append(self._format_plan_line(plan))

        if red_flags:
            lines.extend(["", f"红旗提醒：{red_flags}"])

        lines.extend(
            [
                "",
                "你下一步别问“这个学校好不好”，你要查三张表：",
                "1. 查本省教育考试院近三年投档位次，看这个学校在你位次附近是上升还是下降。",
                "2. 查学校招生网当年招生计划，看专业组里有没有你不能接受的调剂专业。",
                "3. 查专业选科要求和培养方案，看是不是你孩子真能学、愿意学、学完能就业的方向。",
                "",
                "数据口径：这里的冲稳保是按画像和本地库做的粗排，不是真实录取承诺；收入、就业稳定性、技术壁垒是本地估算，只能辅助判断方向。最终一定以教育考试院和学校招生网为准。",
            ]
        )
        if citations:
            lines.append(self._format_citations(citations))
        return "\n".join(lines)

    def _build_insight_fallback(
        self,
        request: ConsultRequest,
        agent_text: str,
        citations: list[str],
    ) -> str:
        """把Agent洞察结果转成更像咨询的兜底回答。"""
        profile = self._format_profile(request)
        cleaned = self._humanize_decision_terms(agent_text)
        lines = [
            "我跟你说，这个问题不能干等，先按你的画像给判断。",
            "",
            f"先按你的画像看：{profile}。判断专业和学校，别只看热不热，要看它能不能把普通孩子送到一个稳定出口。",
            "",
            cleaned,
            "",
            "我的建议是：能走技术门槛，就别走纯拼表达、纯拼资源的路；必须走文史方向，就把城市、学校平台、考研考编路径一起算进去。",
            "",
            "数据口径：以上属于本地库估算和规则判断，不是官方精确统计；录取位次、招生计划、选科要求要回到教育考试院和学校招生网核验。",
        ]
        if citations:
            lines.append(self._format_citations(citations))
        return "\n".join(lines)

    def _parse_agent_recommendation(self, agent_text: str) -> tuple[str, list[dict], str]:
        summary = ""
        plans: list[dict] = []
        red_flags = ""
        for raw_line in agent_text.splitlines():
            line = raw_line.strip()
            if not line or line == "冲稳保方案：":
                continue
            if line.startswith("红旗提醒："):
                red_flags = line.split("：", 1)[1].strip()
                continue
            match = re.match(r"^\d+\.\s*\[(?P<risk>[^\]]+)\]\s*(?P<school>.*?)\s*-\s*(?P<major>.*?)，(?P<rest>.*)$", line)
            if match:
                rest = match.group("rest")
                plans.append(
                    {
                        "risk": match.group("risk").strip(),
                        "school": match.group("school").strip(),
                        "major": match.group("major").strip(),
                        "position": self._extract_fragment(rest, r"冲稳保参考(\d+%)"),
                        "salary": self._extract_fragment(rest, r"收入参考([^，。]+)"),
                        "barrier": self._extract_fragment(rest, r"被替代风险([^。]+)"),
                        "reason": self._extract_fragment(rest, r"决策含义：(.+)$"),
                    }
                )
                continue
            if not summary:
                summary = line
        return summary, plans, red_flags

    def _extract_fragment(self, text: str, pattern: str) -> str:
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    def _format_plan_line(self, plan: dict) -> str:
        pieces = [f"- {plan['school']}：{plan['major']}。"]
        if plan.get("position"):
            pieces.append("按画像粗排给到这个档位，意思是它在这一档里有参考价值，但不能当真实录取率。")
        if plan.get("salary"):
            pieces.append("普通毕业生几年后的饭碗质量要看就业质量报告和行业去向，不能把本地估算当官方工资条。")
        if plan.get("barrier"):
            pieces.append(f"技术门槛：{self._format_barrier_for_parent(plan['barrier'])}。")
        if plan.get("reason"):
            reason = re.sub(r"，?普通毕业生几年后的收入参考约\d+K", "", plan["reason"])
            reason = re.sub(r"，?技术壁垒高，被替代风险相对低", "", reason)
            pieces.append(f"为什么能看：{reason}。")
        return "".join(pieces)

    def _format_barrier_for_parent(self, barrier: str) -> str:
        match = re.match(r"(?P<score>\d+)/100，(?P<desc>.+)", barrier)
        if not match:
            return self._strip_numeric_estimates(barrier)
        score = int(match.group("score"))
        desc = match.group("desc")
        desc = re.sub(r"^壁垒(高|中上|一般|偏低)，?", "", desc)
        if score >= 85:
            level = "偏高"
        elif score >= 70:
            level = "中上"
        elif score >= 55:
            level = "一般"
        else:
            level = "偏低"
        return f"{level}，{desc}。这里不是让你看分值，是提醒你看它有没有真本事、会不会被轻易替代"

    def _format_profile(self, request: ConsultRequest) -> str:
        if not request.context:
            return "你现在给的信息还不完整"
        ctx = request.context.model_dump(exclude_none=True)
        labels = {
            "province": "省份",
            "score": "分数",
            "rank": "位次",
            "subjects": "选科",
            "family_background": "家庭",
            "city_preference": "目标地区",
            "major_preference": "方向",
            "risk_appetite": "风险偏好",
        }
        readable = []
        for key, label in labels.items():
            value = ctx.get(key)
            if value is None or value == "":
                continue
            if isinstance(value, list):
                value = "、".join(map(str, value))
            readable.append(f"{label}{value}")
        return "，".join(readable) if readable else "你现在给的信息还不完整"

    def _format_major_hint(self, request: ConsultRequest) -> str:
        if not request.context or not request.context.major_preference:
            return ""
        return "、".join(request.context.major_preference)

    def _format_citations(self, citations: list[str]) -> str:
        source_lines = "\n".join(f"- {url}" for url in citations[:5])
        return f"\n本次联网来源：\n{source_lines}"

    def _clean_display_text(self, text: str) -> str:
        """清理模型偶尔返回的 Markdown 标记，避免前端对话里露出星号。"""
        cleaned_lines = []
        for line in str(text).split("\n"):
            cleaned = line.strip()
            cleaned = cleaned.lstrip("#").strip()
            cleaned = cleaned.replace("**", "")
            if cleaned.startswith("* "):
                cleaned = "· " + cleaned[2:].strip()
            elif cleaned.startswith("*"):
                cleaned = cleaned[1:].strip()
            cleaned = self._replace_visible_technical_terms(cleaned)
            cleaned_lines.append(cleaned)
        return "\n".join(cleaned_lines).strip()

    def _replace_visible_technical_terms(self, text: str) -> str:
        """把用户可见回答里的系统技术词替换成咨询场景语言。"""
        replacements = {
            "后端Agent": "老师",
            "后端模型": "系统判断",
            "API 模型": "系统判断",
            "API模型": "系统判断",
            "模型回答": "回答",
            "模型生成": "系统整理",
            "提示词": "表达要求",
            "上下文": "已知信息",
            "Agent推荐结果": "老师给出的粗筛结果",
            "Agent洞察结果": "老师给出的分析结果",
            "Agent推荐": "老师推荐",
            "Agent洞察": "老师分析",
            "Agent输出": "画像粗排输出",
            "Agent给": "老师给",
            "Agent会": "老师会",
            "Agent已": "老师已",
            "Agent": "老师",
            "agent": "老师",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _enforce_data_notice(
        self,
        response: ConsultResponse,
        extra_context: str,
        citations: list[str],
    ) -> ConsultResponse:
        """保证主回答里明确标注本地估算/规则模拟，不能只藏在 thinking_process。"""
        fact_data_mode = "本轮识别为数据/事实咨询" in (extra_context or "")
        context_needs_notice = any(
            marker in extra_context
            for marker in [
                "Agent推荐结果",
                "Agent洞察结果",
                "本地估算",
                "规则模拟",
                "estimate",
                "simulated",
                "数据真实性边界",
            ]
        )
        if not context_needs_notice:
            return response

        answer = response.answer or ""
        has_clear_notice = "数据口径" in answer
        if not has_clear_notice:
            if fact_data_mode:
                answer = (
                    answer.rstrip()
                    + "\n\n数据口径：薪资、就业率和技术壁垒属于本地估算或公开材料辅助判断，"
                    "不是官方精确统计，只能用于方向参考；具体学校和专业去向要回到就业质量报告核验。"
                )
            else:
                answer = (
                    answer.rstrip()
                    + "\n\n数据口径：本地库里的录取概率是规则模拟，薪资、就业率和不可替代性是经验估算，"
                    "不是官方精确统计，只能用于方向判断和冲稳保排序参考。最终投档位次、招生计划、选科要求，必须以教育考试院和学校招生网为准。"
                )

        if citations and "本次联网来源" not in answer:
            source_lines = "\n".join(f"- {url}" for url in citations[:5])
            answer = answer.rstrip() + f"\n\n本次联网来源：\n{source_lines}"

        answer = self._soften_simulated_certainty(answer)
        answer = self._humanize_decision_terms(answer)
        if not fact_data_mode:
            answer = self._remove_main_answer_numeric_estimates(answer)
        response.answer = self._clean_display_text(answer)
        return response

    def _soften_simulated_certainty(self, text: str) -> str:
        """把模型对模拟录取结果的绝对化话术降温。"""
        replacements = {
            "绝对兜死": "相对更稳",
            "绝对保底": "更适合作为保底参考",
            "错不了": "方向上比较稳",
            "稳了": "相对稳",
            "稳稳的幸福": "偏稳，但必须核验专业组",
            "一定能上": "需要按官方投档位次再核验",
            "基本稳上": "从模拟排序看偏稳",
            "非常强": "有机会，但还要看专业组",
            "黄埔军校": "行业认可度较高的学校",
            "长期霸榜": "在相关行业就业里有较强存在感",
            "天作之合": "方向匹配度较高",
            "没有之一": "但不能只靠口号判断",
            "必须、一定、肯定": "必须",
            "嫡系": "对口",
            "每年大量招人": "有明确招聘场景",
            "你必须、立刻、马上": "下一步要",
            "闭着眼睛就能上": "不用核验就能上",
            "替换不了你的人": "更不容易被低端岗位替代的路径",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(
            r"这个分数段跟[^。]*?往年录取位次比[^。]*?。",
            "从本地规则粗排看，这个位置有机会，但必须回到上海市教育考试院投档表和学校招生网核验专业组。",
            text,
        )
        return text

    def _remove_main_answer_numeric_estimates(self, text: str) -> str:
        """主回答不展示本地估算薪资数字，避免用户误当官方数据。"""
        text = re.sub(r"普通毕业生几年后[^。\n]*?\d+\s*K[^。\n]*。", "普通毕业生几年后的收入要回到学校就业质量报告和行业去向核验，不能把本地估算当官方工资。", text)
        text = re.sub(r"对口的饭碗收入估算在\d+\s*K左右[^。\n]*。", "对口收入只能先看方向，最终要以就业质量报告和真实行业去向核验。", text)
        text = re.sub(r"收入参考区间[^。\n]*?\d+\s*K[^。\n]*。", "收入参考先不看本地估算数字，要回到就业质量报告和真实行业去向核验。", text)
        text = re.sub(r"薪资[^。\n]*?\d+\s*K[^。\n]*。", "薪资先不看本地估算数字，要回到就业质量报告和真实行业去向核验。", text)
        text = re.sub(r"粗排参考\d+%", "粗排参考", text)
        text = re.sub(r"模拟概率\d+%", "模拟概率只用于后台排序", text)
        text = re.sub(r"录取概率\d+%", "录取风险只用于后台排序", text)
        text = re.sub(r"概率\d+%", "概率只用于后台排序", text)
        text = re.sub(r"\d+\s*/\s*100", "后台分值", text)
        text = re.sub(r"约\d+\s*K", "待官方就业质量报告核验", text)
        text = re.sub(r"\d+\s*K左右", "待官方就业质量报告核验", text)
        text = re.sub(r"\d+\s*K\s*-\s*\d+\s*K", "待官方就业质量报告核验", text)
        text = re.sub(r"\b\d+\s*K\b", "待官方就业质量报告核验", text)
        text = re.sub(r"\b\d+%", "后台粗排参考", text)
        return text

    def _strip_numeric_estimates(self, text: str) -> str:
        text = re.sub(r"\d+\s*/\s*100", "后台分值", text or "")
        text = re.sub(r"(?:约)?\d+\s*K(?:\s*-\s*\d+\s*K)?", "待核验", text)
        text = re.sub(r"\d+%", "后台粗排", text)
        return text

    def _humanize_decision_terms(self, text: str) -> str:
        """把面向模型的指标名改成面向家长的决策语言。"""
        replacements = {
            "5年后薪资中位数": "普通毕业生工作几年后的收入参考",
            "5年估算中位数薪资": "普通毕业生几年后的收入参考",
            "估算中位数薪资": "普通毕业生几年后的收入参考",
            "中位数薪资": "普通毕业生几年后的收入参考",
            "薪资中位数": "普通毕业生几年后的收入参考",
            "中位数": "普通人水平",
            "不可替代性评分": "技术壁垒评分",
            "不可替代性估算": "技术壁垒/被替代风险估算",
            "不可替代性": "技术壁垒/被替代风险",
            "规则模拟概率": "按画像粗排的冲稳保参考",
            "模拟概率": "按画像粗排的冲稳保参考",
            "录取概率": "录取风险参考",
            "估算就业率": "就业稳定性参考",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text


llm_client = ZXFLLMClient()
