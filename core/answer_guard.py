import re
from dataclasses import dataclass
from typing import Iterable

from core.models import ConsultResponse, ThinkingStep


SECTION_ORDER = ["分析过程", "核心判断", "灵魂追问", "院校推荐", "红旗风险", "核验清单", "金句"]

SECTION_ALIASES = {
    "分析过程": ["分析过程", "分析拆解", "分析"],
    "核心判断": ["核心判断", "总判断", "判断结论", "核心结论", "结论"],
    "灵魂追问": ["灵魂追问", "继续追问", "关键追问", "必须追问"],
    "院校推荐": ["院校推荐", "推荐院校", "学校推荐", "推荐学校", "冲稳保推荐", "具体推荐", "方案推荐"],
    "红旗风险": ["红旗风险", "风险提醒", "风险提示", "注意事项"],
    "核验清单": ["核验清单", "下一步核验清单", "数据核验", "核验入口", "下一步"],
    "金句": ["金句", "一句话总结", "总结"],
}

RECOMMENDATION_LABELS = set(SECTION_ALIASES["院校推荐"])


@dataclass
class GuardResult:
    answer: str
    sections: dict[str, str]
    follow_up_questions: list[str]


class AnswerGuard:
    """Final safety pass for consultation answers before they are shown to users."""

    def guard_response(
        self,
        response: ConsultResponse,
        *,
        extra_context: str = "",
        citations: list[str] | None = None,
        allowed_schools: Iterable[str] | None = None,
        known_school_names: Iterable[str] | None = None,
        require_recommendation_guard: bool = False,
    ) -> ConsultResponse:
        result = self.guard_answer(
            response.answer,
            extra_context=extra_context,
            citations=citations or response.citations or [],
            follow_up_questions=response.follow_up_questions,
            allowed_schools=allowed_schools,
            known_school_names=known_school_names,
            require_recommendation_guard=require_recommendation_guard,
        )
        response.answer = result.answer
        if result.follow_up_questions:
            response.follow_up_questions = result.follow_up_questions
        if not response.thinking_process:
            response.thinking_process = [ThinkingStep(step="回答质检", analysis="已按固定段落顺序和数据边界完成展示前校验")]
        return response

    def guard_answer(
        self,
        answer: str,
        *,
        extra_context: str = "",
        citations: list[str] | None = None,
        follow_up_questions: list[str] | None = None,
        allowed_schools: Iterable[str] | None = None,
        known_school_names: Iterable[str] | None = None,
        require_recommendation_guard: bool = False,
    ) -> GuardResult:
        sections, prefix = self.parse_sections(answer or "")
        allowed = self._normalize_names(allowed_schools) or self._extract_allowed_schools(extra_context)
        known = self._normalize_names(known_school_names)

        if prefix and not any(sections.values()):
            sections["核心判断"] = prefix
        elif prefix:
            sections["分析过程"] = self._join_blocks(sections["分析过程"], prefix)

        has_recommendation = bool(sections["院校推荐"].strip())
        if not has_recommendation and allowed and self._contains_any(answer or "", allowed):
            has_recommendation = True
        if require_recommendation_guard:
            has_recommendation = True
        remove_simulated_numbers = require_recommendation_guard or self._context_has_simulated_data(extra_context)

        if has_recommendation:
            sections["灵魂追问"] = self._ensure_soul_questions(
                sections["灵魂追问"],
                follow_up_questions or [],
            )
            sections["核验清单"] = self._ensure_checklist(sections["核验清单"], citations or [])

        if allowed and known:
            sections["院校推荐"] = self._remove_forbidden_school_mentions(
                sections["院校推荐"],
                allowed_schools=allowed,
                known_school_names=known,
            )

        for key in SECTION_ORDER:
            sections[key] = self._sanitize_visible_text(sections[key], remove_numbers=remove_simulated_numbers)
            if not citations:
                sections[key] = self._downgrade_unsupported_source_claims(sections[key])

        rebuilt = self.build_answer(sections)
        return GuardResult(
            answer=rebuilt,
            sections=sections,
            follow_up_questions=self._extract_follow_up_questions(sections["灵魂追问"]),
        )

    def parse_sections(self, text: str) -> tuple[dict[str, str], str]:
        sections = {name: "" for name in SECTION_ORDER}
        prefix_lines: list[str] = []
        current_section: str | None = None

        for line in str(text or "").splitlines():
            matched, remainder = self._match_heading(line)
            if matched:
                current_section = matched
                if remainder:
                    sections[current_section] = self._join_blocks(sections[current_section], remainder)
                continue
            if current_section:
                sections[current_section] += line + "\n"
            else:
                prefix_lines.append(line)

        return {key: value.strip() for key, value in sections.items()}, "\n".join(prefix_lines).strip()

    def build_answer(self, sections: dict[str, str]) -> str:
        parts: list[str] = []
        for name in SECTION_ORDER:
            body = (sections.get(name) or "").strip()
            if body:
                parts.append(f"[{name}]\n{body}")
        return "\n\n".join(parts).strip()

    def choose_more_complete(self, final_answer: str, streamed_answer: str) -> str:
        final = (final_answer or "").strip()
        streamed = (streamed_answer or "").strip()
        if not final:
            return streamed
        if not streamed:
            return final
        final_score = self._completeness_score(final)
        streamed_score = self._completeness_score(streamed)
        if streamed_score >= final_score + 2 and len(streamed) >= max(120, int(len(final) * 0.65)):
            return streamed
        if self._has_recommendation(streamed) and not self._has_recommendation(final):
            return streamed
        if len(streamed) > len(final) + 260 and streamed_score >= final_score:
            return streamed
        return final

    def _match_heading(self, line: str) -> tuple[str | None, str]:
        stripped = re.sub(r"^(?:#{1,6}\s*|[-•]\s*)+", "", line.strip()).strip()
        for section_name, aliases in SECTION_ALIASES.items():
            alias_pattern = "|".join(re.escape(alias) for alias in aliases)
            bracket_match = re.match(
                rf"^(?:\[\s*(?:{alias_pattern})\s*\]|【\s*(?:{alias_pattern})\s*】)\s*(.*)$",
                stripped,
            )
            plain_match = re.match(
                rf"^(?:{alias_pattern})\s*(?:(?:[：:]\s*(.*))|$)",
                stripped,
            )
            match = bracket_match or plain_match
            if match:
                remainder = next((group.strip() for group in match.groups() if group), "")
                return section_name, remainder
        return None, ""

    def _ensure_soul_questions(self, soul_text: str, follow_up_questions: list[str]) -> str:
        existing = (soul_text or "").strip()
        if self._extract_follow_up_questions(existing):
            return existing

        questions = [item.strip() for item in follow_up_questions if item and item.strip()]
        if not questions:
            questions = [
                "目标城市是硬约束，还是为了专业和学校层次可以适当放宽？",
                "能不能接受被调剂到相邻专业，哪些专业绝对不能碰？",
                "家庭对读研、转专业、复读或高成本城市生活的承受能力到哪一步？",
            ]
        return "\n".join(f"{index + 1}. {question}" for index, question in enumerate(questions[:3]))

    def _ensure_checklist(self, checklist_text: str, citations: list[str]) -> str:
        existing = (checklist_text or "").strip()
        required_terms = ["教育考试院", "阳光高考", "学校招生网", "专业组", "调剂"]
        if existing and all(term in existing for term in required_terms):
            return existing

        lines = []
        if existing:
            lines.append(existing)
        lines.extend(
            [
                "1. 查本省教育考试院近三年投档表，先看院校专业组和位次，不只看学校最低分。",
                "2. 查阳光高考和学校本科招生网，核对招生计划、选科要求、专业组包含专业和调剂规则。",
                "3. 查目标学院培养方案和就业质量报告，确认普通毕业生出口、读研比例和行业去向。",
                "4. 如果本轮没有官方来源支撑，所有冲稳保和收入判断都只能当作粗筛参考。",
            ]
        )
        if citations:
            lines.append("5. 已给出的联网入口只作为核验入口，最终仍以官方发布页面为准。")
        return "\n".join(self._dedupe_lines(lines))

    def _sanitize_visible_text(self, text: str, *, remove_numbers: bool = False) -> str:
        cleaned = str(text or "")
        cleaned = cleaned.replace("**", "")
        cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"(?m)^\s*\*\s+", "· ", cleaned)
        if remove_numbers:
            cleaned = self._remove_numeric_estimates(cleaned)
        cleaned = self._replace_technical_terms(cleaned)
        return cleaned.strip()

    def _remove_numeric_estimates(self, text: str) -> str:
        text = re.sub(r"(?:约|大概|估算)?\s*\d+(?:\.\d+)?\s*[Kk]\s*(?:[-~—至到]\s*\d+(?:\.\d+)?\s*[Kk])?", "待就业质量报告核验", text)
        text = re.sub(r"\d+\s*/\s*100", "后台分值", text)
        text = re.sub(r"(模拟概率|录取概率|概率|粗排参考|参考概率)\s*[:：]?\s*\d{1,3}\s*%", r"\1只用于后台排序", text)
        text = re.sub(r"\d{1,3}\s*%\s*(?:录取概率|模拟概率|概率|粗排参考)", "后台粗排参考", text)
        text = re.sub(r"(?<!\d)\d{1,3}\s*%(?!\d)", "后台粗排参考", text)
        text = re.sub(r"薪资区间[^。\n]*待就业质量报告核验[^。\n]*", "薪资区间需以学校就业质量报告和真实行业去向核验", text)
        return text

    def _downgrade_unsupported_source_claims(self, text: str) -> str:
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

    def _replace_technical_terms(self, text: str) -> str:
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
            "Agent": "老师",
            "agent": "老师",
            "recommendation_plans": "同步方案",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _remove_forbidden_school_mentions(
        self,
        text: str,
        *,
        allowed_schools: set[str],
        known_school_names: set[str],
    ) -> str:
        forbidden = known_school_names - allowed_schools
        if not text or not forbidden:
            return text
        kept_lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                kept_lines.append(raw_line)
                continue
            forbidden_hits = [name for name in forbidden if name in line]
            if not forbidden_hits:
                kept_lines.append(raw_line)
                continue
            allowed_hit = any(name in line for name in allowed_schools)
            if not allowed_hit:
                continue
            fragments = re.split(r"(?<=[。；;])|、|，", raw_line)
            safe_fragments = [
                fragment
                for fragment in fragments
                if fragment.strip() and not any(name in fragment for name in forbidden_hits)
            ]
            if safe_fragments:
                kept_lines.append("".join(safe_fragments).strip())
        cleaned = "\n".join(kept_lines).strip()
        if not cleaned:
            return "本轮院校推荐只保留老师粗筛结果里的学校；其他学校不在当前候选池内，不能直接加入。"
        return cleaned

    def _extract_allowed_schools(self, extra_context: str) -> set[str]:
        names: set[str] = set()
        for line in str(extra_context or "").splitlines():
            match = re.search(r"^\s*\d+\.\s*\[[^\]]+\]\s*([^-，,]+?)\s*-\s*", line)
            if match:
                names.add(match.group(1).strip())
        return names

    def _extract_follow_up_questions(self, text: str) -> list[str]:
        questions: list[str] = []
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0].isdigit() or stripped.startswith(("-", "•", "·")):
                cleaned = stripped.lstrip("-•· ").lstrip("0123456789.)、 ").strip()
                if cleaned:
                    questions.append(cleaned)
        return questions[:3]

    def _completeness_score(self, text: str) -> int:
        sections, prefix = self.parse_sections(text)
        score = sum(1 for value in sections.values() if value.strip())
        if sections.get("灵魂追问", "").strip():
            score += 2
        if sections.get("院校推荐", "").strip():
            score += 2
        if sections.get("核验清单", "").strip():
            score += 1
        if prefix:
            score += 1
        return score

    def _has_recommendation(self, text: str) -> bool:
        sections, _ = self.parse_sections(text)
        return bool(sections.get("院校推荐", "").strip())

    def _contains_any(self, text: str, names: set[str]) -> bool:
        return any(name and name in text for name in names)

    def _context_has_simulated_data(self, extra_context: str) -> bool:
        markers = ["Agent推荐结果", "Agent洞察结果", "本地估算", "规则模拟", "estimate", "simulated", "数据真实性边界"]
        return any(marker in (extra_context or "") for marker in markers)

    def _normalize_names(self, names: Iterable[str] | None) -> set[str]:
        return {str(name).strip() for name in (names or []) if str(name).strip()}

    def _join_blocks(self, first: str, second: str) -> str:
        first = (first or "").strip()
        second = (second or "").strip()
        if not first:
            return second
        if not second:
            return first
        return first + "\n" + second

    def _dedupe_lines(self, lines: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for line in lines:
            normalized = line.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result


answer_guard = AnswerGuard()
