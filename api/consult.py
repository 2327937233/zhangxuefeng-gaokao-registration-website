import json
import queue
import re
import threading

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from core.models import ConsultRequest, ConsultResponse
from core.consult_orchestrator import consult_orchestrator
from core.answer_guard import answer_guard
from core.llm_client import llm_client
from core.session_manager import session_manager

router = APIRouter(prefix="/consult", tags=["consult"])


def _prepare_history(request: ConsultRequest) -> list[dict] | None:
    history = None
    if request.session_id:
        session = session_manager.get_session(request.session_id)
        if session:
            if not request.context and session.user_profile:
                request.context = session.user_profile
            history = session_manager.get_history_messages(request.session_id, limit=10)
    return history


def _sse(event: str, data) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _has_recommendation_block(text: str) -> bool:
    return bool(re.search(
        r"(?:^|\n)\s*(?:\[|【)?\s*(?:院校推荐|推荐院校|学校推荐|推荐学校|冲稳保推荐|具体推荐|方案推荐)\s*(?:\]|】)?",
        text or "",
    ))


def _is_explicit_recommendation_question(question: str) -> bool:
    if consult_orchestrator._is_fact_data_question(question):
        return False
    compact = re.sub(r"\s+", "", question or "")
    return bool(re.search(
        r"(院校推荐|学校推荐|推荐院校|推荐学校|冲稳保|志愿|填报|报考|推荐哪些学校|推荐什么学校|哪些学校适合|适合哪些学校|能报哪些|能上哪些|报哪些|报什么学校|选哪些学校|选什么学校|该报|怎么报|怎么选学校)",
        compact,
    ))


def _reconcile_stream_final(
    request: ConsultRequest,
    response: ConsultResponse,
    streamed_text: str,
) -> ConsultResponse:
    streamed = (streamed_text or "").strip()
    final = (response.answer or "").strip()
    if not streamed or streamed == final:
        return response

    enriched = consult_orchestrator._enrich_request_context(request)
    intent = consult_orchestrator._detect_intent(enriched)
    streamed_has_recommendation = _has_recommendation_block(streamed)
    final_has_recommendation = _has_recommendation_block(final)
    explicit_recommendation = _is_explicit_recommendation_question(enriched.question)
    streamed_unrequested = consult_orchestrator._is_unrequested_recommendation_answer(streamed, enriched, intent)
    streamed_major_conflict = consult_orchestrator._answer_conflicts_with_major_scope(streamed, enriched, intent)
    streamed_safe = not streamed_unrequested and not streamed_major_conflict
    final_scope_fallback = "这轮问题是" in final and "没有明确要求列学校" in final

    use_streamed = False
    if consult_orchestrator._is_fact_data_question(enriched.question):
        final_off_topic = consult_orchestrator._is_fact_answer_off_topic(final, enriched)
        streamed_off_topic = consult_orchestrator._is_fact_answer_off_topic(streamed, enriched)
        use_streamed = final_off_topic and not streamed_off_topic
    elif not explicit_recommendation and final_has_recommendation and not streamed_has_recommendation:
        use_streamed = True
    elif (
        consult_orchestrator._answer_conflicts_with_major_scope(final, enriched, intent)
        and not streamed_major_conflict
    ):
        use_streamed = True
    elif streamed_safe and not explicit_recommendation and final_scope_fallback and len(streamed) >= 80:
        use_streamed = True
    else:
        preferred = answer_guard.choose_more_complete(final, streamed)
        use_streamed = streamed_safe and preferred == streamed
        if not use_streamed and streamed_safe and len(streamed) >= 120 and len(final) < len(streamed) * 0.45:
            use_streamed = True

    if not use_streamed:
        return response

    response.answer = streamed
    if not explicit_recommendation and not streamed_has_recommendation:
        response.recommendation_plans = []
    if consult_orchestrator._is_fact_data_question(enriched.question):
        response.follow_up_questions = []
        response.recommendation_plans = []
    return response


@router.post("", response_model=ConsultResponse)
async def consult(request: ConsultRequest):
    """
    张雪峰式智能咨询（支持会话多轮对话）

    接收用户问题和背景信息，返回基于5个心智模型+8条决策启发式的分析回答。
    携带 session_id 时，自动继承该会话的历史上下文和考生画像。

    **示例请求（携带会话）：**
    ```json
    {
      "session_id": "sess_abc123",
      "question": "刚才推荐的第一个学校再详细说说？",
      "context": {
        "score": 620,
        "province": "山东",
        "family_background": "普通家庭"
      }
    }
    ```
    """
    if not request.question or not request.question.strip():
        from middleware.error_handler import ValidationError
        raise ValidationError("问题不能为空")

    history = None

    # 如果携带了 session_id，自动继承会话中的考生画像和历史记录
    if request.session_id:
        session = session_manager.get_session(request.session_id)
        if session:
            # 自动继承 session 中已绑定的考生画像（用户未显式提供时）
            if not request.context and session.user_profile:
                request.context = session.user_profile
            history = session_manager.get_history_messages(request.session_id, limit=10)

    # 调用编排器（传入历史消息实现多轮对话）
    response = consult_orchestrator.consult(request, history=history)

    # 保存对话记录到会话
    if request.session_id:
        session_manager.add_message(request.session_id, "user", request.question)
        session_manager.add_message(request.session_id, "assistant", response.answer)

    return response


@router.post("/stream")
async def consult_stream(request: ConsultRequest):
    if not request.question or not request.question.strip():
        from middleware.error_handler import ValidationError
        raise ValidationError("问题不能为空")

    history = _prepare_history(request)

    def event_generator():
        events: queue.Queue[tuple[str, object]] = queue.Queue()
        streamed_parts: list[str] = []

        def push_delta(text: str) -> None:
            streamed_parts.append(text)
            events.put(("delta", {"text": text}))

        def worker() -> None:
            try:
                events.put(("status", {"message": "正在分析画像和检索上下文"}))
                with llm_client.stream_deltas_to(push_delta):
                    response = consult_orchestrator.consult(request, history=history)
                response = _reconcile_stream_final(request, response, "".join(streamed_parts))
                if request.session_id:
                    session_manager.add_message(request.session_id, "user", request.question)
                    session_manager.add_message(request.session_id, "assistant", response.answer)
                events.put(("final", response.model_dump()))
            except Exception as exc:
                events.put(("error", {"message": str(exc)}))
            finally:
                events.put(("done", {}))

        threading.Thread(target=worker, daemon=True).start()

        while True:
            event, data = events.get()
            yield _sse(event, data)
            if event == "done":
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
