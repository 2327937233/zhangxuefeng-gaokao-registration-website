import json
import queue
import threading

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from core.models import ConsultRequest, ConsultResponse
from core.consult_orchestrator import consult_orchestrator
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
        raise ValidationError("闂涓嶈兘涓虹┖")

    history = _prepare_history(request)

    def event_generator():
        events: queue.Queue[tuple[str, object]] = queue.Queue()

        def push_delta(text: str) -> None:
            events.put(("delta", {"text": text}))

        def worker() -> None:
            try:
                events.put(("status", {"message": "正在分析画像和检索上下文"}))
                with llm_client.stream_deltas_to(push_delta):
                    response = consult_orchestrator.consult(request, history=history)
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
