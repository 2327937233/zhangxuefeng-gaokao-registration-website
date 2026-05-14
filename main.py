from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.models import ConsultRequest
from middleware.error_handler import (
    app_exception_handler,
    global_exception_handler,
    AppException,
)
from middleware.logging import setup_logging, logging_middleware
from middleware.rate_limit import rate_limit_middleware
from api import consult, evaluate, data, agent, sessions

# 配置日志
setup_logging()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="张雪峰思维操作系统后端 API。提供基于5个核心心智模型和8条决策启发式的智能咨询、决策评估与Agent服务。",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# 注册异常处理器
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# 注册中间件
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(logging_middleware)

# CORS：从配置读取，不再硬编码 ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Routers
app.include_router(consult.router, prefix="/api")
app.include_router(evaluate.router, prefix="/api")
app.include_router(data.router, prefix="/api")
app.include_router(agent.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else None,
        "endpoints": {
            "consult": "POST /api/consult (支持 session_id 多轮对话)",
            "evaluate": "POST /api/evaluate",
            "soul_questions": "POST /api/evaluate/soul-questions",
            "quote": "GET /api/data/quote",
            "majors": "GET /api/data/majors",
            "schools": "GET /api/data/schools",
            "sessions": {
                "create": "POST /api/sessions",
                "list": "GET /api/sessions",
                "get": "GET /api/sessions/{id}",
                "delete": "DELETE /api/sessions/{id}",
                "update_profile": "PUT /api/sessions/{id}/profile",
                "rename": "PUT /api/sessions/{id}/rename",
            },
            "agent": {
                "recommend": "POST /api/agent/recommend",
                "compare": "POST /api/agent/compare",
                "insights": "POST /api/agent/insights",
                "pressure_test": "POST /api/agent/pressure-test",
                "analyze": "POST /api/agent/analyze",
            },
        },
    }


@app.get("/health")
async def health():
    from core.llm_client import llm_client
    from core.config import settings
    return {
        "status": "ok",
        "llm_available": llm_client.is_available(),
        "llm_provider": settings.llm_provider,
        "llm_model": llm_client.model,
        "recognition_version": "school_chance_alias_v2",
    }


@app.post("/debug/intent")
async def debug_intent(request: ConsultRequest):
    from core.consult_orchestrator import consult_orchestrator

    enriched = consult_orchestrator._enrich_request_context(request)
    intent = consult_orchestrator._detect_intent(enriched)
    queries = consult_orchestrator._build_research_queries(enriched, intent)
    return {
        "question": enriched.question,
        "context": enriched.context.model_dump(exclude_none=True) if enriched.context else None,
        "intent": intent.intent,
        "schools": intent.school_names,
        "majors": intent.major_names,
        "needs_research": intent.needs_research,
        "queries": queries[:10],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
