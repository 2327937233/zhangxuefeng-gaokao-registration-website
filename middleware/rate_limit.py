import time
from fastapi import Request
from fastapi.responses import JSONResponse

# 简单的内存限流器（生产环境建议用Redis）
_request_log: dict[str, list[float]] = {}

# 配置：每IP每60秒最多30次请求
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 30


async def rate_limit_middleware(request: Request, call_next):
    """简单的IP级限流中间件"""
    # 跳过健康检查和文档页面的限流
    if request.url.path in {"/", "/health", "/docs", "/redoc", "/openapi.json"}:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # 清理过期记录
    window_start = now - RATE_LIMIT_WINDOW
    history = _request_log.get(client_ip, [])
    history = [t for t in history if t > window_start]

    if len(history) >= RATE_LIMIT_MAX:
        return JSONResponse(
            status_code=429,
            content={
                "error_code": "RATE_LIMITED",
                "message": f"请求过于频繁，每{RATE_LIMIT_WINDOW}秒最多{RATE_LIMIT_MAX}次",
                "detail": None,
            },
        )

    history.append(now)
    _request_log[client_ip] = history

    return await call_next(request)
