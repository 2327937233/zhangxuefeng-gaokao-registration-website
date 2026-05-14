import logging
import time
from fastapi import Request

logger = logging.getLogger("app.access")


def setup_logging():
    """配置全局日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 降低第三方库的日志噪音
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def logging_middleware(request: Request, call_next):
    """请求日志中间件：记录每个请求的耗时和状态"""
    start = time.time()
    client = request.client.host if request.client else "unknown"

    try:
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.info(
            f"{request.method} {request.url.path} | status={response.status_code} | {duration:.1f}ms | client={client}"
        )
        return response
    except Exception as exc:
        duration = (time.time() - start) * 1000
        logger.error(
            f"{request.method} {request.url.path} | status=500 | {duration:.1f}ms | client={client} | error={str(exc)}"
        )
        raise
