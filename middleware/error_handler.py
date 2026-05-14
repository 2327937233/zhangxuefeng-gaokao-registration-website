import logging
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("app.error")


class AppException(Exception):
    """应用层自定义异常"""

    def __init__(self, message: str, status_code: int = 400, error_code: str = "BAD_REQUEST"):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)


class ValidationError(AppException):
    """输入校验失败"""

    def __init__(self, message: str):
        super().__init__(message, status_code=422, error_code="VALIDATION_ERROR")


class NotFoundError(AppException):
    """资源不存在"""

    def __init__(self, message: str):
        super().__init__(message, status_code=404, error_code="NOT_FOUND")


class ExternalServiceError(AppException):
    """外部服务异常（LLM/搜索等）"""

    def __init__(self, message: str = "外部服务暂不可用，请稍后重试"):
        super().__init__(message, status_code=503, error_code="EXTERNAL_SERVICE_ERROR")


class RateLimitError(AppException):
    """请求频率超限"""

    def __init__(self, message: str = "请求过于频繁，请稍后再试"):
        super().__init__(message, status_code=429, error_code="RATE_LIMITED")


async def app_exception_handler(request: Request, exc: AppException):
    """处理自定义应用异常"""
    logger.warning(
        f"AppException [{exc.error_code}] {request.method} {request.url.path}: {exc.message}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "detail": None,
        },
    )


async def global_exception_handler(request: Request, exc: Exception):
    """捕获所有未处理异常，防止内部信息泄露"""
    trace = traceback.format_exc()
    logger.error(
        f"UnhandledException {request.method} {request.url.path}: {str(exc)}\n{trace}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "系统内部错误，请稍后重试",
            "detail": None,
        },
    )
