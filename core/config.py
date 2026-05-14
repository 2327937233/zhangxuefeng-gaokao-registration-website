import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "张雪峰思维操作系统 API"
    app_version: str = "1.0.0"
    debug: bool = False

    # CORS：生产环境必须限制来源，禁止用 ["*"]
    # 开发环境可设为 ["http://localhost:3000", "http://localhost:5173", "http://localhost:5500", "http://127.0.0.1:5500"]
    # 生产环境应设为具体域名，如 ["https://yourdomain.com"]
    cors_origins_str: str = os.getenv(
        "CORS_ORIGINS",
        "null,http://localhost:3000,http://localhost:5173,http://localhost:5500,"
        "http://localhost:8000,http://127.0.0.1:5500,http://127.0.0.1:8000",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_str.split(",") if origin.strip()]

    # LLM Provider
    llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")

    # 超时配置（秒）
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "40"))
    research_timeout: int = int(os.getenv("RESEARCH_TIMEOUT", "6"))

    # 限流配置
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    rate_limit_max: int = int(os.getenv("RATE_LIMIT_MAX", "30"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
