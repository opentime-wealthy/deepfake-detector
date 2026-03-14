# © 2026 TimeWealthy Limited — DeepGuard
"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "DeepGuard API"
    app_version: str = "0.1.0"
    debug: bool = False

    # Database
    database_url: str = "sqlite:///./dev.db"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379"

    # Security
    secret_key: str = "changeme-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24h

    # S3-compatible storage
    s3_bucket: str = "deepguard-uploads"
    s3_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""

    # Upload limits
    max_upload_mb: int = 500

    # Quota per plan
    plan_quotas: dict = {
        "free": 10,
        "journalist": 100,
        "pro": 500,
        "enterprise": 999999,
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
