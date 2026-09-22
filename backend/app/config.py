import os
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application Configuration Settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Cloud VLM Provider ("gemini", "anthropic", "mock")
    vlm_provider: Literal["gemini", "anthropic", "mock"] = "gemini"
    vlm_model: str = "gemini-2.5-flash"

    # API Keys
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Decision & Fuzzy Matching Thresholds (Strict Binary: APPROVED or REJECTED)
    fuzzy_name_threshold: float = 0.85
    fuzzy_dl_threshold: float = 0.88
    composite_approval_threshold: float = 0.85

    # Privacy Face Cropping
    face_crop_padding_ratio: float = 0.15

    # Verification Pipeline Latency & Timeouts
    pipeline_timeout_seconds: float = 45.0

    # Storage Configuration
    uploads_dir: str = "storage/uploads"

    # Security & Rate Limiting
    api_secret_key: Optional[str] = None
    enable_api_key_auth: bool = False
    rate_limit_per_minute: int = 60

    # Server Info & Deployment Configuration
    app_name: str = "Komüte Driver Verification Service v2"
    app_version: str = "2.0.0"
    environment: str = "production"
    host: str = "0.0.0.0"
    port: int = 8080


# Global cached settings instance
settings = Settings()

