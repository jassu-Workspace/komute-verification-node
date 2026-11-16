from __future__ import annotations

import base64
import binascii
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Real environment variables win over .env. The .env file is git-ignored; see .env.example.
    model_config = SettingsConfigDict(env_prefix="KOMUTE_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["dev", "production"] = "dev"
    api_key: SecretStr | None = None  # unset or empty = no authentication (local testing only)
    storage_root: Path
    storage_key_b64: SecretStr
    storage_key_id: str
    manifest_key_b64: SecretStr
    manifest_key_id: str
    models_dir: Path = Path("backend/models")
    liveness_model: Path | None = None
    dev_skip_liveness: bool = False
    response_barcode_fields: bool = False  # true adds every decoded barcode element (address too) to the response
    face_match_threshold: float = Field(0.35, ge=0.2, le=0.95)  # AgeDB-30: 0/3000 false accepts at 0.35
    face_detect_score: float = Field(0.7, gt=0, lt=1)
    liveness_threshold: float = Field(0.5, gt=0, lt=1)
    name_match_threshold: float = Field(90.0, ge=50, le=100)
    min_age: int = Field(18, ge=16, le=100)
    plate_detect_score: float = Field(0.35, gt=0, lt=1)
    barcode_budget_s: float = Field(10.0, gt=0, le=60)
    max_text_chars: int = Field(200, ge=20, le=1000)
    # MiniFASNet-style defaults. Unverified until the owner supplies the liveness model file.
    liveness_crop_scale: float = 2.7
    liveness_input_size: int = 80
    liveness_live_index: int = 1
    plate_min_confidence: float = 0.6
    max_upload_bytes: int = 10_000_000
    max_image_side: int = 2048
    compressed_max_side: int = 1280  # the compressed tier is smaller than the capped working image
    retry_after_s: int = 5
    rejected_images_ttl_h: int = 72
    approved_originals_ttl_d: int = 30
    sweep_interval_s: int = 3600
    warmup_max_rss_mb: float = 400  # warm the models at start only while the node is below this; 0 disables

    @field_validator("api_key")
    @classmethod
    def _empty_key_is_no_key(cls, v: SecretStr | None) -> SecretStr | None:
        return v if v is not None and v.get_secret_value() else None

    @field_validator("liveness_model", mode="before")
    @classmethod
    def _blank_path_is_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v  # a blank .env line must not become Path(".")

    @field_validator("storage_key_b64", "manifest_key_b64")
    @classmethod
    def _key_is_32_bytes(cls, v: SecretStr) -> SecretStr:
        try:
            raw = base64.b64decode(v.get_secret_value(), validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError("key must be valid base64") from e
        if len(raw) != 32:
            raise ValueError("key must decode to 32 bytes")
        return v

    @field_validator("storage_root")
    @classmethod
    def _not_onedrive(cls, v: Path) -> Path:
        # String check only (no filesystem access); splits on both separators.
        for part in re.split(r"[\\/]", str(v)):
            p = part.lower()
            if p == "onedrive" or p.startswith("onedrive - "):
                raise ValueError("storage_root must not be inside OneDrive")
        return v

    @model_validator(mode="after")
    def _production_rules(self) -> Settings:
        if self.env == "production":
            if self.api_key is None or len(self.api_key.get_secret_value()) < 24:
                raise ValueError("api_key (at least 24 characters) is required in production")
            if self.dev_skip_liveness:
                raise ValueError("dev_skip_liveness is forbidden in production")
            if self.liveness_model is None:
                raise ValueError("liveness_model is required in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
