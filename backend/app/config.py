import os
from pathlib import Path
from typing import List, Literal, Optional
from dotenv import dotenv_values, load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory for backend and repo root
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent

# Deterministic .env precedence: lowest first, repo root is authoritative.
ENV_LOAD_ORDER: List[Path] = [
    Path.cwd() / ".env",
    BACKEND_DIR / ".env",
    REPO_ROOT / ".env",
]

# Keys whose value differs between two .env files are a silent-misconfiguration hazard.
SENSITIVE_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "VLM_API_KEY",
    "OPENROUTER_API_KEY",
    "VLLM_API_KEY",
    "API_SECRET_KEY",
}


class EnvSourceReport:
    """
    Phase 0 diagnostic: records which .env files were discovered, which keys each
    defines, and which keys collide with differing values.

    Prevents the historical bug where `load_dotenv(..., override=True)` let a
    CWD-relative .env silently win over the authoritative repo-root file.
    """

    def __init__(self) -> None:
        self.loaded: List[str] = []
        self.missing: List[str] = []
        self.key_sources: dict[str, str] = {}
        self.conflicts: List[dict] = []

    def build(self, candidates: List[Path]) -> "EnvSourceReport":
        seen: set = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue

            # The same file can be reached by two candidates (e.g. launching
            # from backend/ makes cwd/.env and BACKEND_DIR/.env identical).
            resolved_key = str(resolved).lower()
            if resolved_key in seen:
                continue

            if not resolved.is_file():
                self.missing.append(str(resolved))
                continue

            seen.add(resolved_key)
            self.loaded.append(str(resolved))
            try:
                values = dotenv_values(resolved)
            except Exception:
                continue

            for key, value in values.items():
                if value is None:
                    continue
                value = str(value)
                previous_source = self.key_sources.get(key)
                if previous_source is not None and previous_source != str(resolved):
                    if key in SENSITIVE_ENV_KEYS:
                        self.conflicts.append(
                            {
                                "key": key,
                                "authoritative_source": str(resolved),
                                "overridden_source": previous_source,
                                "severity": "critical",
                            }
                        )
                elif previous_source is None:
                    self.key_sources[key] = str(resolved)

        return self

    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    def to_dict(self) -> dict:
        return {
            "loaded_files": self.loaded,
            "missing_candidates": self.missing,
            "key_file_map": self.key_sources,
            "conflicts": self.conflicts,
            "precedence_order": "cwd < backend < repo_root (repo root wins)",
        }


env_source_report = EnvSourceReport().build(ENV_LOAD_ORDER)

# Apply the environment using the same lowest-to-highest precedence.
for env_candidate in ENV_LOAD_ORDER:
    if env_candidate.is_file():
        load_dotenv(dotenv_path=env_candidate, override=True)


def _split_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings(BaseSettings):
    """Application Configuration Settings."""

    model_config = SettingsConfigDict(
        env_file=[str(REPO_ROOT / ".env"), str(BACKEND_DIR / ".env")],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Cloud VLM Provider
    # ("openai_compatible", "anthropic", "mock")
    vlm_provider: Literal["openai_compatible", "anthropic", "mock"] = "openai_compatible"

    # Ordered model-level fallbacks for the primary provider (Phase 2).
    vlm_model_fallbacks: str = ""

    # Ordered provider-level failover chain (Phase 2).
    vlm_fallback_chain: str = "openai_compatible,anthropic"

    # API Keys
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-5"

    # OpenAI-compatible gateway settings (9Router / vLLM / OpenRouter / Azure OpenAI).
    # Every value comes from the environment (.env). Nothing is hardcoded, so
    # retargeting the gateway is a one-line .env edit, never a code change.
    vlm_api_base_url: Optional[str] = None
    vlm_api_key: Optional[str] = None
    vlm_openai_model: Optional[str] = None

    # Deterministic mock provider (tests / local staging only)
    mock_vlm_match: bool = False
    mock_vlm_confidence: float = 0.95
    mock_vlm_liveness: bool = True

    # Scoring Mode & Composite Acceptance Criteria
    scoring_mode: str = "field_average"  # "field_average" or "stage_weighted"
    composite_approval_threshold: float = 0.70
    strict_stage_pass_required: bool = False
    enforce_hard_rejections: bool = True

    # Field-level matching weights (used when scoring_mode == "field_average")
    field_weight_name: float = 1.0
    field_weight_dl_number: float = 1.0
    field_weight_dob: float = 1.0
    field_weight_expiry: float = 1.0
    field_weight_face: float = 1.0
    field_weight_plate: float = 1.0
    field_weight_color: float = 1.0

    # Decision & Fuzzy Matching Thresholds (Strict Binary: APPROVED or REJECTED)
    fuzzy_name_threshold: float = 0.85
    fuzzy_dl_threshold: float = 0.88
    # Field-level OCR gates, on the 0-100 similarity scale used by dl_ocr.
    fuzzy_dob_threshold: float = 80.0
    fuzzy_expiry_threshold: float = 80.0
    region_gate_score: float = 75.0
    severe_blur_variance: float = 20.0

    # Composite stage weights (used when scoring_mode == "stage_weighted").
    stage1_weight: float = 0.30
    stage2_weight: float = 0.45
    stage3_weight: float = 0.25
    stage3_plate_weight: float = 0.75
    stage3_color_weight: float = 0.25

    # Hard rejection floors (pipeline-level, independent of the fuzzy scores).
    vlm_mismatch_floor: float = 0.50
    plate_hard_floor: float = 0.40
    plate_review_floor: float = 0.85

    # Vehicle ALPR gates.
    plate_match_threshold: float = 0.85
    color_agreement_min: float = 0.15

    # Face detector sensitivity (YuNet + verification layer).
    face_yunet_score: float = 0.50
    face_yunet_nms: float = 0.30
    face_verify_threshold: float = 0.45
    face_nms_score: float = 0.40
    face_nms_iou: float = 0.35

    # OCR engine sensitivity (RapidOCR DBNet post-processing).
    ocr_box_thresh: float = 0.38
    ocr_db_thresh: float = 0.20
    ocr_unclip_ratio: float = 1.95
    # Working resolution cap for DL OCR passes. Measured on i7-1255U:
    # 1800px -> ~26s/pass, 1200px -> ~16s/pass with identical accuracy on
    # scored fields (name/number/dates are large print). Cards are deskewed
    # upright, so the angle classifier is dead weight when disabled.
    ocr_working_max_dim: int = 1200
    ocr_use_cls: bool = False
    # Recovery passes (upscale/binarize retries) only start inside this
    # stage-elapsed budget, so a slow image degrades to Pass-1 scores
    # instead of blowing the 45s pipeline SLA. Pass 1 on a large photo
    # costs ~20s, so budgets above that re-enable recovery for big inputs.
    stage1_recovery_budget_seconds: float = 10.0

    # Image quality gates (0 = unusable, higher variance = sharper).
    blur_variance_min: float = 40.0
    blurry_variance: float = 65.0
    glare_ratio_max: float = 0.30
    noise_variance_max: float = 5000.0

    # VLM call shaping (reasoning models may need a larger token budget).
    vlm_max_tokens: int = 900
    vlm_connectivity_max_tokens: int = 8
    vlm_temperature: float = 0
    vlm_repair_context_chars: int = 800
    vlm_use_response_format: bool = True

    # Uploads & previews.
    max_upload_mb: int = 15
    preview_max_dim: int = 1400
    telemetry_default_limit: int = 50
    telemetry_cache_ttl: float = 30.0

    # VLM decision thresholds
    vlm_min_match_confidence: float = 0.70

    # Privacy Face Cropping
    face_crop_padding_ratio: float = 0.15

    # Verification Pipeline Latency & Timeouts
    pipeline_timeout_seconds: float = 45.0
    # Per-provider attempt ceiling and the overall VLM budget. The startup guard
    # in app/main.py enforces vlm_total < pipeline_timeout * VLM_BUDGET_CEILING.
    vlm_call_timeout_seconds: float = 12.0
    vlm_total_timeout_seconds: float = 25.0
    vlm_parse_retry_count: int = 1

    # Storage Configuration
    uploads_dir: str = "storage/uploads"

    # Security & Rate Limiting
    api_secret_key: Optional[str] = None
    enable_api_key_auth: bool = False
    rate_limit_per_minute: int = 60
    rate_limit_window_seconds: float = 60.0
    rate_limit_max_ips: int = 2000
    cors_allow_origins: str = ""

    # Server Info & Deployment Configuration
    app_name: str = "Komüte Driver Verification Service v2"
    app_version: str = "2.0.0"
    environment: str = "production"
    host: str = "0.0.0.0"
    port: int = 8080

    # ---- Derived helpers -------------------------------------------------

    @property
    def uploads_root(self) -> Path:
        """Absolute storage root, independent of the process working directory."""
        candidate = Path(self.uploads_dir)
        if candidate.is_absolute():
            return candidate
        return (REPO_ROOT / candidate).resolve()

    @property
    def active_model(self) -> str:
        """Model identifier of the primary provider, for health/telemetry output."""
        if self.vlm_provider == "openai_compatible":
            return self.vlm_openai_model or "<VLM_OPENAI_MODEL not set>"
        if self.vlm_provider == "anthropic":
            return self.anthropic_model
        return self.vlm_provider

    @property
    def model_fallback_list(self) -> List[str]:
        return _split_csv(self.vlm_model_fallbacks)

    @property
    def provider_fallback_list(self) -> List[str]:
        chain = _split_csv(self.vlm_fallback_chain)
        # The configured provider always gets first refusal.
        if self.vlm_provider and self.vlm_provider not in chain:
            chain.insert(0, self.vlm_provider)
        return chain

    @property
    def resolved_cors_origins(self) -> List[str]:
        return _split_csv(self.cors_allow_origins)


# Global cached settings instance
settings = Settings()

# Stage weights must sum to 1.0 when running in stage_weighted mode.
if settings.scoring_mode.lower() == "stage_weighted":
    _weight_total = settings.stage1_weight + settings.stage2_weight + settings.stage3_weight
    if abs(_weight_total - 1.0) > 1e-6:
        raise ValueError(
            f"STAGE1/2/3_WEIGHT must sum to 1.0, got {_weight_total:.4f}. "
            "Fix the values in .env."
        )
    del _weight_total

# The VLM budget must leave headroom for Stage 1 OCR and Stage 3 ALPR inside
# pipeline_timeout_seconds, otherwise a slow VLM is misreported as a pipeline
# timeout. Ratio chosen so VLM can consume at most 60% of the pipeline budget.
VLM_BUDGET_CEILING = 0.6


def enforce_timeout_budget() -> dict:
    """Validate the VLM/pipeline timeout relationship. Returns a report dict."""
    ceiling = settings.pipeline_timeout_seconds * VLM_BUDGET_CEILING
    ok = settings.vlm_total_timeout_seconds <= ceiling
    report = {
        "pipeline_timeout_seconds": settings.pipeline_timeout_seconds,
        "vlm_total_timeout_seconds": settings.vlm_total_timeout_seconds,
        "vlm_call_timeout_seconds": settings.vlm_call_timeout_seconds,
        "max_allowed_vlm_total": round(ceiling, 2),
        "within_budget": ok,
        "clamped": False,
    }
    if not ok:
        settings.vlm_total_timeout_seconds = round(ceiling, 2)
        report["clamped"] = True
        report["vlm_total_timeout_seconds"] = settings.vlm_total_timeout_seconds
    return report
