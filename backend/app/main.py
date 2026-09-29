import os

# Clamp CPU threads globally before any C-extensions (NumPy, OpenCV, OpenMP, ONNX Runtime) initialize
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")
os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.config import enforce_timeout_budget, env_source_report, settings
from app.routes import router as api_router
from core.face_privacy_cropper import face_privacy_cropper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")
    logger.info(f"VLM Provider: {settings.vlm_provider} (Model: {settings.active_model})")

    # --- Phase 0: env source + provider readiness reporting -----------------
    env_report = env_source_report.to_dict()
    logger.info(
        "Env files loaded (precedence %s): %s",
        env_report["precedence_order"],
        env_report["loaded_files"] or "none found",
    )
    for conflict in env_report["conflicts"]:
        logger.error(
            "ENV CONFLICT: '%s' is defined in both %s and %s. The repo-root file wins. "
            "This silently overrides credentials and is the most common cause of a "
            "mysteriously unconfigured VLM.",
            conflict["key"],
            conflict["overridden_source"],
            conflict["authoritative_source"],
        )

    budget = enforce_timeout_budget()
    if budget["clamped"]:
        logger.warning(
            "VLM total timeout exceeded its budget: clamped %ss -> %ss (ceiling is %s%% of the %ss pipeline timeout).",
            budget["vlm_total_timeout_seconds"],
            settings.vlm_total_timeout_seconds,
            int(budget["max_allowed_vlm_total"] / max(budget["pipeline_timeout_seconds"], 1e-9) * 100),
            budget["pipeline_timeout_seconds"],
        )
    else:
        logger.info(
            "Timeout budget OK: VLM %ss of a %ss pipeline ceiling.",
            budget["vlm_total_timeout_seconds"],
            budget["pipeline_timeout_seconds"],
        )

    from core.vlm_diagnostics import vlm_diagnostics

    probe = vlm_diagnostics.probe_all()
    logger.info("VLM readiness: ready=%s effective=%s chain=%s",
                probe["vlm_ready"], probe["effective_provider"], probe["provider_fallback_chain"])
    for entry in probe["providers"]:
        if entry["ready"]:
            logger.info("  [OK]   %-18s model=%-24s key=%s", entry["provider"], entry["model"], entry["key_masked"])
        else:
            logger.error("  [FAIL] %-18s reason=%s | %s", entry["provider"], entry["blocking_reason"], entry["key_shape_hint"])
    if probe["key_shape_mismatches"]:
        logger.warning(
            "VLM key shape hint for %s: the credential does not look like it belongs to "
            "that provider. This is ADVISORY only - gateways and proxies legitimately "
            "issue keys of other shapes, and the shape warning is suppressed once a "
            "provider returns a valid verdict. Confirm with GET /api/v1/diagnostics/vlm?live=true.",
            probe["key_shape_mismatches"],
        )
    if not probe["vlm_ready"]:
        logger.error(
            "No VLM provider is usable. POST /api/v1/verify will return HTTP 200 with "
            "decision=REJECTED, service_status='degraded' and Stage 2 verdict "
            "SERVICE_UNAVAILABLE. Set a valid credential or point VLM_FALLBACK_CHAIN at a "
            "working provider."
        )

    # Warm up local models in background
    try:
        from core.ocr_engine import shared_ocr
        shared_ocr.get_engine()
        logger.info("Local OCR and CV engines pre-warmed successfully.")
    except Exception as e:
        logger.warning(f"Could not pre-warm models: {e}")

    yield

    # Phase 4: release the shared HTTP connection pool.
    try:
        from core.vlm_providers import close_shared_httpx_client

        await close_shared_httpx_client()
    except Exception as e:  # noqa: BLE001 - shutdown must not raise
        logger.debug(f"Shared HTTP client close skipped: {type(e).__name__}")

    logger.info("Shutting down KomÃ¼te Verifier Service...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-Grade Driver Identity Verification with Zero-PII Cloud Biometrics & Vehicle ALPR",
    lifespan=lifespan,
)

# CORS Middleware (Phase 7)
# NOTE: the previous configuration combined allow_origins=["*"] with
# allow_credentials=True, which is an unsafe pairing. Credentials are now only
# enabled when an explicit origin allowlist is configured.
_allowed_origins = settings.resolved_cors_origins
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS restricted to %d configured origin(s).", len(_allowed_origins))
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["Content-Type"],
    )
    logger.warning(
        "CORS_ALLOW_ORIGINS is not set. Cross-origin browser requests are disabled. "
        "Set it to a comma-separated allowlist to enable the dashboard from another host."
    )

# Include API Router
app.include_router(api_router)


def _get_frontend_path(filename: str) -> str:
    """Safely resolve frontend files across local dev and Render container paths."""
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend", filename)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend", filename)),
        os.path.abspath(os.path.join(os.getcwd(), "frontend", filename)),
        os.path.abspath(os.path.join(os.getcwd(), "../frontend", filename)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
@app.head("/", include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
@app.head("/dashboard", include_in_schema=False)
async def get_dashboard() -> HTMLResponse:
    """Serve the interactive driver verification and privacy audit dashboard."""
    path = _get_frontend_path("dashboard.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})



@app.get("/system", response_class=HTMLResponse, tags=["Telemetry"], include_in_schema=False)
@app.get("/telemetry", response_class=HTMLResponse, tags=["Telemetry"], include_in_schema=False)
async def get_system_page() -> HTMLResponse:
    """Serve the deep system vitals, AI library matrix, and model response inspector subpage."""
    path = _get_frontend_path("system.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    """Return empty or SVG favicon to avoid 404 in browser."""
    svg_icon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">ðŸ›¡ï¸</text></svg>'
    from fastapi.responses import Response
    return Response(content=svg_icon, media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", getattr(settings, "port", 8080)))
    host = os.environ.get("HOST", getattr(settings, "host", "0.0.0.0"))
    reload_mode = os.environ.get("ENVIRONMENT", "production").lower() not in ["production", "prod"]
    logger.info(f"Launching server on {host}:{port} (reload={reload_mode})...")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload_mode)

