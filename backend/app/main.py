import os

# Clamp CPU threads globally before any C-extensions (NumPy, OpenCV, OpenMP, ONNX Runtime) initialize
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["ORT_INTRA_OP_NUM_THREADS"] = "2"
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.config import settings
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
    logger.info(f"VLM Provider: {settings.vlm_provider} (Model: {settings.vlm_model})")

    # Warm up local models in background
    try:
        from core.ocr_engine import shared_ocr
        shared_ocr.get_engine()
        logger.info("Local OCR and CV engines pre-warmed successfully.")
    except Exception as e:
        logger.warning(f"Could not pre-warm models: {e}")

    yield
    logger.info("Shutting down Komüte Verifier Service...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-Grade Driver Identity Verification with Zero-PII Cloud Biometrics & Vehicle ALPR",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    svg_icon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🛡️</text></svg>'
    from fastapi.responses import Response
    return Response(content=svg_icon, media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", getattr(settings, "port", 8080)))
    host = os.environ.get("HOST", getattr(settings, "host", "0.0.0.0"))
    reload_mode = os.environ.get("ENVIRONMENT", "production").lower() not in ["production", "prod"]
    logger.info(f"Launching server on {host}:{port} (reload={reload_mode})...")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload_mode)

