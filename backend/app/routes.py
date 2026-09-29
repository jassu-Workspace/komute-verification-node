import os
from pathlib import Path
import time
from datetime import datetime, timezone
import logging
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from app.config import settings
from app.schemas import (
    HealthResponse,
    SubsystemChecks,
    PrivacyCropPreviewRequest,
    PrivacyCropPreviewResponse,
    VerificationRequest,
    VerificationResponse,
    VlmDiagnosticsResponse,
)

from app.security import check_rate_limit, verify_api_key
from core.face_privacy_cropper import face_privacy_cropper
from core.image_utils import (
    compress_and_normalize_base64,
    decode_base64_to_image,
    encode_image_to_base64,
)
from core.pipeline import pipeline_engine

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/api/v1/verify",
    response_model=VerificationResponse,
    summary="Full 3-Stage Driver Verification",
    description="Executes Stage 1 License OCR, Stage 2 Zero-PII Face Biometrics, and Stage 3 ALPR & Color Verification.",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def verify_driver(
    request: VerificationRequest,
    background_tasks: BackgroundTasks,
) -> VerificationResponse:
    """Execute end-to-end multi-stage driver verification pipeline."""
    try:
        response = await pipeline_engine.execute_verification(request, background_tasks=background_tasks)
        return response
    except Exception as e:
        logger.exception(f"Unhandled error in /verify: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification pipeline failed: {str(e)}",
        )


@router.post(
    "/api/v1/privacy-crop-preview",
    response_model=PrivacyCropPreviewResponse,
    summary="Zero-PII Face Isolation Preview",
    description="Tests local face detection and returns isolated face crop, verifying 0% text or card PII leakage.",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def privacy_crop_preview(request: PrivacyCropPreviewRequest) -> PrivacyCropPreviewResponse:
    """Isolate driver face from card image to audit zero-PII sanitization."""
    try:
        img_bgr = decode_base64_to_image(request.image_base64)
        crop_bgr, bbox, sanitized = face_privacy_cropper.extract_isolated_face(
            img_bgr=img_bgr,
            margin_ratio=request.margin_ratio,
        )

        if crop_bgr is None:
            return PrivacyCropPreviewResponse(
                face_detected=False,
                bounding_box=None,
                face_crop_base64=None,
                pii_sanitized=False,
                message="No face detected in the provided image.",
            )

        comp_bgr, _ = compress_and_normalize_base64(request.image_base64, out_format="webp", max_dim=settings.preview_max_dim)
        crop_b64 = encode_image_to_base64(crop_bgr, format_ext=".webp")
        import uuid
        preview_id = f"prev_{uuid.uuid4().hex[:8]}"
        from storage.storage import storage_manager
        saved_storage = storage_manager.save_preview_images(
            preview_id=preview_id,
            original_img=img_bgr,
            cropped_img=crop_bgr,
            compressed_img=comp_bgr,
        )

        return PrivacyCropPreviewResponse(
            face_detected=True,
            bounding_box=bbox,
            face_crop_base64=crop_b64,
            pii_sanitized=sanitized,
            message="Face portrait successfully isolated. All card text, numbers, and PII excluded.",
            saved_artifacts=saved_storage,
        )
    except Exception as e:
        logger.error(f"Error in privacy crop preview: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image: {str(e)}",
        )


@router.post(
    "/api/v1/compress",
    summary="Compress Image to WebP/AVIF",
    description="Compresses any uploaded image to WebP/AVIF, strips EXIF metadata, downsizes if requested, and optimizes storage/bandwidth.",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def compress_image_endpoint(
    file: UploadFile = File(...),
    compressionPercentage: int = Form(80),
    format: str = Form("webp"),
    maxWidth: int = Form(settings.preview_max_dim),
):
    """
    Compress an uploaded image file into lightweight WebP or AVIF format.
    Reduces disk and DB storage and lowers network transmission latency.
    """
    from fastapi import Response
    from core.image_utils import compress_image_bytes

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        )

    # Maximum file size comes from .env (MAX_UPLOAD_MB)
    MAX_FILE_SIZE = settings.max_upload_mb * 1024 * 1024
    input_bytes = await file.read()
    if len(input_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {settings.max_upload_mb} MB limit.",
        )

    quality = max(1, min(100, 100 - compressionPercentage))
    try:
        compressed_bytes, mime_type, orig_sz, comp_sz = compress_image_bytes(
            input_bytes=input_bytes,
            out_format=format,
            max_dim=maxWidth,
            quality=quality,
        )
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image compression failed: {str(e)}",
        )

    filename = (file.filename or "image").rsplit(".", 1)[0]
    out_ext = mime_type.split("/")[-1]
    headers = {
        "Content-Disposition": f'attachment; filename="compressed_{filename}.{out_ext}"',
        "X-Original-Size": str(orig_sz),
        "X-Compressed-Size": str(comp_sz),
        "X-Saved-Bytes": str(orig_sz - comp_sz),
        "Cache-Control": "public, max-age=31536000, immutable",
    }

    return Response(content=compressed_bytes, media_type=mime_type, headers=headers)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Primary Service Health Check & Node Status",
    tags=["Health & Diagnostics"],
)
@router.head("/health", include_in_schema=False)
@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="API v1 Service Health Status",
    tags=["Health & Diagnostics"],
)
@router.head("/api/v1/health", include_in_schema=False)
@router.get("/healthz", response_model=HealthResponse, include_in_schema=False)
async def health_check() -> HealthResponse:
    """
    Production-grade Health Check Node for Render & Cloud Monitoring.
    Returns real-time health status, uptime, subsystem readiness, and memory usage.

    NOTE: a VLM outage marks this endpoint "degraded". Container/orchestrator
    liveness probes must use /live or /ping instead, otherwise a cloud-provider
    blip will cause the service to be killed and restarted.
    """
    from core.telemetry_service import SERVICE_START_TIME, format_uptime

    # 1. OCR Subsystem Readiness
    ocr_avail = True
    try:
        from core.ocr_engine import shared_ocr
        _ = shared_ocr.get_engine()
    except Exception:
        ocr_avail = False

    # 2. Face Detector Subsystem Readiness
    face_avail = face_privacy_cropper.yunet_detector is not None

    # 3. Storage Subsystem Readiness
    storage_writable = True
    try:
        try:
            from storage.storage import storage_manager
        except ImportError:
            from storage import storage_manager
        storage_writable = os.path.exists(storage_manager.base_dir) and os.access(storage_manager.base_dir, os.W_OK)
    except Exception:
        from pathlib import Path
        uploads_path = Path(getattr(settings, "uploads_dir", "storage/uploads")).resolve()
        uploads_path.mkdir(parents=True, exist_ok=True)
        storage_writable = os.path.exists(uploads_path) and os.access(uploads_path, os.W_OK)

    # 4. Memory Diagnostics
    memory_mb = None
    try:
        import psutil
        process = psutil.Process()
        memory_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        pass

    # 5. Cloud VLM Readiness (Phase 0/1)
    # Previously /health reported "healthy" while Stage 2 was entirely dead,
    # because VLM was never part of the health computation.
    from core.vlm_diagnostics import vlm_diagnostics
    vlm_probe = vlm_diagnostics.probe_all()
    vlm_ready = bool(vlm_probe.get("vlm_ready"))
    vlm_configured = bool(vlm_probe.get("usable_providers")) or any(
        p.get("configured") for p in vlm_probe.get("providers", [])
    )
    vlm_last_error = vlm_probe.get("last_error")
    vlm_key_shape_ok = not vlm_probe.get("key_shape_mismatches")

    now_ts = datetime.now(timezone.utc).isoformat()
    uptime_sec = round(time.time() - SERVICE_START_TIME, 2)
    uptime_str = format_uptime(uptime_sec)

    is_healthy = ocr_avail and face_avail and storage_writable and vlm_ready
    status_str = "healthy" if is_healthy else "degraded"

    return HealthResponse(
        status=status_str,
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=uptime_sec,
        uptime_human=uptime_str,
        timestamp=now_ts,
        vlm_provider=settings.vlm_provider,
        vlm_model=settings.active_model,
        vlm_configured=vlm_configured,
        vlm_ready=vlm_ready,
        vlm_key_shape_valid=vlm_key_shape_ok,
        vlm_last_error=(vlm_last_error or {}).get("detail") if vlm_last_error else None,
        vlm_last_error_at=None,
        ocr_available=ocr_avail,
        face_detector_available=face_avail,
        checks=SubsystemChecks(
            ocr_engine=ocr_avail,
            face_detector=face_avail,
            storage_writable=storage_writable,
            memory_usage_mb=memory_mb,
        ),
    )


@router.get("/api/v1/diagnostics/vlm",
    response_model=VlmDiagnosticsResponse,
    summary="VLM Configuration & Readiness Diagnostics",
    description=(
        "Reports which VLM providers are configured, whether each credential has a "
        "plausible shape, which .env files were loaded, and the classified reason for "
        "the most recent failure. Use live=true to send one minimal connectivity probe "
        "per ready provider."
    ),
    tags=["Health & Diagnostics"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def vlm_diagnostics_endpoint(live: bool = False):
    """
    Phase 0: turn 'VLM is not configured' into an actionable diagnosis.
    Offline by default so it can be polled cheaply.
    """
    from core.vlm_diagnostics import vlm_diagnostics

    report = vlm_diagnostics.probe_all()
    if live:
        report["live"] = await vlm_diagnostics.live_probe()
    return report


@router.get("/live", response_model=HealthResponse, include_in_schema=False)
@router.head("/live", include_in_schema=False)
@router.get("/ready", response_model=HealthResponse, include_in_schema=False)
@router.head("/ready", include_in_schema=False)
async def liveness_check() -> HealthResponse:
    """
    Liveness / readiness probe for orchestrators.

    Deliberately excludes the cloud VLM from the status computation: a third-party
    provider outage must not cause the container to be killed and restarted.
    Use /health (not /live) to assert subsystem health.
    """
    from core.telemetry_service import SERVICE_START_TIME, format_uptime
    from core.vlm_diagnostics import vlm_diagnostics

    ocr_avail = True
    try:
        from core.ocr_engine import shared_ocr
        _ = shared_ocr.get_engine()
    except Exception:
        ocr_avail = False

    face_avail = face_privacy_cropper.yunet_detector is not None

    storage_writable = True
    try:
        from storage.storage import storage_manager

        storage_writable = os.path.exists(storage_manager.base_dir) and os.access(
            storage_manager.base_dir, os.W_OK
        )
    except Exception:
        storage_writable = True

    is_healthy = ocr_avail and face_avail and storage_writable

    vlm_probe = vlm_diagnostics.probe_all()

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - SERVICE_START_TIME, 2),
        uptime_human=format_uptime(round(time.time() - SERVICE_START_TIME, 2)),
        timestamp=datetime.now(timezone.utc).isoformat(),
        vlm_provider=settings.vlm_provider,
        vlm_model=settings.active_model,
        vlm_configured=bool(vlm_probe.get("usable_providers")),
        vlm_ready=bool(vlm_probe.get("vlm_ready")),
        vlm_key_shape_valid=not vlm_probe.get("key_shape_mismatches"),
        ocr_available=ocr_avail,
        face_detector_available=face_avail,
        checks=SubsystemChecks(
            ocr_engine=ocr_avail,
            face_detector=face_avail,
            storage_writable=storage_writable,
        ),
    )


@router.get(
    "/ping",
    summary="Lightweight Ping Probe",
    tags=["Health & Diagnostics"],
)
async def ping():
    """Ultra-fast ping probe returning 200 OK for load balancers and uptime bots."""
    return {
        "status": "ok",
        "ping": "pong",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "app": settings.app_name,
        "version": settings.app_version,
    }



@router.get(
    "/api/v1/telemetry/system",
    summary="System Vitals, Libraries & AI Model Matrix",
    description="Returns detailed segregated telemetry on hardware vitals, installed libraries, AI models, and active pipeline configuration.",
)
async def get_system_telemetry():
    """Retrieve segregated system status, library versions, and AI model health."""
    from core.telemetry_service import telemetry_service
    return {
        "vitals": telemetry_service.get_system_vitals(),
        "libraries": telemetry_service.get_libraries_status(),
        "ai_models": telemetry_service.get_ai_models_status(),
        "pipeline_config": telemetry_service.get_pipeline_configuration(),
    }


@router.get(
    "/api/v1/telemetry/sessions",
    summary="Past Verification Sessions & Audits",
    description="List recent driver verification sessions with decision metadata and file metrics.",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def get_verification_sessions(limit: int = settings.telemetry_default_limit):
    """List recent verification sessions."""
    from storage.storage import storage_manager
    sessions = storage_manager.list_verification_sessions(limit=limit)
    return {"total": len(sessions), "sessions": sessions}


@router.get(
    "/api/v1/telemetry/session/{driver_id}/{request_id}",
    summary="Deep Verification Session Audit & Model Inspector",
    description="Retrieve granular stage-by-stage outputs, raw model responses, and generated artifacts for a session.",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def get_session_detail(driver_id: str, request_id: str):
    """Retrieve detailed session audit including stage results, VLM outputs, and files."""
    from storage.storage import storage_manager
    detail = storage_manager.get_session_details(driver_id=driver_id, request_id=request_id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification session for driver '{driver_id}' and request '{request_id}' not found.",
        )
    return detail


@router.post(
    "/api/v1/telemetry/self-test",
    summary="Live Diagnostics & Model Self-Test",
    description="Executes instant live benchmarks across Face Privacy Cropper, EasyOCR, and Image Compressor.",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def run_diagnostics_self_test():
    """Run real-time benchmark self-tests across pipeline models."""
    from core.telemetry_service import telemetry_service
    return telemetry_service.run_engine_self_test()


@router.get(
    "/api/v1/storage/{file_path:path}",
    summary="Serve Saved Verification Artifacts",
    description="Securely streams saved originals, compressed WebP images, and cropped faces.",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def get_storage_file(file_path: str):
    """Serve image files from uploads directory with traversal protection."""
    from pathlib import Path
    from fastapi.responses import FileResponse

    base_dir = settings.uploads_root
    target_file = (base_dir / file_path).resolve()

    # Prevent directory traversal.
    # NOTE: the historical check used str(target).startswith(str(base_dir)),
    # which accepts sibling directories such as "storage/uploads_evil/".
    try:
        target_file.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: invalid file path.")

    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail="Requested artifact image not found.")

    ext = target_file.suffix.lower()
    mime_map = {
        ".webp": "image/webp",
        ".avif": "image/avif",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".json": "application/json",
    }
    media_type = mime_map.get(ext, "application/octet-stream")
    return FileResponse(path=str(target_file), media_type=media_type)

