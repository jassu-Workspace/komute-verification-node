import importlib
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)

# Track process startup time
SERVICE_START_TIME = time.time()


def format_uptime(seconds: float) -> str:
    """Format seconds into human readable uptime string."""
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if mins > 0:
        parts.append(f"{mins}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def format_bytes(size_bytes: int) -> str:
    """Format bytes into KB, MB, GB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


class TelemetryService:
    """
    Central telemetry & diagnostics service.
    Collects system vitals, library dependencies, AI/ML model health, and session audits.

    Phase 1: caches now expire on a TTL. Previously `_cached_models` / `_cached_libraries`
    / `_cached_config` were latched for the process lifetime, so a VLM that recovered
    (or a key that was fixed) still reported "Not Configured" until a restart.
    """

    #: Seconds a computed telemetry snapshot stays fresh (overridable via .env).
    CACHE_TTL_SECONDS = settings.telemetry_cache_ttl

    def __init__(self) -> None:
        self._cached_libraries = None
        self._cached_models = None
        self._cached_config = None
        self._cache_stamps: Dict[str, float] = {}
        self._storage_vitals_cache = {"time": 0, "total_files": 0, "total_size": 0}

    def invalidate_caches(self) -> None:
        """Force the next telemetry read to recompute from live state."""
        self._cached_libraries = None
        self._cached_models = None
        self._cached_config = None
        self._cache_stamps = {}

    def _is_fresh(self, key: str) -> bool:
        return time.time() - self._cache_stamps.get(key, 0.0) < self.CACHE_TTL_SECONDS

    def _store(self, key: str, value: Any) -> Any:
        self._cache_stamps[key] = time.time()
        return value

    def get_system_vitals(self) -> Dict[str, Any]:
        """Collect host machine, OS, memory, and storage metrics."""
        now = time.time()
        uptime_sec = now - SERVICE_START_TIME

        # Memory & Process Info
        memory_rss_mb = 0.0
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            memory_rss_mb = round(mem_info.rss / (1024 * 1024), 1)
            cpu_percent = process.cpu_percent(interval=None)
        except Exception:
            cpu_percent = 0.0
            memory_rss_mb = 0.0

        # Disk & Storage usage for uploads/ (Cached for 30s to prevent IO bottlenecks)
        uploads_path = Path(getattr(settings, "uploads_dir", "uploads")).resolve()
        
        if now - self._storage_vitals_cache["time"] > 30:
            total_files = 0
            total_size = 0
            if uploads_path.exists():
                for p in uploads_path.rglob("*"):
                    if p.is_file():
                        total_files += 1
                        try:
                            total_size += p.stat().st_size
                        except Exception:
                            pass
            self._storage_vitals_cache = {"time": now, "total_files": total_files, "total_size": total_size}
            
        total_files = self._storage_vitals_cache["total_files"]
        total_size = self._storage_vitals_cache["total_size"]

        return {
            "application": {
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
                "uptime_seconds": round(uptime_sec, 1),
                "uptime_formatted": format_uptime(uptime_sec),
                "started_at": datetime.fromtimestamp(SERVICE_START_TIME, timezone.utc).isoformat(),
            },
            "runtime": {
                "python_version": platform.python_version(),
                "python_compiler": platform.python_compiler(),
                "python_executable": sys.executable,
                "os": platform.system(),
                "os_release": platform.release(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
                "process_pid": os.getpid(),
                "memory_rss_mb": memory_rss_mb,
                "cpu_percent": cpu_percent,
            },
            "storage": {
                "uploads_root": str(uploads_path),
                "total_stored_files": total_files,
                "total_stored_size_bytes": total_size,
                "total_stored_size_formatted": format_bytes(total_size),
            },
        }

    def get_libraries_status(self) -> List[Dict[str, Any]]:
        """Check installed/active status of all runtime libraries."""
        if self._cached_libraries is not None and self._is_fresh("libraries"):
            return self._cached_libraries

        libraries = []

        # PyTorch
        try:
            import torch
            libraries.append({
                "name": "PyTorch",
                "package": "torch",
                "installed": True,
                "version": torch.__version__,
                "status": "active",
                "details": f"CUDA Available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}, Threads: {torch.get_num_threads()}",
            })
        except ImportError:
            libraries.append({"name": "PyTorch", "package": "torch", "installed": False, "version": None, "status": "missing", "details": "Not installed"})

        # OpenCV
        try:
            libraries.append({
                "name": "OpenCV",
                "package": "cv2",
                "installed": True,
                "version": cv2.__version__,
                "status": "active",
                "details": f"DNN Backend: Enabled, Threads: {cv2.getNumThreads()}, OpenCL: {cv2.ocl.haveOpenCL()}",
            })
        except Exception:
            libraries.append({"name": "OpenCV", "package": "cv2", "installed": False, "version": None, "status": "error", "details": "Error querying OpenCV"})

        # ONNX Runtime
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            libraries.append({
                "name": "ONNX Runtime",
                "package": "onnxruntime",
                "installed": True,
                "version": ort.__version__,
                "status": "active",
                "details": f"Execution Providers: {', '.join(providers)}",
            })
        except ImportError:
            libraries.append({"name": "ONNX Runtime", "package": "onnxruntime", "installed": False, "version": None, "status": "missing", "details": "Not installed"})

        # RapidOCR (ONNX Runtime)
        try:
            import rapidocr_onnxruntime
            libraries.append({
                "name": "RapidOCR ONNX",
                "package": "rapidocr-onnxruntime",
                "installed": True,
                "version": getattr(rapidocr_onnxruntime, "__version__", "1.2+"),
                "status": "active",
                "details": "PP-OCRv4 ONNX Runtime Engine (Detection DBNet + Recognition CRNN)",
            })
        except ImportError:
            libraries.append({"name": "RapidOCR ONNX", "package": "rapidocr-onnxruntime", "installed": False, "version": None, "status": "missing", "details": "Not installed"})

        # Pillow (PIL)
        try:
            import PIL
            from PIL import features
            webp_support = features.check("webp")
            libraries.append({
                "name": "Pillow (PIL)",
                "package": "Pillow",
                "installed": True,
                "version": PIL.__version__,
                "status": "active",
                "details": f"WebP Compression: {webp_support}, Transcoding: Enabled",
            })
        except Exception:
            libraries.append({"name": "Pillow", "package": "Pillow", "installed": False, "version": None, "status": "missing", "details": "Not installed"})

        # OpenAI-compatible gateway transport
        try:
            import httpx
            from core.vlm_diagnostics import mask_secret

            gateway_key = settings.vlm_api_key
            libraries.append({
                "name": "OpenAI-compatible Gateway (httpx)",
                "package": "httpx",
                "installed": True,
                "version": getattr(httpx, "__version__", "unknown"),
                "status": "active" if gateway_key else "standby",
                "details": (
                    f"Base URL: {settings.vlm_api_base_url or 'Not Configured'}, "
                    f"API Key: {mask_secret(gateway_key) or 'Not Configured'}, "
                    f"Model: {settings.vlm_openai_model or 'Not Configured'}"
                ),
            })
        except ImportError:
            libraries.append({"name": "OpenAI-compatible Gateway (httpx)", "package": "httpx", "installed": False, "version": None, "status": "missing", "details": "Not installed"})

        # Anthropic SDK
        try:
            import anthropic
            from core.vlm_diagnostics import mask_secret

            anthropic_key = settings.anthropic_api_key
            libraries.append({
                "name": "Anthropic SDK",
                "package": "anthropic",
                "installed": True,
                "version": getattr(anthropic, "__version__", "0.40+"),
                "status": "active" if anthropic_key else "standby",
                "details": f"API Key: {mask_secret(anthropic_key) or 'Not Configured'}, Model: {settings.anthropic_model}",
            })
        except ImportError:
            libraries.append({"name": "Anthropic SDK", "package": "anthropic", "installed": False, "version": None, "status": "missing", "details": "Not installed"})

        # OpenAI-compatible gateway (vLLM / OpenRouter / Azure OpenAI)
        try:
            import httpx
            from core.vlm_diagnostics import mask_secret

            oai_key = settings.vlm_api_key
            libraries.append({
                "name": "OpenAI-compatible Gateway",
                "package": "httpx",
                "installed": True,
                "version": httpx.__version__,
                "status": "active" if (oai_key and settings.vlm_api_base_url) else "standby",
                "details": (
                    f"API Key: {mask_secret(oai_key) or 'Not Configured'}, "
                    f"Base URL: {settings.vlm_api_base_url or 'Not Configured'}, "
                    f"Model: {settings.vlm_openai_model or 'Not Configured'}"
                ),
            })
        except ImportError:
            libraries.append({"name": "OpenAI-compatible Gateway", "package": "httpx", "installed": False, "version": None, "status": "missing", "details": "Not installed"})

        # FastAPI & Uvicorn
        try:
            import fastapi
            import uvicorn
            libraries.append({
                "name": "FastAPI / Uvicorn",
                "package": "fastapi",
                "installed": True,
                "version": f"FastAPI {fastapi.__version__} / Uvicorn {uvicorn.__version__}",
                "status": "active",
                "details": "Asynchronous High-Throughput REST Gateway with Swagger OpenAPI",
            })
        except Exception:
            pass

        self._cached_libraries = libraries
        self._store("libraries", libraries)
        return libraries

    def get_ai_models_status(self) -> List[Dict[str, Any]]:
        """Check live loaded status of all AI/ML models in the verification pipeline."""
        if self._cached_models is not None and self._is_fresh("models"):
            return self._cached_models

        from core.face_privacy_cropper import face_privacy_cropper

        models = []

        # 1. YuNet libfacedetection DNN
        yunet_loaded = face_privacy_cropper.yunet_detector is not None
        models.append({
            "id": "yunet_dnn",
            "name": "YuNet Face Detector (OpenCV DNN)",
            "role": "Stage 2: Facial Biometrics",
            "ready": yunet_loaded,
            "status": "Loaded & Operational" if yunet_loaded else "Offline / Standby",
            "model_path": "OpenCV FaceDetectorYN Built-in / face_detection_yunet_2023mar.onnx",
            "architecture": "Ultra-lightweight Deep Neural Network (libfacedetection)",
            "output": "5-Point Facial Geometry (Eyes, Nose, Mouth Corners) + Presence Confidence",
        })

        # 4. RapidOCR ONNX Engine
        ocr_ready = False
        try:
            from core.ocr_engine import shared_ocr
            _ = shared_ocr.get_engine()
            ocr_ready = True
        except Exception:
            pass

        models.append({
            "id": "rapidocr_onnx",
            "name": "RapidOCR ONNX Runtime Engine",
            "role": "Stage 1 & 3: DL OCR & Vehicle ALPR",
            "ready": ocr_ready,
            "status": "Loaded & Operational" if ocr_ready else "Offline / Standby",
            "model_path": "rapidocr_onnxruntime / PP-OCRv4 ONNX",
            "architecture": "DBNet (Detection) + MobileNetV3 CRNN (Recognition)",
            "output": "Bounding Quadrilaterals, Recognized Unicode Text, Confidence Scores",
        })

        # 5. Cloud VLM Biometric Verifier (Phase 1: real readiness, not just key presence)
        from core.vlm_diagnostics import vlm_diagnostics, mask_secret

        vlm_probe = vlm_diagnostics.probe_all()
        vlm_ready = bool(vlm_probe.get("vlm_ready"))
        shape_mismatches = vlm_probe.get("key_shape_mismatches", [])
        last_error = vlm_probe.get("last_error") or {}
        usable = vlm_probe.get("usable_providers", [])

        models.append({
            "id": "cloud_vlm",
            "name": f"Cloud VLM Biometric Engine ({settings.vlm_provider.upper()})",
            "role": "Stage 2: Cross-Image Forensic Biometrics",
            "ready": vlm_ready,
            "status": (
                f"Ready via {', '.join(usable)} (model: {settings.active_model})"
                if vlm_ready
                else "Fail-Safe Standby (Strict Rejection)"
            ),
            "model_path": f"Cloud Endpoint (Provider: {settings.vlm_provider})",
            "architecture": "Multimodal Vision-Language Model (OpenAI-compatible gateway / Claude)",
            "output": "Craniofacial Structural Alignment, Anti-Spoof Liveness, Forensic Match Score",
            "provider_fallback_chain": settings.provider_fallback_list,
            "model_fallbacks": settings.model_fallback_list,
            "key_shape_mismatches": shape_mismatches,
            "last_error_reason": last_error.get("reason"),
            "last_error_detail": last_error.get("detail"),
            "last_error_provider": last_error.get("provider"),
        })



        # 7. Image Compression & Transcoding Engine
        models.append({
            "id": "image_compression_engine",
            "name": "Adaptive WebP/AVIF Compression Engine",
            "role": "Storage & Archival Optimization",
            "ready": True,
            "status": "Loaded & Operational",
            "model_path": "OpenCV LibWebP / Pillow Transcoder",
            "architecture": "Lossy/Lossless WebP & AVIF Quantization + Lanczos4 Downsampling",
            "output": "Optimized WebP Assets (70-90% Storage Reduction, Zero EXIF Leakage)",
        })

        self._cached_models = models
        self._store("models", models)
        return models

    def get_pipeline_configuration(self) -> Dict[str, Any]:
        """Return active verification pipeline thresholds and security parameters."""
        if self._cached_config is not None and self._is_fresh("config"):
            return self._cached_config

        config = {
            "thresholds": {
                "face_crop_padding_ratio": f"{int(settings.face_crop_padding_ratio * 100)}%",
                "min_face_confidence": settings.face_verify_threshold,
                "vlm_match_confidence_threshold": settings.vlm_min_match_confidence,
                "ocr_name_similarity_threshold": settings.fuzzy_name_threshold,
                "ocr_dl_number_similarity_threshold": settings.fuzzy_dl_threshold,
                "ocr_dob_similarity_threshold": settings.fuzzy_dob_threshold,
                "ocr_expiry_similarity_threshold": settings.fuzzy_expiry_threshold,
                "vehicle_plate_similarity_threshold": settings.plate_match_threshold,
                "scoring_mode": settings.scoring_mode,
                "composite_approval_threshold": settings.composite_approval_threshold,
                "strict_stage_pass_required": settings.strict_stage_pass_required,
                "enforce_hard_rejections": settings.enforce_hard_rejections,
                "field_weights": {
                    "name": settings.field_weight_name,
                    "dl_number": settings.field_weight_dl_number,
                    "dob": settings.field_weight_dob,
                    "expiry": settings.field_weight_expiry,
                    "face": settings.field_weight_face,
                    "plate": settings.field_weight_plate,
                    "color": settings.field_weight_color,
                },
                "stage_weights": {
                    "stage_1_dl_ocr": settings.stage1_weight,
                    "stage_2_face_biometrics": settings.stage2_weight,
                    "stage_3_vehicle_alpr": settings.stage3_weight,
                },
            },
            "security": {
                "rate_limiting_enabled": settings.rate_limit_per_minute > 0,
                "rate_limit_requests_per_minute": settings.rate_limit_per_minute,
                "rate_limit_window_seconds": settings.rate_limit_window_seconds,
                "api_key_auth_enabled": settings.enable_api_key_auth,
                "pii_sanitization_enforced": True,
                "deferred_license_compression": True,
            },
            "models": {
                "primary_vlm_provider": settings.vlm_provider,
                "primary_vlm_model": settings.active_model,
                # Derived from live config instead of a hardcoded, drifting list.
                "vlm_provider_fallback_chain": settings.provider_fallback_list,
                "vlm_model_fallbacks": settings.model_fallback_list,
                "anthropic_model": settings.anthropic_model,
                "vlm_api_base_url": settings.vlm_api_base_url,
                "vlm_openai_model": settings.vlm_openai_model,
            },
            "timeouts": {
                "pipeline_timeout_seconds": settings.pipeline_timeout_seconds,
                "vlm_total_timeout_seconds": settings.vlm_total_timeout_seconds,
                "vlm_call_timeout_seconds": settings.vlm_call_timeout_seconds,
            },
        }
        self._cached_config = config
        self._store("config", config)
        return config

    def run_engine_self_test(self) -> Dict[str, Any]:
        """Execute on-demand live benchmark self-tests for local AI components."""
        results = {}

        # 1. Test Face Privacy Cropper (RetinaFace / YuNet)
        try:
            from core.face_privacy_cropper import face_privacy_cropper
            dummy_face = np.full((300, 300, 3), 200, dtype=np.uint8)
            cv2.ellipse(dummy_face, (150, 150), (60, 80), 0, 0, 360, (180, 200, 230), -1)
            cv2.circle(dummy_face, (130, 130), 6, (50, 50, 50), -1)
            cv2.circle(dummy_face, (170, 130), 6, (50, 50, 50), -1)
            cv2.ellipse(dummy_face, (150, 175), (20, 10), 0, 0, 180, (50, 50, 50), 2)

            t0 = time.perf_counter()
            crop, bbox, sanitized = face_privacy_cropper.extract_isolated_face(dummy_face, margin_ratio=0.15)
            dt = (time.perf_counter() - t0) * 1000.0

            results["face_privacy_cropper"] = {
                "status": "PASSED" if crop is not None else "STANDBY",
                "latency_ms": round(dt, 2),
                "isolated_bbox": bbox,
                "sanitized": sanitized,
            }
        except Exception as e:
            results["face_privacy_cropper"] = {"status": "FAILED", "error": str(e)}

        # 2. Test Image Compression Engine
        try:
            from core.image_utils import compress_image_bytes
            test_img = np.random.randint(0, 256, (800, 800, 3), dtype=np.uint8)
            _, raw_jpg = cv2.imencode(".jpg", test_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            raw_bytes = raw_jpg.tobytes()

            t0 = time.perf_counter()
            comp_bytes, mime, orig_sz, comp_sz = compress_image_bytes(raw_bytes, out_format="webp", max_dim=800, quality=80)
            dt = (time.perf_counter() - t0) * 1000.0
            ratio = round((1.0 - comp_sz / float(orig_sz)) * 100, 1) if orig_sz > 0 else 0.0

            results["image_compression_engine"] = {
                "status": "PASSED",
                "latency_ms": round(dt, 2),
                "original_bytes": orig_sz,
                "compressed_bytes": comp_sz,
                "saved_percentage": f"{ratio}%",
                "output_mime": mime,
            }
        except Exception as e:
            results["image_compression_engine"] = {"status": "FAILED", "error": str(e)}

        # 3. Test OCR Engine (RapidOCR ONNX)
        try:
            from core.ocr_engine import shared_ocr
            t0 = time.perf_counter()
            ocr = shared_ocr.get_engine()
            dummy_img = np.full((64, 256, 3), 255, dtype=np.uint8)
            cv2.putText(dummy_img, "TEST", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            res, _ = ocr(dummy_img)
            dt = (time.perf_counter() - t0) * 1000.0
            results["rapidocr_engine"] = {
                "status": "PASSED" if res else "FAILED",
                "init_latency_ms": round(dt, 2),
                "device": "CPU / ONNX",
                "sample_recognition": res[0][1] if (res and len(res[0]) >= 2) else "",
            }
        except Exception as e:
            results["rapidocr_engine"] = {"status": "FAILED", "error": str(e)}

        # 4. Test ALPR Color Classifier
        try:
            from core.vehicle_alpr import vehicle_alpr_engine
            sample_veh = np.full((200, 200, 3), (240, 240, 240), dtype=np.uint8)  # White
            t0 = time.perf_counter()
            _, color_name, color_conf = vehicle_alpr_engine.detect_vehicle_color(sample_veh, "White")
            dt = (time.perf_counter() - t0) * 1000.0
            results["alpr_color_classifier"] = {
                "status": "PASSED",
                "latency_ms": round(dt, 2),
                "detected_color": color_name,
                "color_confidence": color_conf,
            }
        except Exception as e:
            results["alpr_color_classifier"] = {"status": "FAILED", "error": str(e)}

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "ALL_PASSED" if all(r.get("status") == "PASSED" for r in results.values()) else "DEGRADED",
            "components": results,
        }


# Global telemetry service singleton
telemetry_service = TelemetryService()
