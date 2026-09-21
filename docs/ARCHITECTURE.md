# Architecture — Komüte Driver Verifier v2

> System design, pipeline flow, component responsibilities, and key design decisions for production reviewers and maintainers.

## 1. Purpose

Komüte Driver Verifier v2 is a FastAPI microservice for driver onboarding. It verifies:

1. **Who the driver is** — Driving Licence OCR + profile cross-check (Stage 1)
2. **That the person holding the licence is live and matching** — Zero-PII face isolation + cloud VLM biometrics (Stage 2)
3. **That the vehicle is legitimate** — ALPR plate extraction + color verification (Stage 3)

Final output is a strict binary decision: `APPROVED` or `REJECTED`, with a weighted composite confidence score and full audit proof.

## 2. High-Level Architecture

```
Client App (Komüte / onboarding UI)
        |
        |  POST /api/v1/verify (JSON + base64 images)
        |  X-API-Key + Rate Limit
        v
FastAPI (backend/app/main.py + routes.py)
  ├── schemas.py      — Pydantic v2 request/response contracts
  ├── security.py     — API-key auth + sliding-window rate limiter
  ├── config.py       — env-driven Settings (VLM, thresholds, storage)
        |
        v
VerificationPipeline (backend/core/pipeline.py)
  ├── Stage 1: dl_ocr.py + ocr_engine.py + canadian_dl_grammar.py
  ├── Stage 2A: face_privacy_cropper.py (YuNet ONNX, local only)
  ├── Stage 2B: vlm_face_verifier.py (Gemini / Anthropic / mock)
  ├── Stage 3: vehicle_alpr.py (OpenCV + RapidOCR + color classifier)
  └── shared: image_utils.py (base64, EXIF, CLAHE, WebP compression)
        |
        |  BackgroundTasks
        v
StorageManager (backend/storage/storage.py)
  storage/uploads/sessions/{driver_id}_{request_id}/
    ├── originals/   — raw decoded images
    ├── compressed/  — WebP normalized images
    ├── cropped/     — dl_face, vehicle_plate crops
    └── session_metadata.json
        |
        v
Telemetry (backend/core/telemetry_service.py)
  GET /api/v1/telemetry/system | /sessions | /session/{driver}/{request} | POST /self-test
```

Frontend dashboards (`frontend/dashboard.html`, `frontend/system.html`) are served directly by FastAPI at `/`, `/dashboard`, `/system`.

## 3. Request Lifecycle (`POST /api/v1/verify`)

Source: `backend/core/pipeline.py:execute_verification`

1. **Decode** — `decode_base64_to_image` for selfie, licence, vehicle. Licence is flattened (`flatten_id_card`) to correct perspective.
2. **Normalize** — selfie + vehicle compressed to WebP (max 1200px) for fast inference.
3. **Privacy isolation (Stage 2A, local)** — YuNet detects face on DL card, crops tightly with `face_crop_padding_ratio` (default 0.15). Selfie face is also cropped to strip background. Runs in `asyncio.to_thread` (CPU-bound).
4. **Parallel stage execution** (20s global timeout):
   - Stage 1 OCR (`dl_ocr_engine.verify_license`)
   - Stage 2B VLM (`vlm_face_verifier.verify_biometrics` — only face crops leave the host)
   - Stage 3 ALPR (`vehicle_alpr_engine.verify_vehicle`)
5. **Composite decision** — `_evaluate_composite_decision`:
   - Formula: `(0.30 × Stage1) + (0.45 × Stage2) + (0.25 × Stage3)`
   - Stage2 score = 0 if `is_live == False`
   - Stage3 score = `(plate_similarity × 0.75) + (0.25 if color_matched)`
   - `APPROVED` only if **all stages passed AND composite ≥ threshold (default 0.85) AND zero rejection triggers**. Otherwise `REJECTED`.
6. **Persistence** — licence WebP archival only on `APPROVED` (privacy minimization). All session images + metadata saved via `BackgroundTasks` so API latency is unaffected.
7. **Response** — `VerificationResponse` with per-stage breakdown, `rejection_reasons`, `composite_proof`, `execution_time_ms`.

## 4. Component Reference

| Component | File | Responsibility |
|---|---|---|
| Entrypoint | `backend/app/main.py` | Lifespan, CORS, dashboard routes, thread clamping (`OMP_NUM_THREADS=2`, `ORT_*`) |
| Routes | `backend/app/routes.py` | `/verify`, `/privacy-crop-preview`, `/compress`, `/health`, `/ping`, `/telemetry/*`, `/storage/*` |
| Schemas | `backend/app/schemas.py` | `VerificationRequest/Response`, `LicenseOcrResult`, `FaceBiometricsResult`, `VehicleVerificationResult` |
| DL OCR | `backend/core/dl_ocr.py` | RapidOCR ONNX + RapidFuzz (name ≥0.85, DL ≥0.88), age ≥18, expiry check |
| OCR engine | `backend/core/ocr_engine.py` | Shared RapidOCR singleton |
| DL grammar | `backend/core/canadian_dl_grammar.py` | Province/class/expiry parsing rules |
| Face cropper | `backend/core/face_privacy_cropper.py` | YuNet ONNX (`backend/models/face_detection_yunet_2023mar.onnx`), tight crop, `pii_sanitized` flag |
| VLM verifier | `backend/core/vlm_face_verifier.py` | Gemini / Anthropic / mock provider switch |
| Vehicle ALPR | `backend/core/vehicle_alpr.py` | Contour plate detection, OCR confusion normalization (`O↔0`, `I↔1`, `B↔8`…), HSV/K-Means color |
| Image utils | `backend/core/image_utils.py` | Base64, EXIF fix, CLAHE, WebP/AVIF compression (15 MB cap) |
| Storage | `backend/storage/storage.py` | Session/preview persistence, traversal-safe serving |
| Telemetry | `backend/core/telemetry_service.py` | CPU/RAM/disk, lib versions, model health, pipeline config, self-test |
| Security | `backend/app/security.py` | Optional `X-API-Key` (constant-time), per-IP 60 req/min sliding window, 2000-IP bounded cache |

## 5. Key Design Decisions

1. **Zero-PII cloud boundary** — full licence image never leaves the host. Only 256px-class face crops are sent to Gemini/Claude. Auditable via `POST /privacy-crop-preview`.
2. **Local-first CV** — OCR, face detection, ALPR run on CPU (OpenCV-headless + ONNX Runtime) for cost, latency, and offline resilience.
3. **Strict binary decision** — no `MANUAL_REVIEW` state in v2 schemas; anything below bar is `REJECTED` with explicit reasons. Simplifies downstream automation.
4. **Async + thread offload** — FastAPI async handlers, CPU work in `to_thread`, 20s gather timeout, storage in `BackgroundTasks` to protect p99 latency.
5. **Thread clamping** — `OMP/OPENBLAS/MKL/ORT` limited to 1–2 threads in `main.py` and `render.yaml` to survive 512 MB Render Free instances.
6. **Privacy-minimized storage** — rejected licence archival skipped; previews/sessions namespaced by `driver_id/request_id`.

## 6. Scaling Notes

- Stateless API; horizontal scale behind any L7 LB using `/health` + `/ping`.
- In-memory rate limiter is per-instance — use Redis (e.g. `slowapi` + Redis) when running >1 replica (see `docs/OPERATIONS_RUNBOOK.md`).
- ONNX + YuNet are CPU-bound; for throughput, raise Gunicorn/Uvicorn workers and set `ORT_INTRA_OP_NUM_THREADS` per vCPU.
- VLM calls dominate latency — cache by image hash for retries, set provider timeouts, and use `VLM_PROVIDER=mock` in load tests.
