# API Reference — Komüte Driver Verifier v2

> Complete endpoint contract. Base URL local: `http://localhost:8080`. Production: `https://<service>.onrender.com`. Interactive docs: `/docs`.

Auth: when `ENABLE_API_KEY_AUTH=true`, send `X-API-Key: <API_SECRET_KEY>` on all `Auth: Yes` endpoints. Rate limit: 60 req/min/IP → `429`.

## 1. `GET /health` (no auth)

Primary health probe. Aliases: `GET /api/v1/health`, `/healthz`, `/live`, `/ready`. `HEAD /health` returns headers only.

```bash
curl -i http://localhost:8080/health
```

Response `200` (`healthy` | `degraded`):

```json
{
  "status": "healthy",
  "app": "Komüte Driver Verification Service v2",
  "version": "2.0.0",
  "environment": "production",
  "uptime_seconds": 128.45,
  "uptime_human": "2m 8s",
  "timestamp": "2026-08-29T11:45:00.000000Z",
  "vlm_provider": "gemini",
  "vlm_model": "gemini-2.5-flash",
  "ocr_available": true,
  "face_detector_available": true,
  "checks": { "ocr_engine": true, "face_detector": true, "storage_writable": true, "memory_usage_mb": 115.4 }
}
```

## 2. `GET /ping` (no auth)

LB-friendly probe.

```bash
curl http://localhost:8080/ping
# {"status":"ok","ping":"pong","timestamp":"...","app":"...","version":"2.0.0"}
```

## 3. `POST /api/v1/verify` (auth) — Full 3-stage verification

```bash
curl -X POST http://localhost:8080/api/v1/verify \
  -H "Content-Type: application/json" -H "X-API-Key: $API_SECRET_KEY" \
  -d '{
    "request_id": "req_komute_887219",
    "driver_id": "drv_jaswanth_001",
    "personal_info": {
      "full_name": "Jaswanth Sri Sai Venkat Dangeti",
      "email_address": "driver@komute.com",
      "mobile_number": "+918500923656",
      "date_of_birth": "2005-12-14",
      "gender": "Male", "role": "Driver"
    },
    "license_details": {
      "license_number": "DDNPV1331Q",
      "issuing_province": "ON",
      "license_expiry_date": "2030-12-14"
    },
    "vehicle_details": {
      "make": "Hyundai", "model": "Venue", "year": 2020,
      "color": "Black", "plate": "DL 7CQ 1939"
    },
    "images": {
      "selfie_base64": "<BASE64>",
      "license_image_base64": "<BASE64>",
      "vehicle_photo_base64": "<BASE64>"
    }
  }'
```

Notes:

- Aliases accepted: `expiry_date` ↔ `license_expiry_date`, `plate_number` ↔ `plate`. Missing province defaults to `ON`, missing expiry to `2030-01-01`.
- Images: raw base64 with or without `data:image/...;base64,` prefix. Corrupt payload → `REJECTED` with `rejection_reasons: ["Invalid or corrupted image payload..."]`.
- Success `200` shape (`schemas.py:VerificationResponse`):

```json
{
  "request_id": "req_komute_887219",
  "driver_id": "drv_jaswanth_001",
  "decision": "APPROVED",
  "composite_confidence": 0.954,
  "execution_time_ms": 1420.5,
  "stages": {
    "license_ocr": { "passed": true, "extracted_dl_number": "DDNPV1331Q", "number_similarity": 1.0, "name_similarity": 0.98, "is_expired": false, "driver_age_valid": true, "confidence": 0.98 },
    "face_biometrics": { "passed": true, "vlm_confidence": 0.96, "is_live": true, "is_match": true, "verdict": "MATCH_CONFIRMED", "pii_leakage_prevented": true },
    "vehicle_verification": { "passed": true, "extracted_plate": "DL7CQ1939", "plate_matched": true, "color_matched": true, "detected_color": "Black" }
  },
  "rejection_reasons": [],
  "manual_review_reasons": [],
  "composite_proof": { "formula": "(0.30 * Stage_1_OCR) + (0.45 * Stage_2_Biometrics) + (0.25 * Stage_3_Vehicle)" }
}
```

- Errors: `422` missing fields, `401` bad API key, `429` rate limit, `500` pipeline failure.

## 4. `POST /api/v1/privacy-crop-preview` (auth)

Audit Zero-PII isolation without running full verification.

```bash
curl -X POST http://localhost:8080/api/v1/privacy-crop-preview \
  -H "Content-Type: application/json" -H "X-API-Key: $API_SECRET_KEY" \
  -d '{"image_base64":"<BASE64_DL>","margin_ratio":0.15}'
```

- Face found → `{face_detected:true, bounding_box:{x,y,w,h}, face_crop_base64:"...", pii_sanitized:true, saved_artifacts:{...}}`
- No face → `{face_detected:false, bounding_box:null, face_crop_base64:null, pii_sanitized:false}`

## 5. `POST /api/v1/compress` (auth, multipart)

```bash
curl -X POST http://localhost:8080/api/v1/compress \
  -H "X-API-Key: $API_SECRET_KEY" \
  -F "file=@/path/to/image.jpg" -F "compressionPercentage=80" -F "format=webp" -F "maxWidth=1400" \
  --output compressed.webp -D headers.txt
```

- `format`: `webp` | `avif`. Response headers: `X-Original-Size`, `X-Compressed-Size`, `X-Saved-Bytes`.
- Errors: `400` non-image, `413` >15 MB.

## 6. Telemetry (system + sessions)

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/telemetry/system` | No | `{vitals, libraries, ai_models, pipeline_config}` |
| `GET` | `/api/v1/telemetry/sessions?limit=50` | Yes | `{total, sessions[]}` |
| `GET` | `/api/v1/telemetry/session/{driver_id}/{request_id}` | Yes | Full stage audit + artifacts; `404` if missing |
| `POST` | `/api/v1/telemetry/self-test` | Yes | `{face_privacy_cropper, ocr_engine, image_compressor}` live benchmarks |

```bash
curl http://localhost:8080/api/v1/telemetry/system
curl -H "X-API-Key: $API_SECRET_KEY" "http://localhost:8080/api/v1/telemetry/sessions?limit=10"
```

## 7. `GET /api/v1/storage/{file_path}` (auth)

Streams saved artifacts. Traversal-protected (`403` on `../` escape, `404` if missing).

## 8. Dashboards (no auth, HTML)

- `GET /` and `GET /dashboard` — verification + privacy audit UI
- `GET /system` and `GET /telemetry` — vitals, library matrix, inspector

## 9. Postman

Import `Komute_Driver_Verification_v2.postman_collection.json` + `Komute_Local.postman_environment.json` (`base_url`, `api_key`). Full scripted assertions live in `POSTMAN_TESTING_GUIDE.md`.
