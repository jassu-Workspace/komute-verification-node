# Komüte Driver Verification Service v2 — Postman Testing Guide

## 1. Setup & Configuration

### 1.1 Import Postman Collection

Create a new Collection in Postman named **"Komüte Driver Verification v2"**.

### 1.2 Environment Variables

Create a Postman Environment called `Komüte Local` with:

| Variable | Value |
|----------|-------|
| `base_url` | `http://localhost:8080` |
| `api_key` | `kmt_live_v2_f83jf98a2h4j90xj23` |

### 1.3 Global Headers

For every authenticated request, set these headers:

```
X-API-Key: {{api_key}}
Content-Type: application/json
```

---

## 2. API Endpoints — Full Reference

---

### 2.1 `GET /health`

**Purpose:** Primary health check for Render & cloud monitoring.

**Auth Required:** No

**Request:**
```
GET {{base_url}}/health
```

**Expected Response (200 OK):**
```json
{
  "status": "healthy",
  "app": "Komüte Driver Verification Service v2",
  "version": "2.0.0",
  "environment": "production",
  "uptime_seconds": 1234.56,
  "uptime_human": "20m 34s",
  "timestamp": "2026-09-20T10:30:00+00:00",
  "vlm_provider": "gemini",
  "vlm_model": "gemini-2.5-flash",
  "ocr_available": true,
  "face_detector_available": true,
  "checks": {
    "ocr_engine": true,
    "face_detector": true,
    "storage_writable": true,
    "memory_usage_mb": 245.32
  }
}
```

**Alternate Endpoints (same response):**
- `GET /api/v1/health`
- `GET /healthz`
- `GET /live`
- `GET /ready`
- `HEAD /health` (returns headers only, no body)

**Test Assertions:**
- [ ] Status code is `200`
- [ ] `status` is `"healthy"` or `"degraded"`
- [ ] `ocr_available` is `true`
- [ ] `face_detector_available` is `true`
- [ ] `checks.storage_writable` is `true`
- [ ] `uptime_seconds` > 0

---

### 2.2 `GET /ping`

**Purpose:** Ultra-fast ping probe for load balancers.

**Auth Required:** No

**Request:**
```
GET {{base_url}}/ping
```

**Expected Response (200 OK):**
```json
{
  "status": "ok",
  "ping": "pong",
  "timestamp": "2026-09-20T10:30:00+00:00",
  "app": "Komüte Driver Verification Service v2",
  "version": "2.0.0"
}
```

**Test Assertions:**
- [ ] Status code is `200`
- [ ] `ping` is `"pong"`

---

### 2.3 `POST /api/v1/verify` — Full 3-Stage Driver Verification

**Purpose:** Executes the complete verification pipeline — Stage 1 (DL OCR), Stage 2 (Face Biometrics), Stage 3 (ALPR & Color).

**Auth Required:** Yes (`X-API-Key`)

**Request:**
```
POST {{base_url}}/api/v1/verify
Headers:
  X-API-Key: {{api_key}}
  Content-Type: application/json
```

**Request Body (Raw JSON):**
```json
{
  "request_id": "req_test_001",
  "driver_id": "driver_test_001",
  "personal_info": {
    "full_name": "John Smith",
    "email_address": "john.smith@email.com",
    "mobile_number": "+14165551234",
    "date_of_birth": "1990-05-15",
    "gender": "Male",
    "role": "Driver"
  },
  "license_details": {
    "license_number": "DL1234567",
    "issuing_province": "ON",
    "license_expiry_date": "2028-12-31"
  },
  "vehicle_details": {
    "make": "Toyota",
    "model": "Camry",
    "year": 2022,
    "color": "White",
    "plate": "CMBY815"
  },
  "images": {
    "selfie_base64": "<BASE64_ENCODED_SELFIE_IMAGE>",
    "license_image_base64": "<BASE64_ENCODED_DL_IMAGE>",
    "vehicle_photo_base64": "<BASE64_ENCODED_VEHICLE_IMAGE>"
  }
}
```

**How to Encode Images in Postman:**
1. In the `images` fields, paste Base64 strings (without `data:image/...;base64,` prefix)
2. Or use the Pre-request Script below to read files:

**Pre-request Script (optional — for file upload):**
```javascript
// Convert binary image to base64 before sending
// Place image in collection variables after manual conversion
```

**Expected Response (200 OK) — APPROVED:**
```json
{
  "request_id": "req_test_001",
  "driver_id": "driver_test_001",
  "decision": "APPROVED",
  "composite_confidence": 0.92,
  "execution_time_ms": 4521.33,
  "stages": {
    "license_ocr": {
      "passed": true,
      "extracted_dl_number": "DL1234567",
      "extracted_name": "John Smith",
      "extracted_dob": "1990-05-15",
      "extracted_expiry": "2028-12-31",
      "extracted_province": "ON",
      "number_matched": true,
      "number_similarity": 1.0,
      "name_matched": true,
      "name_similarity": 0.95,
      "dob_matched": true,
      "is_expired": false,
      "driver_age_valid": true,
      "confidence": 0.95,
      "details": "Stage 1 PASSED: DL number matched, name matched, DOB matched, license not expired, age >= 18",
      "raw_ocr_lines": ["DRIVER LICENCE", "Ontario", "DL1234567", "SMITH/JOHN", "DOB: 1990-05-15", "EXP: 2028-12-31"],
      "missing_fields": [],
      "confidence_proof": {}
    },
    "face_biometrics": {
      "passed": true,
      "vlm_confidence": 0.88,
      "is_live": true,
      "is_match": true,
      "estimated_age_delta_years": 2,
      "privacy_face_cropped": true,
      "pii_leakage_prevented": true,
      "reasoning": "Face crops show consistent craniofacial features. Same person confirmed.",
      "verdict": "MATCH_CONFIRMED",
      "dl_face_detected": true,
      "selfie_face_detected": true,
      "confidence_proof": {}
    },
    "vehicle_verification": {
      "passed": true,
      "confidence": 0.91,
      "extracted_plate": "CMBY815",
      "plate_matched": true,
      "plate_similarity": 1.0,
      "color_matched": true,
      "detected_color": "White",
      "color_confidence": 0.87,
      "raw_ocr_lines": ["CMBY815"],
      "plate_box_detected": true,
      "details": "Stage 3 PASSED: Plate matched, color matched",
      "confidence_proof": {}
    }
  },
  "rejection_reasons": [],
  "manual_review_reasons": [],
  "saved_artifacts": {
    "session_directory": "storage/uploads/sessions/driver_test_001_req_test_001",
    "originals_folder": "...",
    "compressed_folder": "...",
    "cropped_folder": "...",
    "metadata_file": "..."
  },
  "composite_proof": {}
}
```

**Expected Response (200 OK) — REJECTED:**
```json
{
  "request_id": "req_test_002",
  "driver_id": "driver_test_002",
  "decision": "REJECTED",
  "composite_confidence": 0.35,
  "execution_time_ms": 3200.00,
  "stages": {
    "license_ocr": {
      "passed": false,
      "is_expired": true,
      "confidence": 0.40,
      "details": "Stage 1 FAILED: License has expired"
    },
    "face_biometrics": { "..." : "..." },
    "vehicle_verification": { "..." : "..." }
  },
  "rejection_reasons": [
    "Driving license has expired",
    "Face biometric verification failed"
  ],
  "manual_review_reasons": []
}
```

**Test Assertions:**
- [ ] Status code is `200`
- [ ] `decision` is `"APPROVED"` or `"REJECTED"`
- [ ] `composite_confidence` is between 0.0 and 1.0
- [ ] `execution_time_ms` > 0
- [ ] All 3 stages present in `stages` object
- [ ] `rejection_reasons` is populated when `decision` is `"REJECTED"`

---

### 2.4 `POST /api/v1/privacy-crop-preview` — Face Isolation Preview

**Purpose:** Tests local face detection and returns isolated face crop, verifying 0% text or card PII leakage.

**Auth Required:** Yes (`X-API-Key`)

**Request:**
```
POST {{base_url}}/api/v1/privacy-crop-preview
Headers:
  X-API-Key: {{api_key}}
  Content-Type: application/json
```

**Request Body:**
```json
{
  "image_base64": "<BASE64_ENCODED_DL_IMAGE>",
  "margin_ratio": 0.15
}
```

**Expected Response (200 OK) — Face Detected:**
```json
{
  "face_detected": true,
  "bounding_box": {
    "x": 120,
    "y": 45,
    "w": 85,
    "h": 100
  },
  "face_crop_base64": "<BASE64_ENCODED_CROPPED_FACE>",
  "pii_sanitized": true,
  "message": "Face portrait successfully isolated. All card text, numbers, and PII excluded.",
  "saved_artifacts": {
    "original": "storage/uploads/previews/prev_abc12345/original.webp",
    "compressed": "storage/uploads/previews/prev_abc12345/compressed.webp",
    "cropped": "storage/uploads/previews/prev_abc12345/cropped_face.webp"
  }
}
```

**Expected Response (200 OK) — No Face Detected:**
```json
{
  "face_detected": false,
  "bounding_box": null,
  "face_crop_base64": null,
  "pii_sanitized": false,
  "message": "No face detected in the provided image.",
  "saved_artifacts": null
}
```

**Test Assertions:**
- [ ] Status code is `200`
- [ ] `face_detected` is boolean
- [ ] When `face_detected` is `true`, `pii_sanitized` is `true`
- [ ] When `face_detected` is `true`, `face_crop_base64` is not null
- [ ] When `face_detected` is `false`, `message` contains "No face detected"

---

### 2.5 `POST /api/v1/compress` — Image Compression

**Purpose:** Compresses uploaded image to WebP/AVIF, strips EXIF metadata.

**Auth Required:** Yes (`X-API-Key`)

**Request:**
```
POST {{base_url}}/api/v1/compress
Headers:
  X-API-Key: {{api_key}}
  Content-Type: multipart/form-data
Body (form-data):
  file: <select image file>
  compressionPercentage: 80
  format: webp
  maxWidth: 1400
```

**Form-Data Parameters:**

| Key | Type | Value | Description |
|-----|------|-------|-------------|
| `file` | File | (select image) | Image to compress |
| `compressionPercentage` | Text | `80` | Compression level (1-100) |
| `format` | Text | `webp` | Output format: `webp` or `avif` |
| `maxWidth` | Text | `1400` | Max output width in pixels |

**Expected Response (200 OK):**
- Body: Binary image data (compressed WebP/AVIF)
- Headers:
  ```
  Content-Type: image/webp
  X-Original-Size: 2048576
  X-Compressed-Size: 312456
  X-Saved-Bytes: 1736120
  Content-Disposition: attachment; filename="compressed_image.webp"
  ```

**Test Assertions:**
- [ ] Status code is `200`
- [ ] Response Content-Type is `image/webp` or `image/avif`
- [ ] `X-Compressed-Size` < `X-Original-Size`
- [ ] `X-Saved-Bytes` > 0
- [ ] Response body is valid image binary

**Error Cases:**
- [ ] Non-image file → Status `400`, detail: `"Uploaded file is not a valid image."`
- [ ] File > 15MB → Status `413`, detail: `"File size exceeds 15 MB limit."`

---

### 2.6 `GET /api/v1/telemetry/system` — System Telemetry

**Purpose:** Returns system vitals, library versions, AI model health, and pipeline config.

**Auth Required:** No

**Request:**
```
GET {{base_url}}/api/v1/telemetry/system
```

**Expected Response (200 OK):**
```json
{
  "vitals": {
    "cpu_percent": 12.5,
    "memory_mb": 245.32,
    "disk_usage_percent": 45.2,
    "python_version": "3.11.0",
    "platform": "linux"
  },
  "libraries": {
    "rapidocr_onnxruntime": { "version": "1.3.0", "available": true },
    "opencv": { "version": "4.8.0", "available": true },
    "rapidfuzz": { "version": "3.0.0", "available": true },
    "numpy": { "version": "1.24.0", "available": true }
  },
  "ai_models": {
    "yunet_face_detector": { "loaded": true, "path": "backend/models/face_detection_yunet_2023mar.onnx" },
    "vlm_provider": "gemini",
    "vlm_model": "gemini-2.5-flash",
    "ocr_engine": "rapidocr_onnx"
  },
  "pipeline_config": {
    "fuzzy_name_threshold": 0.85,
    "fuzzy_dl_threshold": 0.88,
    "composite_approval_threshold": 0.85,
    "face_crop_padding_ratio": 0.15,
    "rate_limit_per_minute": 60
  }
}
```

**Test Assertions:**
- [ ] Status code is `200`
- [ ] `vitals.memory_mb` > 0
- [ ] `libraries` contains all key libraries
- [ ] `ai_models.yunet_face_detector.loaded` is `true`
- [ ] `pipeline_config` contains all thresholds

---

### 2.7 `GET /api/v1/telemetry/sessions` — Verification Sessions

**Purpose:** List recent verification sessions.

**Auth Required:** Yes (`X-API-Key`)

**Request:**
```
GET {{base_url}}/api/v1/telemetry/sessions?limit=10
Headers:
  X-API-Key: {{api_key}}
```

**Expected Response (200 OK):**
```json
{
  "total": 2,
  "sessions": [
    {
      "driver_id": "driver_test_001",
      "request_id": "req_test_001",
      "decision": "APPROVED",
      "timestamp": "2026-09-20T10:30:00+00:00",
      "execution_time_ms": 4521.33,
      "composite_confidence": 0.92
    },
    {
      "driver_id": "driver_test_002",
      "request_id": "req_test_002",
      "decision": "REJECTED",
      "timestamp": "2026-09-20T10:25:00+00:00",
      "execution_time_ms": 3200.00,
      "composite_confidence": 0.35
    }
  ]
}
```

**Test Assertions:**
- [ ] Status code is `200`
- [ ] `total` matches length of `sessions` array
- [ ] Each session has `driver_id`, `request_id`, `decision`

---

### 2.8 `GET /api/v1/telemetry/session/{driver_id}/{request_id}` — Session Detail

**Purpose:** Retrieve granular stage-by-stage audit for a specific session.

**Auth Required:** Yes (`X-API-Key`)

**Request:**
```
GET {{base_url}}/api/v1/telemetry/session/driver_test_001/req_test_001
Headers:
  X-API-Key: {{api_key}}
```

**Expected Response (200 OK):**
```json
{
  "driver_id": "driver_test_001",
  "request_id": "req_test_001",
  "decision": "APPROVED",
  "composite_confidence": 0.92,
  "stages": { "..." : "..." },
  "artifacts": {
    "originals": [...],
    "compressed": [...],
    "cropped": [...]
  },
  "metadata": { "..." : "..." }
}
```

**Test Assertions:**
- [ ] Status code is `200`
- [ ] Response contains `driver_id` and `request_id` matching request
- [ ] Contains `stages` breakdown
- [ ] Non-existent session → Status `404`

---

### 2.9 `POST /api/v1/telemetry/self-test` — Diagnostics Self-Test

**Purpose:** Run real-time benchmark self-tests across pipeline models.

**Auth Required:** Yes (`X-API-Key`)

**Request:**
```
POST {{base_url}}/api/v1/telemetry/self-test
Headers:
  X-API-Key: {{api_key}}
```

**Expected Response (200 OK):**
```json
{
  "face_privacy_cropper": {
    "status": "operational",
    "latency_ms": 45.2,
    "test_passed": true
  },
  "ocr_engine": {
    "status": "operational",
    "latency_ms": 120.5,
    "test_passed": true
  },
  "image_compressor": {
    "status": "operational",
    "latency_ms": 30.1,
    "test_passed": true
  }
}
```

**Test Assertions:**
- [ ] Status code is `200`
- [ ] All 3 subsystem tests return `test_passed: true`

---

## 3. Security Testing Scenarios

### 3.1 Missing API Key

**Request:**
```
POST {{base_url}}/api/v1/verify
(No X-API-Key header)
```

**Expected:** Status `401 Unauthorized`
```json
{
  "detail": "Invalid or missing X-API-Key header"
}
```

### 3.2 Wrong API Key

**Request:**
```
POST {{base_url}}/api/v1/verify
Headers:
  X-API-Key: wrong_key_12345
```

**Expected:** Status `401 Unauthorized`

### 3.3 Rate Limiting

**Request:** Send 61+ requests within 1 minute to any authenticated endpoint.

**Expected:** Status `429 Too Many Requests`
```json
{
  "detail": "Rate limit exceeded: maximum 60 requests per minute"
}
```

### 3.4 Path Traversal on Storage Endpoint

**Request:**
```
GET {{base_url}}/api/v1/storage/../../etc/passwd
Headers:
  X-API-Key: {{api_key}}
```

**Expected:** Status `403 Forbidden`
```json
{
  "detail": "Access denied: invalid file path."
}
```

---

## 4. Negative / Edge Case Testing

### 4.1 Expired License

**Request Body (verify):**
```json
{
  "request_id": "req_expired_001",
  "driver_id": "driver_expired_001",
  "personal_info": {
    "full_name": "Jane Doe",
    "date_of_birth": "1985-03-20"
  },
  "license_details": {
    "license_number": "DL9999999",
    "issuing_province": "ON",
    "license_expiry_date": "2020-01-01"
  },
  "vehicle_details": {
    "plate": "ABC1234",
    "color": "Blue"
  },
  "images": {
    "selfie_base64": "<BASE64>",
    "license_image_base64": "<BASE64>",
    "vehicle_photo_base64": "<BASE64>"
  }
}
```

**Expected:** `decision: "REJECTED"`, `rejection_reasons` contains `"Driving license has expired"`

### 4.2 No Face Detected

**Request Body (privacy-crop-preview):**
```json
{
  "image_base64": "<BASE64_ENCODED_IMAGE_WITH_NO_FACE>",
  "margin_ratio": 0.15
}
```

**Expected:** `face_detected: false`

### 4.3 Invalid Image Data

**Request Body (verify):**
```json
{
  "images": {
    "selfie_base64": "not_valid_base64!!!",
    "license_image_base64": "not_valid_base64!!!",
    "vehicle_photo_base64": "not_valid_base64!!!"
  }
}
```

**Expected:** Status `400` or `500` with error detail

### 4.4 Missing Required Fields

**Request Body:**
```json
{
  "request_id": "req_missing_001"
}
```

**Expected:** Status `422 Unprocessable Entity` with validation error details

---

## 5. Postman Test Scripts (Auto-Assertions)

Add these to the **Tests** tab of each request in Postman:

### Health Check Tests:
```javascript
pm.test("Status code is 200", () => {
  pm.response.to.have.status(200);
});

pm.test("Health status is healthy or degraded", () => {
  const data = pm.response.json();
  pm.expect(data.status).to.be.oneOf(["healthy", "degraded"]);
});

pm.test("OCR is available", () => {
  const data = pm.response.json();
  pm.expect(data.ocr_available).to.be.true;
});

pm.test("Face detector is available", () => {
  const data = pm.response.json();
  pm.expect(data.face_detector_available).to.be.true;
});
```

### Verify Endpoint Tests:
```javascript
pm.test("Status code is 200", () => {
  pm.response.to.have.status(200);
});

pm.test("Decision is APPROVED or REJECTED", () => {
  const data = pm.response.json();
  pm.expect(data.decision).to.be.oneOf(["APPROVED", "REJECTED"]);
});

pm.test("Composite confidence is between 0 and 1", () => {
  const data = pm.response.json();
  pm.expect(data.composite_confidence).to.be.within(0, 1);
});

pm.test("Execution time is positive", () => {
  const data = pm.response.json();
  pm.expect(data.execution_time_ms).to.be.above(0);
});

pm.test("All 3 stages are present", () => {
  const data = pm.response.json();
  pm.expect(data.stages).to.have.property("license_ocr");
  pm.expect(data.stages).to.have.property("face_biometrics");
  pm.expect(data.stages).to.have.property("vehicle_verification");
});

pm.test("Rejection reasons provided when rejected", () => {
  const data = pm.response.json();
  if (data.decision === "REJECTED") {
    pm.expect(data.rejection_reasons).to.be.an("array").that.is.not.empty;
  }
});
```

### Privacy Crop Preview Tests:
```javascript
pm.test("Status code is 200", () => {
  pm.response.to.have.status(200);
});

pm.test("Face detected is boolean", () => {
  const data = pm.response.json();
  pm.expect(data.face_detected).to.be.a("boolean");
});

pm.test("PII sanitized when face detected", () => {
  const data = pm.response.json();
  if (data.face_detected) {
    pm.expect(data.pii_sanitized).to.be.true;
    pm.expect(data.face_crop_base64).to.not.be.null;
  }
});
```

### Auth Failure Tests:
```javascript
pm.test("Returns 401 for invalid API key", () => {
  pm.response.to.have.status(401);
});

pm.test("Error message about API key", () => {
  const data = pm.response.json();
  pm.expect(data.detail).to.include("API-Key");
});
```

---

## 6. Testing Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    POSTMAN TESTING FLOW                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Health Check (No Auth)                                  │
│     GET /health → 200 OK, all subsystems true               │
│                                                              │
│  2. Ping (No Auth)                                          │
│     GET /ping → 200 OK, "pong"                              │
│                                                              │
│  3. System Telemetry (No Auth)                              │
│     GET /api/v1/telemetry/system → 200 OK                   │
│                                                              │
│  4. Privacy Crop Preview (Auth Required)                    │
│     POST /api/v1/privacy-crop-preview → 200 OK             │
│     Verify: face_detected, pii_sanitized                    │
│                                                              │
│  5. Full Verification (Auth Required)                       │
│     POST /api/v1/verify → 200 OK                           │
│     Verify: decision, confidence, stages                    │
│                                                              │
│  6. Image Compression (Auth Required)                       │
│     POST /api/v1/compress → 200 OK (binary)                │
│     Verify: X-Compressed-Size < X-Original-Size            │
│                                                              │
│  7. Session History (Auth Required)                         │
│     GET /api/v1/telemetry/sessions → 200 OK                │
│                                                              │
│  8. Self-Test (Auth Required)                               │
│     POST /api/v1/telemetry/self-test → 200 OK              │
│     Verify: all test_passed = true                          │
│                                                              │
│  9. Security Tests                                          │
│     - No API Key → 401                                      │
│     - Wrong API Key → 401                                   │
│     - Rate Limit → 429                                      │
│     - Path Traversal → 403                                  │
│                                                              │
│  10. Negative Tests                                         │
│      - Expired license → REJECTED                           │
│      - No face detected → face_detected: false              │
│      - Invalid base64 → 400/500                             │
│      - Missing fields → 422                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Quick Reference — All Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | No | Health check |
| `HEAD` | `/health` | No | Health check (headers only) |
| `GET` | `/ping` | No | Ping probe |
| `POST` | `/api/v1/verify` | Yes | Full 3-stage verification |
| `POST` | `/api/v1/privacy-crop-preview` | Yes | Face isolation preview |
| `POST` | `/api/v1/compress` | Yes | Image compression |
| `GET` | `/api/v1/telemetry/system` | No | System telemetry |
| `GET` | `/api/v1/telemetry/sessions` | Yes | List sessions |
| `GET` | `/api/v1/telemetry/session/{driver_id}/{request_id}` | Yes | Session detail |
| `POST` | `/api/v1/telemetry/self-test` | Yes | Diagnostics self-test |
| `GET` | `/api/v1/storage/{file_path}` | Yes | Serve artifacts |

---

## 8. Base64 Image Helper

To convert test images to Base64 for Postman:

**Windows PowerShell:**
```powershell
$imagePath = "path\to\image.png"
$base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($imagePath))
$base64 | Out-File -FilePath "base64_output.txt"
```

**Python:**
```python
import base64
with open("image.png", "rb") as f:
    print(base64.b64encode(f.read()).decode())
```

**Online Tools:**
- https://www.base64-image.de/
- https://codebeautify.org/image-to-base64-converter

---

*Generated for Komüte Driver Verification Service v2 — Postman Testing Guide*
