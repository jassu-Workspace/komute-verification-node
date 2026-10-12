# Komüte Driver Verifier v2 🚗🛡️

### Production-Grade FastAPI Microservice for Driver Onboarding — Zero-PII Cloud Biometrics + ALPR

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Privacy](https://img.shields.io/badge/Privacy-GDPR%20%7C%20DPDP%20%7C%20PIPEDA-purple.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

> Verifies **who the driver is** (licence OCR & provincial grammar), **that they are live and matching** (face biometrics), and **that their vehicle is legitimate** (whole-plate ALPR + exterior color) — in a single `POST /api/v1/verify` call with a strict `APPROVED / REJECTED` verdict.

---

## ✨ Why this service?

- **3-stage pipeline in one call** — DL OCR (Fine-Tuned RapidOCR + Canadian Provincial Grammar + AAMVA Spatial Layout) → Face Isolation (Local YuNet ONNX, 15% tight margin) + Cloud VLM (Gemini / Claude gateway) → Vehicle ALPR (Whole-Plate Matching) + Exterior Color Analysis.
- **Zero-PII cloud boundary** — Full licence image never leaves your host. Zero names, card numbers, barcodes, or addresses transmitted to cloud neural models. Only isolated, cropped driver faces are sent to the VLM. Audit live via `/privacy-crop-preview`.
- **Explainable decisions** — Weighted composite `(0.30×OCR + 0.45×Face + 0.25×Vehicle)` or 7-field active average, field-by-field expected vs extracted audit tables, and full mathematical `composite_proof` in every response.
- **Production-ready** — API-key auth with dynamic dashboard detection, per-IP rate limiting, hardware concurrency governor, session artifact audits, Docker + Render Blueprint deploys.
- **Drop-in integration** — Proxy from your backend, switch on `decision`, done. See `docs/INTEGRATION_GUIDE.md`.

---

## 🛡️ Recent Engine Hardening & Production Updates (v2.1)

### 1. True DL Card Name Extraction & Address Leakage Prevention
- **Eliminated User Input Mirroring**: Previously, when an entered name mismatched, the response echoed the user's input back to `extracted_name`. The system now extracts and returns the **actual name found on the card** (e.g. `DOE JOHN`) or `None` (rendered in the UI as `Not detected`), never echoing submitted input.
- **Address & Header Line Isolation**: Name candidate generation now strictly filters out street address patterns (e.g., `123 ANY STREET`), street suffixes (`STREET`, `AVE`, `BLVD`, `RD`), Canadian postal codes, and city names. Street addresses can no longer contaminate legal name candidates.
- **Bilingual AAMVA Tag Anchoring**: Expanded spatial AAMVA recognition to parse Canadian/Ontario multi-lingual header confusions (`NAWELNOM`, `PNAWEINOM`, `1. NOM`, `1. NAME/NOM`). Vertically pairs Tag 1 (Surname) with the line directly beneath it (Given Name) for 100% clean name extraction.

### 2. Whole-Vehicle License Plate ALPR (Zero False Positives)
- **Length Coverage Enforcement ($Coverage \ge 80\%$)**: Eliminated unpenalized substring matching where short fragments (e.g. 4-character `L7CQ` matching 9-character `DL7CQ1939` at 95%) falsely passed. Substring fragments are now penalized by Levenshtein distance, dropping `L7CQ` vs `DL7CQ1939` to **44.4% (REJECTED)**.
- **Smart Candidate Selection & Noise Stripping**: ALPR candidate selection now prioritizes exact target plate matches and minimal length difference $|len(cand) - len(target)|$. When a license plate is embedded in surrounding text (e.g. dealer frames, provincial tags, or trailing words like `AND`), the engine cleanly isolates the exact plate identifier (`DL7CQ1939`), stripping noise words like `DL7CQ1939AND`.

### 3. Interactive Developer Dashboard & API Key UX
- **Linear/Geist Dark Mode Redesign**: Modernized UI with monospace telemetry indicators, high-contrast badges, and real-time biometric and ALPR crop inspectors.
- **Dynamic Auth Detection (`/api/v1/auth/status`)**: Automatically probes whether API key auth is enabled on the host, displays a safe key prefix hint, securely caches the key in `localStorage`, and injects `X-API-Key` headers on all verification requests.
- **Audit Comparison Tables**: Real-time table rendering: `Field`, `Entered (Expected)`, `Extracted (Found)`, and `Status` with live similarity percentage badges.

### 4. Hardware Governor & Concurrency Scaling
- **Queue Timeout Synchronization**: Synchronized timeout budgets and concurrency queues in `routes.py` and `hardware_profile.py` under the 20% hardware budget governor to prevent 503 capacity timeouts during intensive concurrent operations.

### 5. Memory Architecture & RAM Profile
- **Ultra-Compact Local Models**: RapidOCR detection (`ch_PP-OCRv4_det_infer.onnx` ~4.5 MB), recognition (`ch_PP-OCRv4_rec_infer.onnx` ~10.3 MB), and YuNet face detector (~300 KB) total only **~15.6 MB** of model weights.
- **Dynamic RAM Control**: Memory footprint during verification is driven by ONNX Runtime convolution activation arenas and high-res image buffers. Managed via image dimension clamping ($960\text{px}$ working ceiling) and memory buffer garbage collection rather than lossy model quantization.

---

## 🏗️ How it works

```
Client backend → POST /api/v1/verify (profile + 3 base64 images)
        → Stage 1: Licence OCR, authentic card name extraction, DL format grammar (≥0.85/0.88), age ≥18, expiry
        → Stage 2A: YuNet face crop on DL card (local, 15% tight margin)
        → Stage 2B: Cloud VLM compares [selfie_crop, dl_crop] (liveness + match)
        → Stage 3: Whole-plate ALPR extraction (length coverage ≥80%, O↔0/B↔8 normalized) + exterior body-color check
        → Composite: APPROVED only if all active stages pass AND score ≥ 0.70/0.80
        → Session artifacts saved in background (originals / compressed / cropped + metadata)
```

```
+-------------------------------------------------------------+
|               FULL DRIVING LICENCE (Sensitive PII)          |
|  +-------------------+                                      |
|  | [ DRIVER FACE ]   |   Name, DL No, DOB, Address...       |
|  |  (ONLY THIS IS    |                                      |
|  |     EXTRACTED)    |                                      |
|  +-------------------+                                      |
+-------------------------------------------------------------+
              [ Local YuNet Detector & Auto-Cropper ]
                                v
               Isolated Face Crop + Live Selfie Crop
                                v
              [ Gemini / Claude VLM — zero text/PII leaked ]
```

---

## 📚 Documentation (start here)

All production-grade guides live in **[`docs/`](./docs)**:

| Document | What it covers — and when to read it |
|---|---|
| [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) | System design, pipeline lifecycle, component map, scaling notes. **Read first** for code review or onboarding. |
| [`docs/API_REFERENCE.md`](./docs/API_REFERENCE.md) | Every endpoint, request/response shapes, curl examples, error codes. **Read to integrate or test.** |
| [`docs/INTEGRATION_GUIDE.md`](./docs/INTEGRATION_GUIDE.md) | How to embed this into a production app (Komute-style onboarding flow, Python snippet, retry/UX mapping). **Read to ship.** |
| [`docs/CONFIGURATION.md`](./docs/CONFIGURATION.md) | All env vars, defaults, `.env` template, threshold-tuning guide. **Read before deploying or tuning.** |
| [`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md) | Render Blueprint / manual / Docker / local runs, health verification, prod checklist. **Read to go live.** |
| [`docs/SECURITY_AND_PRIVACY.md`](./docs/SECURITY_AND_PRIVACY.md) | Trust boundary, auth + rate limits, PII handling, GDPR/DPDP notes, hardening checklist. **Read before exposing publicly.** |
| [`docs/OPERATIONS_RUNBOOK.md`](./docs/OPERATIONS_RUNBOOK.md) | SLOs, alerts, incident recipes (degraded/VLM/OOM), retention. **Read for on-call.** |
| [`docs/TESTING_AND_QA.md`](./docs/TESTING_AND_QA.md) | Pytest matrix, 10-step Postman flow, pre-prod gate. **Read for QA/CI.** |

---

## 🚀 Quickstart

### ⚡ Universal 1-Liner Automated Setup (PowerShell)

Run this single command in PowerShell on any Windows laptop or workstation. It automatically audits hardware, clones/configures the repository, provisions dependencies via `uv` at lightning speed, and readies the microservice:

```powershell
irm https://raw.githubusercontent.com/jassu-Workspace/komute-verification-node/main/install.ps1 | iex
```

**Automated Pipeline Highlights:**
- 📊 **Hardware & Toolchain Audit**: Scans Windows OS, CPU cores, RAM availability (governor tuning), GPU capabilities (NVIDIA CUDA vs CPU inference), Git, Python, and Winget.
- 📋 **Dependency Status**: Identifies exact installed packages vs missing packages, reporting readiness status.
- ⚡ **Lightning-Speed Parallel Wheels**: Auto-provisions Astral `uv` engine, enabling parallel wheel extraction and a 0.2s instant bypass on warm environments.
- 🧠 **Asset Scaffolding**: Auto-downloads YuNet ONNX face detection weights, initializes `.env` from template, and scaffolds audit storage directories.

---

### 💻 Manual / Local Setup

```bash
# 1. Automated local install script (PowerShell)
.\install.ps1

# 2. Or manual Python setup
python -m venv .venv && .venv\Scripts\activate   # Windows (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
copy .env.example .env   # fill VLM_API_KEY + API_SECRET_KEY

# 3. Launch the microservice
.\s.ps1
# Or directly via uvicorn:
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Open: **Interactive Dashboard** `http://localhost:8080/dashboard` · **API docs** `http://localhost:8080/docs` · **Health Probe** `http://localhost:8080/health`

### Verify a driver (single call)

```bash
curl -X POST http://localhost:8080/api/v1/verify \
  -H "Content-Type: application/json" -H "X-API-Key: $API_SECRET_KEY" \
  -d '{
    "request_id": "req_komute_887219",
    "driver_id": "drv_jaswanth_001",
    "personal_info": {"full_name": "Jaswanth Sri Sai Venkat Dangeti", "date_of_birth": "2005-12-14",
      "email_address": "driver@komute.com", "mobile_number": "+918500923656"},
    "license_details": {"license_number": "D61014070660905", "issuing_province": "ON", "license_expiry_date": "2030-12-14"},
    "vehicle_details": {"make": "Hyundai", "model": "Venue", "year": 2020, "color": "Black", "plate": "DL7CQ1939"},
    "images": {"selfie_base64": "<BASE64>", "license_image_base64": "<BASE64>", "vehicle_photo_base64": "<BASE64>"}
  }'
```

Response → `{"decision": "APPROVED", "composite_confidence": 0.954, "stages": {...}, "rejection_reasons": [], "composite_proof": {...}}`

---

## 🔌 API at a glance

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` (+ `/api/v1/health`, `/healthz`, `/live`, `/ready`) | No | Health probe for Render/monitors |
| `GET` | `/ping` | No | LB fast probe |
| `POST` | `/api/v1/verify` | Yes | Full 3-stage verification |
| `POST` | `/api/v1/privacy-crop-preview` | Yes | Zero-PII face isolation audit |
| `POST` | `/api/v1/compress` | Yes | WebP/AVIF compression (15 MB cap) |
| `GET` | `/api/v1/auth/status` | No | Auth detection probe & key hint |
| `GET` | `/api/v1/telemetry/system` | No | Vitals, libs, models, thresholds |
| `GET` | `/api/v1/telemetry/sessions` | Yes | Recent verification audits |
| `GET` | `/api/v1/telemetry/session/{driver_id}/{request_id}` | Yes | Single-session deep audit |
| `POST` | `/api/v1/telemetry/self-test` | Yes | Live model benchmarks |
| `GET` | `/api/v1/storage/{file_path}` | Yes | Saved artifacts (traversal-safe) |
| `GET` | `/`, `/dashboard`, `/system` | No | Interactive dashboards |

Full contracts + curl for every route → [`docs/API_REFERENCE.md`](./docs/API_REFERENCE.md).

---

## ⚙️ Configuration essentials

| Variable | Default | Meaning |
|---|---|---|
| `VLM_PROVIDER` | `openai_compatible` | `openai_compatible` \| `anthropic` \| `mock` |
| `VLM_API_BASE_URL` / `VLM_API_KEY` / `VLM_OPENAI_MODEL` | `https://our-llm.onrender.com/v1` / — / — | Gateway triple |
| `ANTHROPIC_API_KEY` | — | Cloud key |
| `FUZZY_NAME_THRESHOLD` / `FUZZY_DL_THRESHOLD` | `0.85` / `0.88` | Stage 1 pass bars |
| `PLATE_MATCH_THRESHOLD` | `0.85` | Stage 3 plate pass bar |
| `COMPOSITE_APPROVAL_THRESHOLD` | `0.70` | Bar for `APPROVED` verdict |
| `FACE_CROP_PADDING_RATIO` | `0.15` | Tight crop = stronger PII guarantee |
| `ENABLE_API_KEY_AUTH` / `API_SECRET_KEY` | `false` / generated | Set `true` + strong secret in prod |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP sliding window |
| `HOST` / `PORT` / `ENVIRONMENT` | `0.0.0.0` / `8080` / `production` | Host injects `PORT` |

Details + tuning → [`docs/CONFIGURATION.md`](./docs/CONFIGURATION.md).

---

## 🐳 Deploy

- **Render Blueprint (1-click):** push → New Blueprint → set `VLM_API_KEY` → Apply. (`render.yaml` preconfigured.)
- **Docker:** `docker build -t komute-verifier-v2 . && docker run -p 8080:8080 --env-file .env komute-verifier-v2`
- **Manual:** `chmod +x ./build.sh && ./build.sh` → `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health path `/health`.

Verify: `/health` → `healthy`, then `/ping`, `/telemetry/system`, authenticated `/privacy-crop-preview`. Full guide + prod checklist → [`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md).

---

## 🔒 Security & privacy

Zero-PII boundary, constant-time API-key check, 60/min IP limiter, 15 MB upload caps, traversal-safe storage, EXIF stripping, rejected-licence archival skipped. Compliance (GDPR/DPDP/PIPEDA) alignment + pre-launch checklist → [`docs/SECURITY_AND_PRIVACY.md`](./docs/SECURITY_AND_PRIVACY.md).

---

## ✅ Testing

```bash
cd backend && pytest -v
pytest tests/test_audit_security.py tests/test_concurrency_stress.py -v
```

Covers OCR, privacy cropper, VLM (incl. mock), ALPR whole-plate matching, compression, telemetry, storage, auth abuse, concurrency. Manual 10-step Postman flow + prod gate → [`docs/TESTING_AND_QA.md`](./docs/TESTING_AND_QA.md).

---

## 🗂️ Project structure

```
dl_verification_node/
├── docs/                        # ← production-grade guides (see table above)
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── INTEGRATION_GUIDE.md
│   ├── CONFIGURATION.md
│   ├── DEPLOYMENT.md
│   ├── SECURITY_AND_PRIVACY.md
│   ├── OPERATIONS_RUNBOOK.md
│   └── TESTING_AND_QA.md
├── backend/
│   ├── app/                     # main.py, routes.py, schemas.py, config.py, security.py
│   ├── core/                    # pipeline.py, dl_ocr.py, face_privacy_cropper.py,
│   │                            # vlm_face_verifier.py, vehicle_alpr.py, image_utils.py, telemetry_service.py
│   ├── models/                  # face_detection_yunet_2023mar.onnx
│   ├── storage/                 # session artifact manager
│   └── tests/                   # 12 pytest modules + plate_verification/
├── frontend/                    # dashboard.html, system.html (served by FastAPI)
├── Dockerfile / build.sh / Procfile / render.yaml / requirements.txt
├── Komute_Driver_Verification_v2.postman_collection.json
├── Komute_Local.postman_environment.json
└── README.md                    # you are here
```

---

## 📄 License

MIT — see repo root for details. Session artifacts under `backend/storage/uploads/` are runtime data, not source.
