# Komüte Driver Verifier v2 🚗🛡️

### Production-Grade FastAPI Microservice for Driver Onboarding — Zero-PII Cloud Biometrics + ALPR

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Privacy](https://img.shields.io/badge/Privacy-GDPR%20%7C%20DPDP%20%7C%20PIPEDA-purple.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

> Verifies **who the driver is** (licence OCR), **that they are live and matching** (face biometrics), and **that their vehicle is legitimate** (plate + color) — in a single `POST /api/v1/verify` call with a strict `APPROVED / REJECTED` verdict.

---

## ✨ Why this service?

- **3-stage pipeline in one call** — DL OCR (RapidOCR + fuzzy match) → face isolation (YuNet, local) + cloud VLM (Gemini / Claude) → vehicle ALPR + color.
- **Zero-PII cloud boundary** — full licence image never leaves your host. Only tight face crops go to the VLM. Audit it live via `/privacy-crop-preview`.
- **Explainable decisions** — weighted composite `(0.30×OCR + 0.45×Face + 0.25×Vehicle)`, per-stage scores, and full `composite_proof` math in every response.
- **Production-ready** — API-key auth, per-IP rate limiting, health/telemetry nodes, session artifact audits, Docker + Render Blueprint deploys.
- **Drop-in integration** — proxy from your backend, switch on `decision`, done. See `docs/INTEGRATION_GUIDE.md`.

## 🏗️ How it works

```
Client backend → POST /api/v1/verify (profile + 3 base64 images)
        → Stage 1: Licence OCR, name/DL fuzzy match (≥0.85/0.88), age ≥18, expiry
        → Stage 2A: YuNet face crop on DL card (local, 15% padding)
        → Stage 2B: Gemini/Claude compares [selfie_crop, dl_crop] (liveness + match)
        → Stage 3: Plate extraction (O↔0/B↔8 normalized) + body-color check
        → Composite: APPROVED only if all stages pass AND score ≥ 0.85
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

## 📚 Documentation (start here)

All production-grade guides live in **[`docs/`](./docs)**. Read them in this order:

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

> Legacy step-by-step notes (`RENDER_DEPLOYMENT.md`, `POSTMAN_TESTING_GUIDE.md` at repo root) remain as companions to `docs/DEPLOYMENT.md` and `docs/TESTING_AND_QA.md`.

## 🚀 Quickstart

```bash
# 1. Setup
python -m venv .venv && .venv\Scripts\activate   # Windows (use source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt

# 2. Configure (see docs/CONFIGURATION.md)
copy .env.example .env   # fill GEMINI_API_KEY + API_SECRET_KEY

# 3. Run (from repo root)
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Open: **Dashboard** `http://localhost:8080/` · **API docs** `http://localhost:8080/docs` · **Health** `http://localhost:8080/health`

### Verify a driver (single call)

```bash
curl -X POST http://localhost:8080/api/v1/verify \
  -H "Content-Type: application/json" -H "X-API-Key: $API_SECRET_KEY" \
  -d '{
    "request_id": "req_komute_887219",
    "driver_id": "drv_jaswanth_001",
    "personal_info": {"full_name": "Jaswanth Sri Sai Venkat Dangeti", "date_of_birth": "2005-12-14",
      "email_address": "driver@komute.com", "mobile_number": "+918500923656"},
    "license_details": {"license_number": "DDNPV1331Q", "issuing_province": "ON", "license_expiry_date": "2030-12-14"},
    "vehicle_details": {"make": "Hyundai", "model": "Venue", "year": 2020, "color": "Black", "plate": "DL 7CQ 1939"},
    "images": {"selfie_base64": "<BASE64>", "license_image_base64": "<BASE64>", "vehicle_photo_base64": "<BASE64>"}
  }'
```

Response → `{"decision": "APPROVED", "composite_confidence": 0.954, "stages": {...}, "rejection_reasons": [], "composite_proof": {...}}`

## 🔌 API at a glance

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` (+ `/api/v1/health`, `/healthz`, `/live`, `/ready`) | No | Health probe for Render/monitors |
| `GET` | `/ping` | No | LB fast probe |
| `POST` | `/api/v1/verify` | Yes | Full 3-stage verification |
| `POST` | `/api/v1/privacy-crop-preview` | Yes | Zero-PII face isolation audit |
| `POST` | `/api/v1/compress` | Yes | WebP/AVIF compression (15 MB cap) |
| `GET` | `/api/v1/telemetry/system` | No | Vitals, libs, models, thresholds |
| `GET` | `/api/v1/telemetry/sessions` | Yes | Recent verification audits |
| `GET` | `/api/v1/telemetry/session/{driver_id}/{request_id}` | Yes | Single-session deep audit |
| `POST` | `/api/v1/telemetry/self-test` | Yes | Live model benchmarks |
| `GET` | `/api/v1/storage/{file_path}` | Yes | Saved artifacts (traversal-safe) |
| `GET` | `/`, `/dashboard`, `/system` | No | Human dashboards |

Full contracts + curl for every route → [`docs/API_REFERENCE.md`](./docs/API_REFERENCE.md). Postman collection + environment JSONs are at repo root.

## ⚙️ Configuration essentials

| Variable | Default | Meaning |
|---|---|---|
| `VLM_PROVIDER` / `VLM_MODEL` | `gemini` / `gemini-2.5-flash` | `gemini` \| `anthropic` \| `mock` |
| `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` | — | Cloud keys (secret, never commit) |
| `FUZZY_NAME_THRESHOLD` / `FUZZY_DL_THRESHOLD` | `0.85` / `0.88` | Stage 1 pass bars |
| `COMPOSITE_APPROVAL_THRESHOLD` | `0.85` | Bar for `APPROVED` |
| `FACE_CROP_PADDING_RATIO` | `0.15` | Tight crop = stronger PII guarantee |
| `ENABLE_API_KEY_AUTH` / `API_SECRET_KEY` | `false` / generated | Set `true` + strong secret in prod |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP sliding window |
| `HOST` / `PORT` / `ENVIRONMENT` | `0.0.0.0` / `8080` / `production` | Host injects `PORT` |

Details + tuning → [`docs/CONFIGURATION.md`](./docs/CONFIGURATION.md).

## 🐳 Deploy

- **Render Blueprint (1-click):** push → New Blueprint → set `GEMINI_API_KEY` → Apply. (`render.yaml` preconfigured.)
- **Docker:** `docker build -t komute-verifier-v2 . && docker run -p 8080:8080 --env-file .env komute-verifier-v2`
- **Manual:** `chmod +x ./build.sh && ./build.sh` → `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health path `/health`.

Verify: `/health` → `healthy`, then `/ping`, `/telemetry/system`, authenticated `/privacy-crop-preview`. Full guide + prod checklist → [`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md).

## 🔒 Security & privacy

Zero-PII boundary, constant-time API-key check, 60/min IP limiter, 15 MB upload caps, traversal-safe storage, EXIF stripping, rejected-licence archival skipped. Compliance (GDPR/DPDP/PIPEDA) alignment + pre-launch checklist → [`docs/SECURITY_AND_PRIVACY.md`](./docs/SECURITY_AND_PRIVACY.md).

> Before public launch: `ENABLE_API_KEY_AUTH=true`, restrict CORS from `*` to your origins (`backend/app/main.py`), rotate the sample `.env` secrets.

## ✅ Testing

```bash
cd backend && pytest -v
pytest tests/test_audit_security.py tests/test_concurrency_stress.py -v
```

Covers OCR, privacy cropper, VLM (incl. mock), ALPR, compression, telemetry, storage, auth abuse, concurrency. Manual 10-step Postman flow + prod gate → [`docs/TESTING_AND_QA.md`](./docs/TESTING_AND_QA.md).

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

## 📄 License

MIT — see repo root for details. Session artifacts under `backend/storage/uploads/` are runtime data, not source.
