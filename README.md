# Komüte Driver Verifier v2 🚗🛡️
### Production-Grade FastAPI Microservice for Driver Onboarding with Zero-PII Cloud Biometrics & ALPR

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Privacy Compliance](https://img.shields.io/badge/Privacy-GDPR%20%7C%20DPDP%20%7C%20PIPEDA-purple.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

---

## 1. System Overview

**Komüte Verifier v2** is a multi-stage identity verification and vehicle validation microservice designed for ridesharing and carpooling onboarding platforms. It combines high-speed local computer vision with cloud Vision-Language Models (VLM) while strictly preserving user privacy:

1. **Stage 1 (License OCR & Profile Cross-Check)**:
   - Local high-precision OCR (EasyOCR / PaddleOCR).
   - Extracts License Number, Full Name, DOB, Expiry Date, and Province/Category.
   - Cross-checks against registered profile fields using fuzzy Levenshtein distance ($\ge 0.88$) and token sort ratio ($\ge 0.85$).
   - Validates driver legal age ($\ge 18$ years) and license expiry.

2. **Stage 2 (Zero-PII Face Crop & Cloud VLM Biometrics)**:
   - Local face isolation engine (OpenCV Haar Cascade / YuNet) locates the driver's portrait coordinates on the DL card.
   - Auto-crops the face tightly with 15% padding, guaranteeing **zero text, numbers, address, or PII** from the DL card are leaked.
   - Sends **only** the two face crops (`[selfie_face_crop, dl_face_crop]`) to Google Gemini 1.5 Pro/Flash Vision API or Anthropic Claude 3.5 Sonnet.
   - Evaluates craniofacial bone structure, inter-pupillary ratio, nasal geometry, and anti-spoof liveness.

3. **Stage 3 (Vehicle ALPR & Visual Verification)**:
   - Isolates number plates using edge detection, morphological closing, and contour aspect ratio filtering.
   - Extracts plate number via OCR with OCR character confusion normalization (`O` $\leftrightarrow$ `0`, `I` $\leftrightarrow$ `1`, `B` $\leftrightarrow$ `8`, `Z` $\leftrightarrow$ `2`, `S` $\leftrightarrow$ `5`, etc.).
   - Analyzes dominant vehicle exterior color via HSV/K-Means color segmentation.

4. **Composite Decision Engine**:
   - `APPROVED`: Composite score $\ge 0.90$ with all stages verified.
   - `MANUAL_REVIEW`: Composite score $0.70 - 0.89$ or minor fuzzy discrepancy.
   - `REJECTED`: Biometric mismatch, spoofed selfie, underage driver, expired license, or plate mismatch.

---

## 2. Privacy Architecture (Zero-PII Face Isolation)

```
+-------------------------------------------------------------+
|               FULL DRIVING LICENSE (Sensitive PII)          |
|  +-------------------+                                      |
|  | [ DRIVER FACE ]   |   Name: Jaswanth Sri Sai...          |
|  |                   |   DL No: DDNPV1331Q                  |
|  |  (ONLY THIS IS    |   DOB: 2005-12-14                    |
|  |     EXTRACTED)    |   Address: 123 Sample St...          |
|  +-------------------+   Signature: [X]                     |
+-------------------------------------------------------------+
                               |
            [ Local Face Detector & Auto-Cropper ]
                               |
                               v
                     Isolated Face Crop (256x256)
                               +
                     Live Selfie Crop (256x256)
                               |
                               v
             [ Sent to Gemini 1.5 / Claude 3.5 Sonnet ]
             (Zero Text, Zero Numbers, Zero PII Leaked)
```

---

## 3. Directory Layout

```
komute_verifier_v2/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI Application Entrypoint
│   ├── config.py                # Environment & API Key Settings (Gemini / Anthropic)
│   ├── routes.py                # REST Endpoints (/verify, /health, /privacy-crop-preview)
│   ├── schemas.py               # Pydantic v2 Models for all stages
│   ├── dashboard.py             # Interactive Web UI Dashboard
│   └── security.py              # API Key Authentication & Sliding Window Rate Limiter
├── core/
│   ├── __init__.py
│   ├── dl_ocr.py                # Stage 1: Local Driving License OCR & Fuzzy Cross-Matching
│   ├── face_privacy_cropper.py  # Stage 2A: Local Face Detection & Zero-PII Auto-Cropper
│   ├── vlm_face_verifier.py     # Stage 2B: Cloud VLM Client (Gemini 1.5 / Claude 3.5)
│   ├── vehicle_alpr.py          # Stage 3: Number Plate & Vehicle Color Verification
│   ├── image_utils.py           # Base64 decoding, EXIF orientation, CLAHE enhancement
│   └── pipeline.py              # Master 3-Stage Orchestrator & Composite Decision Matrix
├── tests/
│   ├── test_dl_ocr.py
│   ├── test_privacy_cropper.py
│   ├── test_vlm_biometrics.py
│   ├── test_vehicle_alpr.py
│   └── test_api_pipeline.py
├── .env.example
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 4. Quickstart Guide

### 4.1 Local Installation
```bash
# Clone or navigate to directory
cd komute_verifier_v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (optional for live Gemini/Claude verification)
cp .env.example .env
```

### 4.2 Running the Service
```bash
# Start FastAPI application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:
- **Interactive UI Dashboard**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 5. API Reference

### 5.1 Verification Request (`POST /api/v1/verify`)

```bash
curl -X POST http://localhost:8000/api/v1/verify \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req_komute_887219",
    "driver_id": "drv_jaswanth_001",
    "personal_info": {
      "full_name": "Jaswanth Sri Sai Venkat Dangeti",
      "email": "jaswanthsrisai0011@gmail.com",
      "phone": "+918500923656",
      "date_of_birth": "2005-12-14",
      "gender": "Male"
    },
    "license_details": {
      "license_number": "DDNPV1331Q",
      "issuing_province": "ON",
      "expiry_date": "2030-12-14"
    },
    "vehicle_details": {
      "make": "Hyundai",
      "model": "Venue",
      "year": 2020,
      "color": "Black",
      "plate_number": "DL 7CQ 1939"
    },
    "images": {
      "selfie_base64": "data:image/jpeg;base64,...",
      "license_image_base64": "data:image/jpeg;base64,...",
      "vehicle_photo_base64": "data:image/jpeg;base64,..."
    }
  }'
```

### 5.2 Verification Response (`200 OK`)

```json
{
  "request_id": "req_komute_887219",
  "driver_id": "drv_jaswanth_001",
  "decision": "APPROVED",
  "composite_confidence": 0.954,
  "execution_time_ms": 1420.5,
  "stages": {
    "license_ocr": {
      "passed": true,
      "extracted_dl_number": "DDNPV1331Q",
      "extracted_name": "JASWANTH SRI SAI VENKAT DANGETI",
      "extracted_dob": "2005-12-14",
      "extracted_expiry": "2030-12-14",
      "number_matched": true,
      "number_similarity": 1.0,
      "name_matched": true,
      "name_similarity": 0.98,
      "dob_matched": true,
      "is_expired": false,
      "driver_age_valid": true,
      "confidence": 0.98,
      "details": "DL number matched (similarity: 1.00); Full name matched (similarity: 0.98)"
    },
    "face_biometrics": {
      "passed": true,
      "vlm_confidence": 0.96,
      "is_live": true,
      "is_match": true,
      "estimated_age_delta_years": 2,
      "privacy_face_cropped": true,
      "pii_leakage_prevented": true,
      "facial_feature_notes": "Matching craniofacial bone structure, eye spacing, and jawline proportion.",
      "reasoning": "Forensic comparison confirms high structural biometric agreement. No screen moire or spoof artifacts detected.",
      "verdict": "MATCH_CONFIRMED"
    },
    "vehicle_verification": {
      "passed": true,
      "extracted_plate": "DL7CQ1939",
      "plate_matched": true,
      "plate_similarity": 1.0,
      "color_matched": true,
      "detected_color": "Black",
      "color_confidence": 0.85,
      "details": "Plate matched 'DL7CQ1939' (similarity: 1.00); Exterior color verified as Black"
    }
  },
  "rejection_reasons": [],
  "manual_review_reasons": []
}
```

---

## 6. Running Tests

Execute pytest across the test suite:
```bash
pytest -v
```

---

## 7. Docker Deployment

```bash
docker build -t komute-verifier-v2 .
docker run -p 8000:8000 --env-file .env komute-verifier-v2
```
