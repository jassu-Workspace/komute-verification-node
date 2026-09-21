# Integration Guide — Embedding Verifier v2 in a Production App

> How to call this microservice from your ridesharing/carpooling backend (e.g. Komute) for driver onboarding.

## 1. Recommended Onboarding Flow

```
1. App collects: profile form + licence photo + live selfie + vehicle photo
2. App compresses client-side (≤1400px, JPEG) → base64
3. Backend → POST /api/v1/verify (server-to-server, with X-API-Key)
4. Switch on decision:
   - APPROVED → activate driver, store request_id + composite_proof
   - REJECTED  → show user-friendly reason, offer retry or manual KYC queue
5. Optional: GET /telemetry/session/{driver}/{request} for dispute review
```

Never call `/verify` directly from mobile/web clients — proxy through your backend so `X-API-Key` and raw PII stay server-side.

## 2. Minimal Backend Call (Python)

```python
import base64, httpx

def b64(path: str) -> str:
    return base64.b64encode(open(path, "rb").read()).decode()

resp = httpx.post(
    "https://<service>.onrender.com/api/v1/verify",
    headers={"X-API-Key": "<API_SECRET_KEY>", "Content-Type": "application/json"},
    json={
        "request_id": "req_komute_887219",
        "driver_id": "drv_jaswanth_001",
        "personal_info": {
            "full_name": "Jaswanth Sri Sai Venkat Dangeti",
            "email_address": "driver@komute.com",
            "mobile_number": "+918500923656",
            "date_of_birth": "2005-12-14",
        },
        "license_details": {
            "license_number": "DDNPV1331Q",
            "issuing_province": "ON",
            "license_expiry_date": "2030-12-14",
        },
        "vehicle_details": {
            "make": "Hyundai", "model": "Venue", "year": 2020,
            "color": "Black", "plate": "DL 7CQ 1939",
        },
        "images": {
            "selfie_base64": b64("selfie.jpg"),
            "license_image_base64": b64("licence.jpg"),
            "vehicle_photo_base64": b64("car.jpg"),
        },
    },
    timeout=30.0,
)
data = resp.json()
decision, score = data["decision"], data["composite_confidence"]
```

Node/`fetch` is equivalent — same JSON contract (see `docs/API_REFERENCE.md`).

## 3. Handling Decisions

- Persist `request_id`, `driver_id`, `decision`, `composite_confidence`, `composite_proof`, `rejection_reasons`, `execution_time_ms`.
- `APPROVED` requires all stages passed + score ≥ threshold — safe to auto-onboard.
- `REJECTED` — map technical reasons to UX copy:
  - `"Driving license has EXPIRED"` → "Your licence has expired. Upload a renewed licence."
  - `"Facial biometrics mismatch"` → "Selfie doesn't match licence photo. Retake in good light."
  - `"Vehicle plate mismatch"` → "Plate not recognized. Retake rear photo."
  - `"under 18"` → hard block per policy.
- Retries: generate a new `request_id` per attempt; cap at 3–5/day/driver to control VLM cost.

## 4. Production Hardening for Integrators

- **Timeouts/retries**: 30s timeout, 1 retry on network/`5xx` only (not on `REJECTED` — that's a verdict, not an error).
- **Idempotency**: `request_id` is your idempotency key — reuse it for safe status polling, mint new ones for new attempts.
- **Webhooks/jobs**: `/verify` is synchronous (VLM-bound, seconds). For scale, wrap calls in a job queue (Celery/BullMQ) with a pending-driver state.
- **Privacy**: forward only required fields; enforce retention parity with `docs/SECURITY_AND_PRIVACY.md`; link, don't duplicate, session artifacts.
- **Cost control**: pre-validate image quality client-side (blur/glare/size) before spending a VLM call; use `/privacy-crop-preview` as a cheap pre-check for faceless cards.
- **Monitoring**: alert on your side for `5xx` rate, p95 latency, and `REJECTED`-rate drift (signals threshold or camera-quality regressions).
