# Testing & QA Strategy

> How to prove the service works: unit, API, security, stress, and manual Postman flows.

## 1. Automated Suite (`backend/tests/`)

```bash
cd backend
pytest -v
# targeted:
pytest tests/test_dl_ocr.py tests/test_privacy_cropper.py -v
pytest tests/test_audit_security.py tests/test_concurrency_stress.py -v
```

| Test file | What it proves |
|---|---|
| `test_api_pipeline.py` | End-to-end `/verify` contract, decision + composite proof |
| `test_dl_ocr.py`, `test_canadian_dl_ocr.py` | Stage 1 OCR, fuzzy thresholds, expiry/age logic |
| `test_privacy_cropper.py` | Stage 2A YuNet isolation, zero-PII guarantee |
| `test_vlm_biometrics.py` | Stage 2B provider switch incl. `mock` |
| `test_vehicle_alpr.py`, `plate_verification/` | Stage 3 plate + color |
| `test_image_compression.py` | WebP/AVIF path, 15 MB cap |
| `test_telemetry.py`, `test_storage.py` | Telemetry shape, session persistence, traversal guards |
| `test_audit_security.py` | `401` no/wrong key, `429` burst, `403` traversal |
| `test_concurrency_stress.py` | Parallel-pipeline stability |

Config: `backend/pytest.ini`, fixtures in `tests/conftest.py`, sample assets in `tests/`, `test_images/`. Helpers: `postman_test_helper.py`, `run_brutal_ocr_tests.py`, `run_ocr_on_images.py`, `benchmark.py`, `generate_md_report.py`, `OCR_Test_Report.md`.

CI expectation: full suite green on every PR; security + stress files mandatory before prod deploy.

## 2. Manual QA with Postman (10-step flow)

Import `Komute_Driver_Verification_v2.postman_collection.json` + `Komute_Local.postman_environment.json` (`base_url=http://localhost:8080`, `api_key=<secret>`). Full scripts in `POSTMAN_TESTING_GUIDE.md`.

1. `GET /health` → `200`, `healthy`, OCR + face + storage true
2. `GET /ping` → `pong`
3. `GET /api/v1/telemetry/system` → libs + models + `pipeline_config`
4. `POST /privacy-crop-preview` → `face_detected`, `pii_sanitized`
5. `POST /verify` → `APPROVED|REJECTED`, confidence 0–1, all 3 stages
6. `POST /compress` → binary smaller than original (`X-Saved-Bytes > 0`)
7. `GET /telemetry/sessions` + `GET /telemetry/session/{driver}/{request}` → audit trail
8. `POST /telemetry/self-test` → all `test_passed: true`
9. Security: no-key `401`, wrong-key `401`, 61-burst `429`, `../` traversal `403`
10. Negative: expired licence → `REJECTED`; faceless image → `face_detected:false`; bad base64 → `400/500`; missing fields → `422`

Base64 helper (PowerShell): `[Convert]::ToBase64String([IO.File]::ReadAllBytes("image.png"))`.

## 3. Pre-Production Gate

- [ ] `pytest -v` fully green in clean venv (Python 3.11)
- [ ] Auth + rate-limit + traversal abuse tests green with `ENABLE_API_KEY_AUTH=true`
- [ ] Live VLM smoke (`gemini`) + `mock` fallback smoke both pass
- [ ] `/privacy-crop-preview` audited on 10+ real card angles — no text leakage
- [ ] p95 `/verify` latency recorded; 20s timeout never hit in staging
- [ ] Threshold changes (`FUZZY_*`, `COMPOSITE_*`) A/B logged with precision/recall impact
