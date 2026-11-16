# Komüte-Local

An offline verification module for driver onboarding. A host application sends a driver's claimed details and four photos. Komüte-Local returns **`APPROVED` or `REJECTED`** with reason codes, with no human in the loop. It runs on CPU in a 1.0–1.3 GB RAM budget and serves 13 Canadian provinces and territories. Nothing leaves the machine.

**Status:** built and tested on three real Canadian licences plus synthetic data. It is **not production-ready**: liveness has no model yet, accuracy is not measured on enough real people, and the Linux container was never run (deferred until handover, by decision). See [`docs/ACCEPTANCE_REPORT.md`](docs/ACCEPTANCE_REPORT.md) for what is proven and what is not.

## What it checks

1. **Licence barcode.** Decodes the AAMVA PDF417 on the back and trusts it over anything printed. Cross-checks the typed jurisdiction, licence number, name (fuzzy, pass mark 90%), date of birth, class, issue date and expiry date against it (all exact except the name), then checks expiry, issue date not in the future and age 18+. Any class and restriction is accepted.
2. **Face.** The selfie against the licence photo (YuNet, `w600k_r50` embeddings), after a single-photo liveness check.
3. **Vehicle.** Plate read by an OCR restricted to `A–Z`, `0–9` and compared to the claimed plate. Body colour is advisory only.

Any failed gate gives `REJECTED` with every reason code, each marked **retryable** (a better photo could fix it) or **final**. A fault inside the module is a `500` error, never a decision.

## Quickstart

On Windows, one command checks everything and starts it:

```powershell
.\setup.ps1           # once, on a fresh x64 Windows 10 1809+/Server 2019+ machine: checks RAM, disk and network; installs the Visual C++ runtime (Microsoft signature checked, asks for admin), uv, Python 3.12, the venv, requirements.lock, the 4 models (SHA-256 checked against models.lock), dev keys; then runs the tests
.\setup.ps1 -SkipTests   # same, without the tests; -Force re-downloads models and reinstalls packages
.\run.ps1               # checks uv, Python 3.12, dependencies (installs missing ones with uv) and model files; starts the backend on port 8000; opens the test page
.\run.ps1 -CheckOnly    # checks only
.\run.ps1 -WithApiKey   # require the X-API-Key header (key in %LOCALAPPDATA%\komute\dev.env)
```

Authentication is **off** by default for testing. If script execution is blocked: `powershell -ExecutionPolicy Bypass -File .\run.ps1`. Or step by step:

```bash
uv python install 3.12
uv venv --python 3.12 "$LOCALAPPDATA/komute/venv"
uv pip install --python "$LOCALAPPDATA/komute/venv" -r requirements.txt
python scripts/make_dev_keys.py && source scripts/dev_env.sh
KOMUTE_DEV_SKIP_LIVENESS=true "$PY" -m uvicorn backend.app.main:create_app --factory --workers 1
```

On an Ubuntu 22.04/24.04 x86_64 server (2 GB RAM or more), from the unpacked source folder:

```bash
sudo ./setup.sh                  # packages, user komute, uv, Python 3.12, requirements.lock, models (SHA-256 checked), /etc/komute/komute.env, systemd service
sudo systemctl restart komute    # the service runs ./start.sh; logs: journalctl -u komute -f
```

Guides (PDF): [`docs/guides/`](docs/guides/).

Open `http://127.0.0.1:8000/` for the test page. Keep `--workers 1`. Settings and the container are in [`docs/DEPLOYMENT_AND_OPERATIONS.md`](docs/DEPLOYMENT_AND_OPERATIONS.md).

## API

`POST /v1/verify`, `multipart/form-data`. No key is needed by default; if `KOMUTE_API_KEY` is set, send header `X-API-Key`. The full contract for the main application is in [`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md). To test every endpoint in Postman, import the collection in [`postman/`](postman/README.md).

| Field | Notes |
|---|---|
| `request_id`, `driver_id` | letters, digits, `-`, `_`, up to 64. Same `request_id` and data returns the stored result; different data returns `409` |
| `consent_at`, `consent_policy_version` | required. ISO-8601 time **with** a time zone |
| `claimed.full_name`, `claimed.date_of_birth` (`YYYY-MM-DD`), `claimed.licence_number`, `claimed.issuing_province`, `claimed.plate_number`, `claimed.vehicle_colour`, `claimed.licence_class`, `claimed.issue_date` and `claimed.expiry_date` (`YYYY-MM-DD`) | province is one of `ON QC BC AB MB SK NB NS NL PE YT NT NU` (`NF` is accepted as `NL`) |
| `claimed.vehicle_make`, `claimed.vehicle_model`, `claimed.vehicle_year` | optional, stored only |
| `selfie`, `licence_front`, `licence_back`, `vehicle` | JPEG, PNG or WebP, up to 10 MB each |

Responses: `200` with `{decision, reasons[{code, retryable, message}], stages, advisories, timings_ms, rss_mb, audit}`; `422` bad input (specific code); `401` bad key; `409` request id reused with other data; `503` busy (one verification at a time; honour `Retry-After`); `500` engine fault with **no** decision. Also `GET /v1/sessions/{request_id}`, `GET /health`, `GET /`.

**Decoded barcode popup:** on the test page, the barcode card has a **View decoded details** button. It shows each typed field next to the barcode value with its match score, and every decoded barcode element when `KOMUTE_RESPONSE_BARCODE_FIELDS=true` (`run.ps1` turns this on; it is off by default because it returns the address).

**System monitor:** open `/stats`, `/system-stats` or `/v1/system` in a browser. It refreshes every second and shows process and system CPU (overall and per core), RSS against the 900 MB target and 1200 MB limit, peak RAM, 15-minute CPU and memory charts, each model's status (running, idle, not loaded, skipped in dev), load time, run timings, tensor shapes and SHA-256, the current engine stage, per-step timing statistics, the last verification, process, disk, host and software details, and the effective settings. Its data comes from `GET /v1/system`: a browser gets the page, other callers get JSON (same `X-API-Key` rule). That endpoint never sends secrets or personal data.

If the barcode cannot be read the result is `REJECTED` / `BARCODE_UNREADABLE` (retryable): "The barcode on the back of your licence is not clearly visible. Please retake the photo of the barcode."

## Measured so far

| | Result |
|---|---|
| Memory (Windows, 50 runs on 3 real cards) | steady 334 MB, peak 383 MB, no drift. Linux not yet measured. Liveness model not included |
| Latency (same run) | p50 1.43 s, p95 1.85 s |
| Real licences | 3 of 3 backs decoded and parsed (NS, ON, SK). All 13 issuer codes decode in synthetic tests |
| Real impostor pairs | 6 pairs scored −0.10 to 0.06 against a provisional 0.40 threshold (far too few to calibrate) |

## Known limits

- **No liveness model.** Single-photo liveness is weaker than multi-frame. Production start requires a model file.
- **Face threshold unproven.** No false-accept or false-reject rate is measured; a rate of 1 in 10,000 needs about 30,000 impostor trials.
- **The plate stage proves a car with that plate text was photographed**, not that it belongs to the driver.
- Suspension and revocation are not checked: the host queries provincial driver records.

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [Stage 1 barcode](docs/STAGE1_BARCODE_SPECIFICATION.md) · [Stage 2 face](docs/STAGE2_BIOMETRICS_SPECIFICATION.md) · [Stage 3 vehicle](docs/STAGE3_ALPR_SPECIFICATION.md) · [Storage](docs/STORAGE_SPECIFICATION.md) · [Models](docs/MODELS_SPECIFICATION.md) and [measured model facts](docs/MODEL_FACTS.md) · [Deployment](docs/DEPLOYMENT_AND_OPERATIONS.md) · [Test page](docs/FRONTEND_ELEMENTS_SPECIFICATION.md) · [Integration guide](docs/INTEGRATION_GUIDE.md) · [Acceptance report](docs/ACCEPTANCE_REPORT.md)
