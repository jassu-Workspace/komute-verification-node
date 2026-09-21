# Deployment — Production Guide

> How to ship this service to production (Render Blueprint, manual, Docker) with zero-downtime health checks.

## 1. Prerequisites

- Python 3.11.9, `pip`, `venv`
- Google Gemini API key (or Anthropic key) for live biometrics; `VLM_PROVIDER=mock` works without keys for smoke tests
- Model asset present: `backend/models/face_detection_yunet_2023mar.onnx`
- Required env (see `docs/CONFIGURATION.md`): `GEMINI_API_KEY`, `API_SECRET_KEY`, `ENABLE_API_KEY_AUTH`, `ENVIRONMENT=production`

## 2. Option A — Render Blueprint (recommended, 1-click)

`render.yaml` already defines service `komute-driver-verifier` (Oregon, `buildCommand: ./build.sh`, `startCommand: cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`, `healthCheckPath: /health`).

1. Push repo to GitHub/GitLab.
2. Render Dashboard → **New + → Blueprint** → connect repo.
3. Confirm auto-detected build/start/health check.
4. Set secret `GEMINI_API_KEY`; keep `API_SECRET_KEY` generated value safe.
5. **Apply** → wait for `GET /health → 200 healthy`.

## 3. Option B — Manual Render Web Service

- **New + → Web Service** → repo → Runtime `Python`, Region `Oregon`.
- Build: `chmod +x ./build.sh && ./build.sh`
- Start: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`, Instance: Free (dev) / Starter 512 MB+ (prod).
- Env vars: `PYTHON_VERSION=3.11.9`, `ENVIRONMENT=production`, `VLM_PROVIDER=gemini`, `VLM_MODEL=gemini-2.5-flash`, `GEMINI_API_KEY=<secret>`, `ENABLE_API_KEY_AUTH=false|true`.

## 4. Option C — Docker (Render / any registry / VPS)

Production `Dockerfile` (python:3.11-slim, OpenCV syslibs, `HEALTHCHECK` on `/health`, `WORKDIR /app/backend`, dynamic `$PORT`):

```bash
docker build -t komute-verifier-v2 .
docker run -p 8080:8080 --env-file .env komute-verifier-v2
curl -i http://localhost:8080/health
```

On Render: **New + → Web Service** → Runtime `Docker`, Dockerfile `./Dockerfile`, context `.`, health path `/health`, same env vars.

`build.sh` does: pip toolchain upgrade → `pip install -r requirements.txt` → `mkdir -p backend/storage/uploads/drivers backend/storage/uploads/previews backend/uploads` → verify YuNet model.

## 5. Local / Staging Run

```bash
cd dl_verification_node
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env  # then fill GEMINI_API_KEY / API_SECRET_KEY
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8080
# UI: http://localhost:8080/ | Docs: http://localhost:8080/docs | Health: http://localhost:8080/health
```

`Procfile` (`web: cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`) covers Heroku-style hosts.

## 6. Verify Deployment

```bash
curl -i https://<service>.onrender.com/health
curl https://<service>.onrender.com/ping
curl https://<service>.onrender.com/api/v1/telemetry/system
# authenticated smoke:
curl -X POST "https://<service>.onrender.com/api/v1/privacy-crop-preview" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_SECRET_KEY" \
  -d '{"image_base64":"<BASE64>","margin_ratio":0.15}'
```

Render zero-downtime deploys gate on `/health`. Expect `status: healthy` with `ocr_available`, `face_detector_available`, `checks.storage_writable` all true. `degraded` → check logs for missing YuNet model or unwritable storage.

## 7. Production Checklist

- [ ] `ENVIRONMENT=production` (disables reload), `PORT` injected by host
- [ ] `ENABLE_API_KEY_AUTH=true` + strong `API_SECRET_KEY` (see `docs/SECURITY_AND_PRIVACY.md`)
- [ ] `GEMINI_API_KEY` set as secret, never committed (`.env` is gitignored)
- [ ] `OMP_NUM_THREADS=2`, `ORT_INTRA_OP_NUM_THREADS=2`, `ORT_INTER_OP_NUM_THREADS=1` retained
- [ ] Persistent disk mounted if session artifacts must survive restarts (Render Free is ephemeral)
- [ ] Uptime monitor on `/health` (30s) + `/ping` for LB; alert on `degraded` or `memory_usage_mb` spike
- [ ] CORS locked down from `*` in `main.py` to production origins before public launch
