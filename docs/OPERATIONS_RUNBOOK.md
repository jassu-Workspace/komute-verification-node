# Operations Runbook — Monitoring, Incidents & SLOs

> For on-call and platform engineers running this service in production.

## 1. Health & Telemetry Endpoints

| Signal | Endpoint | Auth | Use |
|---|---|---|---|
| Liveness | `GET /health`, aliases `/api/v1/health`, `/healthz`, `/live`, `/ready` | No | Render health check, uptime monitors (30s). `healthy` vs `degraded`. |
| Ping | `GET /ping` | No | LB fast probe, returns `pong`. |
| Deep status | `GET /api/v1/telemetry/system` | No | `{vitals, libraries, ai_models, pipeline_config}` — version + threshold audit. |
| Sessions | `GET /api/v1/telemetry/sessions?limit=50` | Yes | Recent decisions, latency, confidence. |
| Session audit | `GET /api/v1/telemetry/session/{driver_id}/{request_id}` | Yes | Stage breakdown + artifacts for disputes. |
| Self-test | `POST /api/v1/telemetry/self-test` | Yes | Live benchmarks: face cropper, OCR, compressor. |
| Dashboards | `/dashboard`, `/system` | No | Human-readable ops views (protect or disable publicly). |

`GET /health` includes `uptime_seconds/human`, `ocr_available`, `face_detector_available`, `checks.storage_writable`, `memory_usage_mb`.

## 2. SLOs (recommended starting point)

- Availability: `GET /health` 99.5% monthly (Render Free cannot guarantee this — use Starter+ for real SLOs).
- Latency p95 `POST /verify`: < 12s (VLM-bound; local-only preview < 2s).
- Success: `5xx` on `/verify` < 1%; `429` rate < 5% of traffic.
- Freshness: YuNet + RapidOCR loaded (`face_detector_available && ocr_available`) 100% of healthy checks.

## 3. Alerting

- `status == "degraded"` for 3 consecutive polls → page.
- `memory_usage_mb` > 85% instance RAM for 5 min → warn; > 95% → page (risk of ONNX OOM).
- `5xx` spike or VLM timeout surge → check provider status/quota, then fail over (`VLM_PROVIDER` switch or `mock` degraded mode).
- Disk usage (via telemetry `vitals`) > 80% → purge old sessions (ephemeral disk fills fast with base64 artifacts).

## 4. Incident Recipes

**`degraded` health**

1. `curl /api/v1/telemetry/system` — identify red subsystem.
2. `ocr_engine=false` → RapidOCR model download/corruption; redeploy (build reinstalls `rapidocr-onnxruntime`).
3. `face_detector=false` → YuNet ONNX missing (`backend/models/face_detection_yunet_2023mar.onnx`); `build.sh` warns on this — restore file and redeploy.
4. `storage_writable=false` → disk full or perms; clear `storage/uploads`, `chmod`, or attach persistent disk.

**High `REJECTED` rate (false rejects)**

1. Sample `GET /telemetry/sessions` + per-session audits — which stage fails?
2. Stage 1: check OCR lines + `number_similarity`/`name_similarity` vs thresholds; image quality (blur/glare) is the #1 cause.
3. Stage 2: check `is_live`, `is_match`, `vlm_confidence`; test crops via `/privacy-crop-preview`.
4. Stage 3: check `extracted_plate` vs `plate_similarity`; OCR confusion (`O/0`, `B/8`) is normalized automatically.

**VLM errors / timeouts**

1. Verify key + quota in provider console; check `VLM_MODEL` spelling.
2. Temporarily set `VLM_PROVIDER=mock` to keep Stages 1+3 live while provider recovers.
3. 20s pipeline gather timeout → `500`; consider retry with smaller images (1200px max already enforced).

**`429` floods**

1. Confirm legitimate burst vs scrape (per-IP session list).
2. Raise `RATE_LIMIT_PER_MINUTE` or add Redis-backed limiter for multi-replica (in-memory limiter is per-instance).

**OOM / slow**

1. Confirm thread clamps (`OMP=2`, `ORT_INTRA=2/INTER=1`) are set.
2. Reduce workers or move to larger instance; avoid full-size base64 retries.

## 5. Deployment & Rollback

- Render auto-deploys on push; gate on `/health`. Rollback = redeploy previous commit via Render dashboard.
- Docker: `docker build -t komute-verifier-v2 . && docker run -p 8080:8080 --env-file .env komute-verifier-v2`, then `curl -i localhost:8080/health`.
- Post-deploy smoke: `/health` → `/ping` → `/telemetry/system` → authenticated `/privacy-crop-preview`.

## 6. Retention & Capacity

- Session artifacts accumulate under `storage/uploads/` — schedule purge (e.g. `find storage/uploads/sessions -mtime +30 -delete`) and back up `session_metadata.json` externally if audits require it.
- Multi-replica note: sessions and rate-limit state are local disk/memory — add shared object storage (S3/GCS) + Redis before scaling past 1 replica.
