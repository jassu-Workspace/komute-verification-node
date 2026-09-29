# Operations Runbook — Monitoring, Incidents & SLOs

> For on-call and platform engineers running this service in production.

## 1. Health & Telemetry Endpoints

| Signal | Endpoint | Auth | Use |
|---|---|---|---|
| Liveness | `GET /live`, alias `/ready` | No | **Container/orchestrator health check.** Deliberately excludes the cloud VLM so a third-party outage never triggers a restart. |
| Readiness | `GET /health`, aliases `/api/v1/health`, `/healthz` | No | `healthy` vs `degraded`. **Gates on VLM readiness.** Use for monitoring and alerting, not as a restart probe. |
| VLM diagnostics | `GET /api/v1/diagnostics/vlm[?live=true]` | Yes | Per-provider credential + key-shape validity, `.env` precedence and conflicts, timeout budget, classified last error. `live=true` sends one minimal call per ready provider. |
| Ping | `GET /ping` | No | LB fast probe, returns `pong`. |
| Deep status | `GET /api/v1/telemetry/system` | No | `{vitals, libraries, ai_models, pipeline_config}` - version + threshold audit. Cached for 30s. |
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

- `GET /live` non-200 for 2 consecutive polls → page (the process itself is unhealthy).
- `GET /health` `status == "degraded"` for 3 consecutive polls → page. Check `vlm_ready`,
  `vlm_key_shape_valid` and `vlm_last_error` first; the usual cause is a VLM credential
  problem, not a host problem.
- `memory_usage_mb` > 85% instance RAM for 5 min → warn; > 95% → page (risk of ONNX OOM).
- Rising `service_status == "degraded"` in `/verify` responses → alert on the rate, not on
  the `REJECTED` rate. Degradation means drivers are being rejected **without evaluation**.
- Disk usage (via telemetry `vitals`) > 80% → purge old sessions (ephemeral disk fills fast
  with base64 artifacts).

## 4. Incident Recipes

**`degraded` health**

1. `curl /health` — read `vlm_ready`, `vlm_key_shape_valid`, `vlm_last_error`.
2. `vlm_ready=false` → `curl -H "X-API-Key: …" /api/v1/diagnostics/vlm` and read each
   provider's `blocking_reason` / `key_shape_hint`.
3. `ocr_engine=false` → RapidOCR model download/corruption; redeploy (build reinstalls `rapidocr-onnxruntime`).
4. `face_detector=false` → YuNet ONNX missing (`backend/models/face_detection_yunet_2023mar.onnx`); `build.sh` warns on this — restore file and redeploy.
5. `storage_writable=false` → disk full or perms; clear `storage/uploads`, `chmod`, or attach persistent disk.

**Stage 2 `SERVICE_UNAVAILABLE` (the "VLM not configured" case)**

`POST /api/v1/verify` still returns **HTTP 200** with `decision: "REJECTED"`. That is by
design: a driver rejection and an infrastructure outage are different things and must not
share a status code. Read these fields instead:

| Field | Meaning |
|---|---|
| `service_status` | `"ok"` or `"degraded"`. If `degraded`, this is **not** an identity decision. |
| `degraded_subsystems[].reason` | `no_credentials` · `invalid_credentials` · `quota_exhausted` · `model_unavailable` · `provider_unreachable` · `timeout` · `unparseable_response` · `sdk_missing` · `config_error` · `provider_error` |
| `degraded_subsystems[].detail` | Masked human-readable cause. Never contains a credential. |
| `stages.face_biometrics.vlm_attempt_chain` | Every provider/model tried, with `ok`, `reason`, and `latency_ms`. |
| `stages.face_biometrics.vlm_confidence` | Always `0.0` on degradation. The system fails closed. |

Triage order:

1. `reason == "no_credentials"` → the provider has no key. Check `key_shape_hint`; a
   credential that does not match the provider (a gateway token missing for
   `openai_compatible`, or a token that does not start with `sk-ant-` for anthropic) is
   almost always the cause.
2. `reason == "invalid_credentials"` → key present but rejected. Rotate it.
3. `reason == "quota_exhausted"` → billing/quota in the provider console.
4. `reason == "model_unavailable"` → the provider's model ID (`VLM_OPENAI_MODEL` or
   `ANTHROPIC_MODEL`) is retired or misspelled. The chain skips it and tries the next
   entry in `VLM_MODEL_FALLBACKS`.
5. `reason == "timeout"` → raise `VLM_CALL_TIMEOUT_SECONDS` within the 60%-of-pipeline
   ceiling, or accept and let `VLM_FALLBACK_CHAIN` carry the request.
6. `reason == "unparseable_response"` → the model replied with prose instead of the required
   JSON. This is **never** treated as a match. Raise `VLM_PARSE_RETRY_COUNT` or switch model.
7. Check `env_sources.conflicts` for two `.env` files disagreeing about a credential.

**High `REJECTED` rate (false rejects)**

1. First separate real rejections from degradation via `service_status`.
2. Sample `GET /telemetry/sessions` + per-session audits — which stage fails?
3. Stage 1: check OCR lines + `number_similarity`/`name_similarity` vs thresholds; image quality (blur/glare) is the #1 cause.
4. Stage 2: check `is_live`, `is_match`, `vlm_confidence`; test crops via `/privacy-crop-preview`.
5. Stage 3: check `extracted_plate` vs `plate_similarity`; OCR confusion (`O/0`, `B/8`) is normalized automatically.

**`429` floods**

1. Confirm legitimate burst vs scrape (per-IP session list).
2. Raise `RATE_LIMIT_PER_MINUTE` or add Redis-backed limiter for multi-replica (in-memory limiter is per-instance).

**OOM / slow**

1. Confirm thread clamps (`OMP=2`, `ORT_INTRA=2/INTER=1`) are set.
2. Reduce workers or move to larger instance; avoid full-size base64 retries.

## 5. Deployment & Rollback

- Render auto-deploys on push; `healthCheckPath` is `/live` (liveness only, so a VLM outage
  does not restart the service). Rollback = redeploy previous commit via Render dashboard.
- Gate on `GET /health` for **readiness** before sending traffic.
- Docker: `docker build -t komute-verifier-v2 . && docker run -p 8080:8080 --env-file .env komute-verifier-v2`,
  then `curl -i localhost:8080/live`. The image `HEALTHCHECK` uses `/live` for the same reason.
- **`.env` must never be baked into an image.** `.dockerignore` excludes `.env`, `.env.*`,
  `*.pem`, `*.key` and `*.p12`. If a key was ever committed to an image layer, rotate it —
  deleting the file later does not remove it from the layer history.
- Post-deploy smoke: `/live` → `/health` → `/ping` → `/api/v1/diagnostics/vlm` → `/telemetry/system` → authenticated `/privacy-crop-preview`.

## 6. Retention & Capacity

- Session artifacts accumulate under `storage/uploads/` — schedule purge (e.g. `find storage/uploads/sessions -mtime +30 -delete`) and back up `session_metadata.json` externally if audits require it.
- Multi-replica note: sessions and rate-limit state are local disk/memory — add shared object storage (S3/GCS) + Redis before scaling past 1 replica.
