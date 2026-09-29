# Configuration — Environment Variables

> Single source of truth for every tunable. Sources: `.env` (local), Render env vars (prod), defaults in `backend/app/config.py`, template in `.env.example`.

## 1. Cloud VLM Providers

| Provider value | Credential | Required settings | Notes |
|---|---|---|---|
| `openai_compatible` | `VLM_API_KEY` (bearer token for the gateway) | `VLM_API_BASE_URL`, `VLM_API_KEY`, `VLM_OPENAI_MODEL` | Primary provider. Any `/chat/completions` endpoint — 9Router gateway, self-hosted vLLM, Azure OpenAI, etc. No extra pip dependency (uses `httpx`). |
| `anthropic` | `ANTHROPIC_API_KEY` (must start with `sk-ant-`) | `ANTHROPIC_MODEL` | Default `claude-sonnet-5`. |
| `mock` | none | `MOCK_VLM_*` | **Refused when `ENVIRONMENT=production`.** Deterministic, for tests/CI only. |

## 2. Full Reference

| Variable | Default | Required | Description |
|---|---|---|---|
| `VLM_PROVIDER` | `openai_compatible` | Yes | Active provider. See table above. |
| `VLM_MODEL_FALLBACKS` | *(empty)* | No | Ordered model-level fallbacks. Tried individually, so one retired/unavailable model ID is skipped without poisoning the provider. |
| `VLM_FALLBACK_CHAIN` | `openai_compatible,anthropic` | No | Ordered provider-level failover. `VLM_PROVIDER` is always tried first. Providers with no credential are skipped and recorded. |
| `ANTHROPIC_API_KEY` | — | If provider=anthropic | **Must start with `sk-ant-`.** |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | No | ⚠️ `claude-3-5-sonnet-20241022` was **retired 2025-10-28** and can never succeed. |
| `VLM_API_BASE_URL` | `https://our-llm.onrender.com/v1` | If provider=openai_compatible | OpenAI-compatible gateway base URL. `VLM_OPENAI_MODEL` selects the vision model on that gateway. |
| `VLM_API_KEY` | — | If provider=openai_compatible | Bearer token for the gateway. Issued by the gateway, not a vendor key. |
| `VLM_OPENAI_MODEL` | — | If provider=openai_compatible | Model ID as named by the gateway, e.g. `gpt-4o-mini`. |
| `MOCK_VLM_MATCH` / `MOCK_VLM_CONFIDENCE` / `MOCK_VLM_LIVENESS` | `true` / `0.95` / `true` | No | Mock provider output. |
| `VLM_MIN_MATCH_CONFIDENCE` | `0.70` | No | Stage 2 pass bar. Match AND live AND ≥ threshold AND `verdict == MATCH_CONFIRMED` are all required. |
| `FUZZY_NAME_THRESHOLD` | `0.85` | No | RapidFuzz token-sort pass bar for full name. |
| `FUZZY_DL_THRESHOLD` | `0.88` | No | Levenshtein pass bar for licence number. |
| `FUZZY_DOB_THRESHOLD` / `FUZZY_EXPIRY_THRESHOLD` | `80.0` / `80.0` | No | Field-level OCR gates on the 0-100 similarity scale. |
| `REGION_GATE_SCORE` | `75.0` | No | Minimum region/province confidence before the province is reported. |
| `SEVERE_BLUR_VARIANCE` | `20.0` | No | Laplacian variance below which characters are considered physically destroyed. |
| `COMPOSITE_APPROVAL_THRESHOLD` | `0.80` | No | Composite bar for `APPROVED`. Weights from `STAGE*_WEIGHT` below. |
| `STAGE1_WEIGHT` / `STAGE2_WEIGHT` / `STAGE3_WEIGHT` | `0.30` / `0.45` / `0.25` | No | Composite stage weights. **Must sum to 1.0** — the service refuses to start otherwise. |
| `STAGE3_PLATE_WEIGHT` / `STAGE3_COLOR_WEIGHT` | `0.75` / `0.25` | No | Vehicle stage internals: plate similarity vs color agreement. |
| `VLM_MISMATCH_FLOOR` | `0.50` | No | VLM confidence below this (with no match) is a hard biometric rejection. |
| `PLATE_HARD_FLOOR` / `PLATE_REVIEW_FLOOR` | `0.40` / `0.85` | No | Plate similarity floors for mismatch vs manual-review messaging. |
| `PLATE_MATCH_THRESHOLD` | `0.85` | No | ALPR string-similarity pass bar. |
| `COLOR_AGREEMENT_MIN` | `0.15` | No | Minimum dominant-color share to accept the registered color. |
| `FACE_YUNET_SCORE` / `FACE_YUNET_NMS` | `0.50` / `0.30` | No | YuNet detector score and NMS thresholds. |
| `FACE_VERIFY_THRESHOLD` | `0.45` | No | Face-presence verification pass bar (5-point landmarks). |
| `FACE_NMS_SCORE` / `FACE_NMS_IOU` | `0.40` / `0.35` | No | Candidate-box NMS score and IoU thresholds. |
| `OCR_BOX_THRESH` / `OCR_DB_THRESH` / `OCR_UNCLIP_RATIO` | `0.38` / `0.20` / `1.95` | No | RapidOCR DBNet detector tuning. |
| `OCR_WORKING_MAX_DIM` | `1200` | No | Working resolution cap for DL OCR passes. 1800px costs ~26s/pass, 1200px ~16s with identical scored-field accuracy. |
| `OCR_USE_CLS` | `false` | No | Angle classifier. Cards are deskewed upright first, so enabling it only burns ~3s/pass. |
| `STAGE1_RECOVERY_BUDGET_SECONDS` | `10.0` | No | Upscale/binarize retry passes only start inside this stage-elapsed budget; slow images degrade to Pass-1 scores instead of blowing the pipeline SLA. |
| `BLUR_VARIANCE_MIN` / `BLURRY_VARIANCE` | `40.0` / `65.0` | No | Reject below min; flag blurry below the advisory mark. |
| `GLARE_RATIO_MAX` | `0.30` | No | Reject when flash/glare covers more than this share of pixels. |
| `NOISE_VARIANCE_MAX` | `5000.0` | No | Flag noisy sensor output above this variance. |
| `VLM_MAX_TOKENS` | `900` | No | Token budget per biometric call (reasoning models need headroom). |
| `VLM_CONNECTIVITY_MAX_TOKENS` | `8` | No | Token budget for the `?live=true` ping. |
| `VLM_TEMPERATURE` | `0` | No | Deterministic verdicts; raise only for experimentation. |
| `VLM_REPAIR_CONTEXT_CHARS` | `800` | No | How much of a bad reply is echoed back on parse-retry. |
| `MAX_UPLOAD_MB` | `15` | No | Rejects larger uploads with `413`. |
| `PREVIEW_MAX_DIM` | `1400` | No | Long-edge cap for preview/compress images. |
| `TELEMETRY_DEFAULT_LIMIT` / `TELEMETRY_CACHE_TTL` | `50` / `30.0` | No | Session-list page size; telemetry snapshot freshness (seconds). |
| `FACE_CROP_PADDING_RATIO` | `0.15` | No | Padding around YuNet box; tight = stronger PII guarantee. |
| `PIPELINE_TIMEOUT_SECONDS` | `45.0` | No | Ceiling for all three stages. |
| `VLM_CALL_TIMEOUT_SECONDS` | `12.0` | No | Per provider/model attempt. |
| `VLM_TOTAL_TIMEOUT_SECONDS` | `25.0` | No | Whole VLM budget. **Must stay ≤ 60% of the pipeline timeout**; the service clamps and warns otherwise, because a slow VLM would otherwise be misreported as a pipeline timeout. |
| `VLM_PARSE_RETRY_COUNT` | `1` | No | Re-ask count when a provider returns an unparseable response. |
| `UPLOADS_DIR` | `storage/uploads` | No | Artifact root, resolved **relative to the repo root**, so it does not depend on the launch directory. |
| `API_SECRET_KEY` | generated | If auth on | Value clients send as `X-API-Key`. |
| `ENABLE_API_KEY_AUTH` | `false` | No | `true` in any public/staging/prod deployment. |
| `CORS_ALLOW_ORIGINS` | *(empty)* | No | Comma-separated allowlist. **Empty disables cross-origin browser access** — the safe default. The previous `*` + `allow_credentials=True` pairing was unsafe and has been removed. |
| `RATE_LIMIT_PER_MINUTE` | `60` | No | Per-IP sliding-window cap. `429` when exceeded. |
| `RATE_LIMIT_WINDOW_SECONDS` / `RATE_LIMIT_MAX_IPS` | `60.0` / `2000` | No | Sliding-window length and the bounded IP-cache size (memory-leak guard). |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | No | Host injects `PORT` (Render). Do not hardcode. |
| `ENVIRONMENT` | `production` | No | `production` disables reload and refuses the mock provider. |
| `OMP_NUM_THREADS` | `2` | No | Clamp OpenMP/NumPy/OpenCV threads. Applied only when unset — `.env` / host env wins over the code fallback. |
| `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` / `NUMEXPR_NUM_THREADS` | `2` | No | Same setdefault rule as above. |
| `ORT_INTRA_OP_NUM_THREADS` | `2` | No | ONNX Runtime intra-op threads. |
| `ORT_INTER_OP_NUM_THREADS` | `1` | No | ONNX Runtime inter-op threads. |

## 3. `.env` Load Precedence

Highest wins:

```
repo-root/.env   >   backend/.env   >   ./.env   (CWD)
```

- The repo-root file is **authoritative**.
- If two files define the same credential with **different values**, the service logs
  an `ERROR` at startup naming both files, and reports the collision under
  `env_sources.conflicts` on `GET /api/v1/diagnostics/vlm`.
- This replaced a previous ordering in which a CWD-relative `.env` could silently win.

## 4. Example `.env` (local dev)

Do not hand-write this — copy the template:

```powershell
Copy-Item .env.example .env
```

Minimum for a working Stage 2:

```ini
VLM_PROVIDER=openai_compatible
VLM_API_BASE_URL=https://our-llm.onrender.com/v1
VLM_API_KEY=...                  # bearer token issued by the gateway
VLM_OPENAI_MODEL=gpt-4o-mini     # vision model as named by the gateway
VLM_FALLBACK_CHAIN=openai_compatible,anthropic
```

Local fallback via a self-hosted vLLM:

```ini
VLM_API_BASE_URL=http://localhost:8000/v1
VLM_API_KEY=...
VLM_OPENAI_MODEL=Qwen2.5-VL-7B-Instruct
```

## 5. Verifying Configuration

```bash
# Offline: key presence, key SHAPE, model IDs, env precedence, timeout budget.
curl -H "X-API-Key: $API_SECRET_KEY" http://localhost:8080/api/v1/diagnostics/vlm

# Live: one minimal connectivity call per ready provider.
curl -H "X-API-Key: $API_SECRET_KEY" "http://localhost:8080/api/v1/diagnostics/vlm?live=true"

# Subsystem health (gates on VLM readiness).
curl http://localhost:8080/health

# Liveness — deliberately excludes the VLM.
curl http://localhost:8080/live
```

`key_shape_mismatches` in the diagnostics response is the fastest way to spot the
most common misconfiguration: a credential that does not match its provider
(e.g. an Anthropic `sk-ant-` key placed in `VLM_API_KEY`, or a vendor key placed where
the gateway token belongs).

## 6. Tuning Guidance

- **Stage 2 always rejects**: check `verdict` and `vlm_attempt_chain` in the response
  before touching thresholds. `SERVICE_UNAVAILABLE` is an infrastructure fault, not an
  identity decision. See `OPERATIONS_RUNBOOK.md`.
- **Too many false rejects on names** (transliteration, initials): lower `FUZZY_NAME_THRESHOLD` to `0.80`. Raising above `0.90` sharply increases manual failures.
- **Too strict overall**: `COMPOSITE_APPROVAL_THRESHOLD` is `0.80`. Lower it further only with fraud-team sign-off.
- **Face crop leaks text**: lower `FACE_CROP_PADDING_RATIO` to `0.10–0.12` and re-audit via `/privacy-crop-preview`.
- **429s under normal traffic**: raise `RATE_LIMIT_PER_MINUTE` or move to Redis-backed limiting. The limiter is per-process and resets on deploy.
- **OOM on 512 MB instances**: keep thread clamps at 1–2; do not raise workers without raising RAM.
