# Configuration — Environment Variables

> Single source of truth for every tunable. Sources: `.env` (local), Render env vars (prod), defaults in `backend/app/config.py`.

## 1. Full Reference

| Variable | Default (`config.py` / `render.yaml`) | Required | Description |
|---|---|---|---|
| `VLM_PROVIDER` | `gemini` | Yes | `gemini` \| `anthropic` \| `mock`. `mock` = no cloud call (tests/staging). |
| `VLM_MODEL` | `gemini-2.5-flash` | Yes | Model ID sent to provider. |
| `GEMINI_API_KEY` | — | If provider=gemini | Google AI key. Keep secret. |
| `ANTHROPIC_API_KEY` | — | If provider=anthropic | Anthropic key. Keep secret. |
| `FUZZY_NAME_THRESHOLD` | `0.85` | No | RapidFuzz token-sort pass bar for full name. |
| `FUZZY_DL_THRESHOLD` | `0.88` | No | Levenshtein pass bar for licence number. |
| `COMPOSITE_APPROVAL_THRESHOLD` | `0.85` | No | Composite score bar for `APPROVED`. Formula `(0.30×S1)+(0.45×S2)+(0.25×S3)`. |
| `FACE_CROP_PADDING_RATIO` | `0.15` | No | Padding around YuNet box; tight = stronger PII guarantee. |
| `API_SECRET_KEY` | generated | If auth on | Value clients send as `X-API-Key`. Generate 32+ random chars. |
| `ENABLE_API_KEY_AUTH` | `false` | No | `true` in any public/staging/prod deployment. |
| `RATE_LIMIT_PER_MINUTE` | `60` | No | Per-IP sliding-window cap. `429` when exceeded. |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | No | Host injects `PORT` (Render). Do not hardcode. |
| `ENVIRONMENT` | `production` | No | `production` disables reload; anything else enables `--reload`. |
| `OMP_NUM_THREADS` | `2` | No | Clamp OpenMP/NumPy/OpenCV threads (also hardcoded in `main.py`). |
| `ORT_INTRA_OP_NUM_THREADS` | `2` | No | ONNX Runtime intra-op threads. |
| `ORT_INTER_OP_NUM_THREADS` | `1` | No | ONNX Runtime inter-op threads. |
| `PYTHON_VERSION` | `3.11.9` | Render only | Pinned runtime. |
| `uploads_dir` | `storage/uploads` | No | Session artifact root (Settings field, rarely overridden). |

## 2. Example `.env` (local dev)

```ini
VLM_PROVIDER=gemini
VLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=<your-key-here>
ANTHROPIC_API_KEY=

FUZZY_NAME_THRESHOLD=0.85
FUZZY_DL_THRESHOLD=0.88
COMPOSITE_APPROVAL_THRESHOLD=0.85
FACE_CROP_PADDING_RATIO=0.15

API_SECRET_KEY=<generate-strong-random-value>
ENABLE_API_KEY_AUTH=true
RATE_LIMIT_PER_MINUTE=60

HOST=0.0.0.0
PORT=8080
ENVIRONMENT=production
```

> The repo ships a committed `.env` with a live-looking key for convenience — **rotate it immediately** and never commit real secrets. Use Render secret env vars in production.

## 3. Tuning Guidance

- **Too many false rejects on names** (transliteration, initials): lower `FUZZY_NAME_THRESHOLD` to `0.80`, or add alias handling upstream. Raising above `0.90` sharply increases manual failures.
- **Too strict overall**: lower `COMPOSITE_APPROVAL_THRESHOLD` to `0.80` only with fraud-team sign-off; every 0.05 drop measurably increases false accepts.
- **Face crop leaks text**: lower `FACE_CROP_PADDING_RATIO` to `0.10–0.12` and re-audit via `/privacy-crop-preview`. Raising improves match rate on angled cards but risks PII leakage.
- **429s under normal traffic**: raise `RATE_LIMIT_PER_MINUTE` or move to Redis-backed limiting (see Runbook). Lower it if scraping is observed.
- **OOM on 512 MB instances**: keep thread clamps at 1–2; do not raise workers without raising RAM.

## 4. Validation

```bash
curl http://localhost:8080/api/v1/telemetry/system
# pipeline_config echoes live thresholds — confirm they match your env
```
