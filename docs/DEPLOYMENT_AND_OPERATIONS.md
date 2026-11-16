# Deployment and operations

## Run

```bash
source scripts/dev_env.sh          # development: loads keys from $LOCALAPPDATA/komute/dev.env
KOMUTE_DEV_SKIP_LIVENESS=true uvicorn backend.app.main:create_app --factory --host 0.0.0.0 --port 8000 --workers 1
```

Keep `--workers 1`: one verification at a time per container, and the models load once. Scale by adding containers. The test page is at `/`.

## Settings (environment, prefix `KOMUTE_`)

| Variable | Meaning |
|---|---|
| `ENV` | `dev` or `production`. Production requires `LIVENESS_MODEL` and refuses `DEV_SKIP_LIVENESS` |
| `API_KEY` | optional. Unset or empty means **no authentication** (testing). When set, callers must send it in `X-API-Key`. Required, 24+ characters, in production |
| `STORAGE_ROOT` | where sessions live. **Never inside OneDrive**; settings reject such a path |
| `STORAGE_KEY_B64`, `STORAGE_KEY_ID` | 32-byte base64 data key and its id, from the host's key service |
| `MANIFEST_KEY_B64`, `MANIFEST_KEY_ID` | 32-byte base64 signing key and its id |
| `LIVENESS_MODEL` | path to the liveness ONNX file; add its SHA-256 to `backend/models/models.lock` |
| `FACE_MATCH_THRESHOLD` (0.35), `LIVENESS_THRESHOLD` (0.5), `PLATE_MIN_CONFIDENCE` (0.6) | decision thresholds. The first two are provisional |
| `MAX_UPLOAD_BYTES` (10 MB), `MAX_IMAGE_SIDE` (2048), `COMPRESSED_MAX_SIDE` (1280) | size limits |
| `REJECTED_IMAGES_TTL_H` (72), `APPROVED_ORIGINALS_TTL_D` (30), `SWEEP_INTERVAL_S` (3600) | retention |
| `RETRY_AFTER_S` (5) | value of `Retry-After` on `503` |

## Endpoints

`POST /v1/verify` (see the README for the contract), `GET /v1/sessions/{request_id}` (stored response; key only if configured), `GET /health` (no key: models loaded and RSS; `503` when a required model is missing), `GET /`.

## Container

`Dockerfile` builds from `python:3.12-slim`, installs `requirements.txt`, runs as a non-root user with one worker, and uses `MALLOC_ARENA_MAX=2`. `.dockerignore` keeps samples, docs, tests, secrets and the unused age models out. **It has not been built or run, by decision:** the owner deferred Docker until the module is handed to the main developers. Until then, measure memory locally with `scripts/bench.py`. At handover, build and memory-check it with:

```bash
docker build -t komute-local .
docker run --rm --memory=1300m --cpus=2 -e KOMUTE_ENV=dev -e KOMUTE_DEV_SKIP_LIVENESS=true \
  --env-file "$LOCALAPPDATA/komute/dev.env" -v "$LOCALAPPDATA/komute/samples:/samples:ro" \
  -e KOMUTE_SAMPLES_DIR=/samples komute-local python scripts/bench.py --runs 50 --out /tmp/bench.md
```

## Troubleshooting

- **Startup stops with a hash error:** a model file differs from `models.lock`. Restore the file or re-lock after review.
- **Startup stops with "clamp inactive":** something imported OpenCV before `backend.core`. Import `backend.core` first.
- **`503 BUSY`:** a verification is running. Retry after the `Retry-After` seconds, or route to another container.
- **`BARCODE_UNREADABLE` on good cards:** ask for a sharp, well-lit photo of the back with the barcode filling a third of the frame.
- **Class or restriction rejections on a real card:** the province's table in `backend/core/policy.py` does not list that code yet.
- **Data in the wrong place:** delete `STORAGE_ROOT` contents you do not want. Nothing else depends on them except replay and audit.
