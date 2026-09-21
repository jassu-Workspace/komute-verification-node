# Security & Privacy

> Threat model, controls, PII handling, and compliance notes. Read before exposing this service publicly or integrating into a production app.

## 1. Trust Boundary: Zero-PII Biometrics

```
FULL LICENCE IMAGE (PII: name, DL no, DOB, address, signature)
        |  local only — never transmitted
        v
YuNet face detect + tight crop (padding 0.15) — backend/core/face_privacy_cropper.py
        |
        v
ONLY [selfie_face_crop, dl_face_crop] → Gemini / Claude
(Zero text, zero numbers, zero address leaked)
```

- Full licence, selfie background, and vehicle photo never leave the host except as cropped faces to the VLM.
- Audit anytime: `POST /api/v1/privacy-crop-preview` returns `face_crop_base64` + `pii_sanitized:true` + saved artifacts.
- Rejected verifications skip licence WebP archival (`pipeline.py`) — privacy minimization by default.

## 2. Authentication & Rate Limiting

Source: `backend/app/security.py`, wired in `backend/app/routes.py`.

- **API key** — `ENABLE_API_KEY_AUTH=true` requires `X-API-Key` header, verified with `secrets.compare_digest` (constant-time). Missing secret on server → `500` (fail-closed config check). Missing/wrong key → `401`.
- **Rate limit** — sliding 60s window per client IP (`X-Forwarded-For`-aware for Render/Cloudflare), default 60 req/min → `429`. Bounded to 2000 tracked IPs with dead-key eviction to prevent memory exhaustion.
- **Production rule**: always `ENABLE_API_KEY_AUTH=true` with a 32+ char random `API_SECRET_KEY`, rotated on leak or staff change. Never ship `false` publicly (current `render.yaml` default is `false` — flip it).

## 3. Input & Storage Hardening

- **Upload caps**: `/compress` rejects non-`image/*` (`400`) and files >15 MB (`413`).
- **Traversal protection**: `/storage/{file_path}` resolves against `uploads_dir` and returns `403` on escape, `404` on missing (tested in `test_audit_security.py`).
- **Header hygiene**: compressed responses set `Cache-Control: public, max-age=31536000, immutable`; dashboards send `no-cache, no-store`.
- **PII in logs**: use `mask_pii()` for DL numbers/phones; never log base64 images or full licence text at INFO.
- **EXIF stripping**: compressor normalizes orientation and strips metadata before storage/transmission.

## 4. Secrets Management

- `.env` is gitignored — real keys live in host secret store (Render env vars with `sync: false`).
- Rotate `GEMINI_API_KEY` / `API_SECRET_KEY` immediately if the sample `.env` in this repo was ever deployed.
- Separate keys per environment (dev / staging / prod). `VLM_PROVIDER=mock` for CI so no secret is needed.

## 5. Compliance Alignment (GDPR / DPDP / PIPEDA)

This design supports — but does not by itself certify — compliance:

- **Data minimization**: only face crops to VLM; rejected licences not archived.
- **Purpose limitation**: artifacts under `storage/uploads/sessions/{driver}_{request}/` with `session_metadata.json` for audit traceability.
- **Retention**: define and enforce a purge job (e.g. delete sessions older than N days) before launch — currently storage grows unbounded on ephemeral disk.
- **Access control**: auth + TLS (Render) + need-to-know artifact access.
- **DPA**: execute DPAs with Google/Anthropic as sub-processors; confirm data-residency (Render `oregon`) meets your policy.

## 6. Pre-Launch Hardening Checklist

- [ ] `ENABLE_API_KEY_AUTH=true`, strong rotated `API_SECRET_KEY`
- [ ] CORS in `main.py` restricted from `["*"]` to production origins
- [ ] HTTPS only (Render default), HSTS via proxy/CDN
- [ ] Centralized log redaction (`mask_pii`) + no base64 in logs
- [ ] Retention/purge cron defined and tested
- [ ] `pytest backend/tests/test_audit_security.py backend/tests/test_concurrency_stress.py -v` green
- [ ] Abuse tests green: no-key `401`, wrong-key `401`, 61-burst `429`, `../` traversal `403`
