# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state (read first)

Komüte-Local is a driver and vehicle verification module that a separate host application calls. **It is built:** engine (`backend/core`), REST wrapper (`backend/app`), encrypted storage (`storage/`), one-file test page (`backend/static/index.html`), evaluation and bench scripts (`scripts/`), tests (`tests/`) and a Postman collection (`postman/`, see `postman/README.md`). Design: `docs/superpowers/specs/2026-11-11-komute-local-verification-design.md`. Build plan: `docs/superpowers/plans/2026-11-11-komute-local-master-plan.md`. Results and open items: `docs/ACCEPTANCE_REPORT.md`. Model facts: `docs/MODEL_FACTS.md`. The folder is a git repository on branch `feat/verification-engine` (no remote; never push).

**Private data:** `Original Lisence's/` holds the owner's real licence photos. It is git-ignored. Never print, log, commit or paste anything read from them (names, numbers, dates, addresses, raw barcode text, face crops). Copies live in `$LOCALAPPDATA/komute/samples/raw/`.

## Scope (agreed with the owner; do not relitigate)

- Focus: backend architecture, storage architecture, verification quality. The host app owns rate limiting, tenants, billing and product UI. **Do not add rate limits.**
- Frontend is one HTML file with inline CSS and script calls, for testing only. No dashboard, no operator console.
- The whole verification fits in about 1.0–1.3 GB RAM. Target: steady-state RSS ≤ 900 MB, peak ≤ 1.2 GB. Do not add Redis, Postgres or Celery.
- One verification at a time per container (concurrency = 1). A busy container returns `503` with `Retry-After`. This is a memory guard, not a rate limit. The host app scales by adding containers.
- The outcome is strictly `APPROVED` or `REJECTED`. It is fully autonomous. **There is no manual review and no review state.** Any doubt means `REJECTED` with a reason code. A crashed stage or a failed model load returns an error, never a decision, and never `APPROVED`.
- Keep the three-stage workflow. Add single-photo passive liveness to Stage 2. The selfie is one photo only, so document the liveness limits honestly.
- Every session gets a signed manifest: a SHA-256 per file, signed with HMAC-SHA256.
- No licence front-to-back cross-check and no OCR of the licence front.

## Commands

Python 3.12 only (installed by `uv`). The venv lives outside OneDrive. Run from the repo root in Git Bash.

`.\setup.ps1` sets up a fresh Windows machine end to end: uv, Python 3.12, venv, `requirements.lock`, the four models (downloaded from their public sources and rejected unless the SHA-256 matches `models.lock`), dev keys, then the tests (`-SkipTests`, `-Force`). One command does the checks and starts the app on Windows: `.\run.ps1` (`-CheckOnly`, `-Port`, `-NoBrowser`, `-WithApiKey`). It installs missing dependencies from `requirements.txt` with uv and verifies model files against `models.lock`. Ubuntu 22.04/24.04 server: `sudo ./setup.sh` (user `komute`, code in `/opt/komute/app`, venv `/opt/komute/venv`, settings `/etc/komute/komute.env`, data `/var/lib/komute`, systemd service `komute` that runs `./start.sh`; `--no-start`). Guides: `docs/guides/*.pdf`, built by `docs/guides/build_guides.py`. Manual steps:

```bash
uv python install 3.12
uv venv --python 3.12 "$LOCALAPPDATA/komute/venv"
uv pip install --python "$LOCALAPPDATA/komute/venv" -r requirements.txt   # requirements.lock = exact frozen set
python scripts/make_dev_keys.py          # once; writes $LOCALAPPDATA/komute/dev.env, refuses to overwrite
source scripts/dev_env.sh                # exports $PY, KOMUTE_* env vars, PYTHONUTF8=1
"$PY" -m pytest -m "not samples"         # unit + API tests (about 1 minute)
"$PY" -m pytest tests/e2e                # real-licence end-to-end (needs samples; about 2 minutes)
"$PY" -m pytest tests/core/test_gates.py::test_underage_17 -v   # single test
KOMUTE_DEV_SKIP_LIVENESS=true "$PY" -m uvicorn backend.app.main:create_app --factory --workers 1   # keep workers=1
"$PY" scripts/bench.py --runs 50         # memory and latency on the real cards
"$PY" scripts/evaluate.py barcode --manifest cards.csv --out docs/calibration/barcode.md
```

Markers: `models` (needs ONNX files), `samples` (skipped unless `KOMUTE_SAMPLES_DIR` exists and is non-empty). `import backend.core` must come before `import cv2` (it sets the pixel clamp and fails closed otherwise). Do not use `cat <<EOF` with several files in one Bash call: the shell here mis-parses it; write files with the Write tool.

## Architecture (big picture)

The host sends claimed data (name, date of birth, licence number, issuing province, plate, colour), a consent timestamp with its policy version, and four images (selfie, licence front, licence back, vehicle) to `POST /v1/verify`. Use multipart, not base64, to save RAM. The call is synchronous and idempotent by `request_id`. Authentication is OFF by default (testing); when `KOMUTE_API_KEY` is set the header `X-API-Key` is required. The main team's contract is `docs/INTEGRATION_GUIDE.md`. The engine is a package with no FastAPI imports, so it can also be embedded as a library. The REST layer is a thin wrapper.

Ingest: cap every image to a fixed maximum size on decode, then write the originals to storage before any stage runs, so a crash cannot lose evidence.

Stages, with early exit on any Stage 1 failure:

1. **Barcode (Stage 1).** Decode the AAMVA PDF417 on the licence back. Parse the header and subfile directory (`DL` plus provincial subfiles such as Ontario `ZO`). Resolve date order by `DCG` (`CAN` is CCYYMMDD, `USA` is MMDDCCYY). The barcode is the ground truth. An unreadable barcode returns `REJECTED`, retryable, with the message that the barcode is not clearly visible and must be retaken. Then run the jurisdiction, name, date of birth, licence number, expiry, age, class and restriction gates.
2. **Biometrics (Stage 2).** YuNet finds 5 landmarks and an affine transform aligns the face to 112×112. Liveness runs on the selfie first. The face embedding is then compared by cosine similarity against a calibrated threshold. The licence front is used only for the face crop.
3. **Vehicle (Stage 3).** YOLOv9-tiny finds the plate and OCR reads it, restricted to A–Z and 0–9. The plate rule below decides the match. CIELAB ΔE classifies the body colour from sheet-metal patches next to the plate. The colour is advisory.
4. **Audit and storage.** Compress images, write crops, write the signed manifest. The pipeline talks only to a storage interface (local disk and S3-compatible implementations). `storage/` must never become an inference dependency.

Decision engine: hard gates only, no composite score. Any failed gate gives `REJECTED` and lists every failed reason code. Each code is marked **retryable** (a better photo could change the result, for example unreadable barcode, no face found, plate not found, poor image quality) or **final** (for example expired, under 18, name mismatch, disallowed class). Thresholds come from data, not from the docs. The docs' weights (0.30/0.45/0.25) and "composite ≥ 80%" are uncalibrated and must not ship.

## Memory rules for any pipeline code

- Stage 1 generates barcode image variants one at a time. Do not pre-build all 19. The docs record a +507 MB surge from doing so.
- Set ONNX session options for a small memory arena. Pin threads: `OMP_NUM_THREADS=2`, `ORT_INTRA_OP_NUM_THREADS=2`, `ORT_INTER_OP_NUM_THREADS=1`, `cv2.setNumThreads(2)`.
- The docs' Windows heap numbers do not transfer to Linux containers. Measure on the target OS.
- The owner's "512 MB floor" is not credible. The docs' own profiling shows a 1.23 GB peak. Keep the 1.0–1.3 GB budget.

## Policy rules (owner-specified 2026-11-11)

Legal citations in the owner's answers (Toronto Chapter 546, Quebec SAAQ rules, Quebec Law 25 "Section 12") are unverified. Get legal review before relying on them.

**Eligibility**
- One profile: minimum age 18, computed from the barcode date of birth. No commercial profile.
- Owner ruling (2026-11-11): **no class, restriction or province limits.** Any licence class and any restriction code is accepted, in all 13 jurisdictions. The only eligibility gates are age 18 or older (barcode date of birth) and an unexpired licence. The jurisdiction check (claimed province = IIN = `DAJ`) stays because it attributes the card to one region.
- Expiry: zero grace days. An expiry date before the verification date fails. Assumed: valid through the end of the expiry date, compared in the issuing jurisdiction's time zone.
- Suspension and revocation are out of scope. The host queries provincial driver records.

**Matching**
- Name: strip accents (Unicode normalisation), turn punctuation and hyphens into spaces, then token-sort fuzzy match with a 90% pass (owner, 2026-11-12; was 85%). Known weakness: 90% still accepts near-names such as `SMITH` and `SMYTH` (score exactly 90). A stricter exact-surname rule is recommended but not adopted.
- Date of birth: exact match to the barcode, zero tolerance.
- Class, issue date and expiry date (owner, 2026-11-12): the host sends them; each must equal the barcode (`DCA`, `DBD`, `DBA`) exactly. An issue date after today fails (`LICENCE_NOT_YET_VALID`). Each cross-checked field returns a check row with a 0–100 score in `stages.barcode.checks` (personal data in the response). The full decoded barcode is returned only with `KOMUTE_RESPONSE_BARCODE_FIELDS=true`.
- Plate: it matches only if the strings are equal after normalisation, or differ by one optical-confusion pair (`O`/`0`, `I`/`1`, `B`/`8`). Any other difference is a mismatch.
- Jurisdiction: the claimed province, the IIN and `DAJ` must agree, or the result is `REJECTED`.

**Vehicle**
- Verify plate and body colour only. Make, model and year are stored as metadata.
- A colour mismatch writes an advisory flag to the audit record and never blocks approval. A plate match alone passes the vehicle stage. Known risk: a photo of any car showing the right plate string passes.

**Data handling**
- Store three tiers per session: original uploads, application crops, and compressed images made after verification. The compressed full-card image holds all card text. The owner accepts that.
- Rejected sessions: delete all images within 72 hours at most. Keep an image-free signed audit record: request id, decision, reason codes, scores, file hashes, consent timestamp. How long to keep that record needs legal advice.
- Approved sessions: delete raw originals after 30 days. Keep compressed images, face crops and plate crops. Keep the signed manifest for the driver's tenure plus 7 years.
- TLS 1.3 in transit, AES-256 at rest. At-rest keys come from the host's key management service through environment variables or a key-vault sidecar. The node never generates keys.
- The manifest key is injected by the host as an environment variable. Include a key id so keys can rotate.
- Consent: the host forwards the consent timestamp and policy version with every request. The node seals both into the signed audit record. A request without them returns an error, not a decision (assumed).

**Accuracy targets**
- False-approve rate ≤ 0.01%. False-reject rate ≤ 3–5% on the first try, under 1% after a retry.
- Proving 0.01% needs about 30,000 independent impostor trials for a 95% bound. Under-20 people cannot show that. Public datasets can bound the face component. Report end-to-end rates with confidence ranges. Do not claim rates the data cannot support.
- Test data: the owner has labelled samples from under 20 people. Supplement with public datasets (check each licence). Split tuning and held-out sets by person. Real licence images are personal data: keep them out of the repo.

## Models

Licensing is closed by the owner (2026-11-11): licensed production models are in hand, so no licence checks are needed. If the production builds differ from the files in `backend/models/`, replace the files and record each file's SHA-256 in `backend/models/models.lock`. The engine refuses to start when a hash does not match.

| Role | File | Notes |
|---|---|---|
| Face detection and 5 landmarks | `face_detection_yunet_2023mar.onnx` | |
| Face embedding (512-d) | `w600k_r50.onnx` | Threshold calibrated on data. |
| Liveness | not in the repo | MiniFASNet is the candidate. The owner supplies or approves one. |
| Plate detection | `yolo-v9-t-384-license-plates-end2end.onnx` | |
| Plate OCR | `cct_xs_v2_global.onnx` | cct-xs-v2-global-model (MIT): ten character slots over 37 classes, not CTC. Replaced PP-OCRv5 (read 1 of 9 real plates exactly; this reads 9 of 9). |
| Age estimation | `genderage.onnx`, `mivolo_v2_age.onnx` | Unused. Dropped by default. |

Python dependency licences were never audited. The owner has put licensing out of scope.

## Defaults adopted without owner confirmation

The owner asked to skip the remaining questions. These are the defaults I chose. Each one is reversible.

1. **Age estimation is dropped.** Identity rests on the face match and the barcode date of birth. This removes `genderage.onnx` (blocked) and `mivolo_v2_age.onnx` (unresolved) and frees about 120 MB of RAM. The docs' `estimated_age_delta_years` field is not produced.
2. **Face model.** `w600k_r50.onnx` stays (licensed per the owner). Its threshold is calibrated on data.
3. **Plate OCR.** `cct_xs_v2_global.onnx` (the owner replaced PP-OCRv5 after testing). Input 128x64 RGB uint8, ten slots, confidence is the lowest slot probability.
4. **Plate detector.** Keep the YOLOv9-tiny file.
5. **Liveness.** Candidate: MiniFASNet (Apache-2.0 per a third-party index; training data undocumented). Benchmark it against print and screen-replay attacks before adopting. A cloud vision API for face crops is not adopted.
6. **Licence-number regexes are a sanity check only.** The barcode `DAQ` value is authoritative. Several owner patterns are looser than outside sources (see the Jurisdictions section).
7. **Barcode on every licence is assumed**, since the owner's design depends on it. It is unverified per issuer.

## Jurisdictions (13, all Canadian)

The issuing jurisdiction is identified by the AAMVA IIN in the barcode header. All 13 IINs were checked against the official AAMVA registry (https://www.aamva.org/identity/issuer-identification-numbers-(iin)) on 2026-11-11 and match. AAMVA lists Newfoundland and Labrador as `NF`, not `NL`. Accept both in `DAJ` and in the claimed province, and normalise to `NL`. Not verified: that all 13 issuers print an AAMVA PDF417. Confirm each with real samples.

| Code | Region | IIN |
|---|---|---|
| ON | Ontario | 636012 |
| QC | Quebec | 604428 |
| BC | British Columbia | 636028 |
| AB | Alberta | 604432 |
| MB | Manitoba | 636048 |
| SK | Saskatchewan | 636044 |
| NB | New Brunswick | 636017 |
| NS | Nova Scotia | 636013 |
| NL | Newfoundland and Labrador | 636016 |
| PE | Prince Edward Island | 604426 |
| YT | Yukon | 604429 |
| NT | Northwest Territories | 604434 |
| NU | Nunavut | 604433 |

Licence-number formats from the owner, matched after normalising to uppercase and removing hyphens (do not strip `*` or the spaces `NS` allows):

```text
ON ^[A-Z]\d{14}$
QC ^[A-Z]\d{12}$
BC ^\d{7,8}$
AB ^(?:\d{6,9}|[A-Z]{1,2}\d{4,5})$
MB ^(?:[A-Z][A-Z*]{4}[A-Z][A-Z*][A-Z0-9]{5,6}|[A-Z]\d{12,13})$
SK ^\d{8}$
NB ^\d{1,7}$
NS ^[A-Z][A-Z* ]{4}\d{9}$
NL ^[A-Z]\d{9}$
PE ^\d{1,6}$
YT ^\d{1,6}$
NT ^\d{5,9}$
NU ^\d{5,6}$
```

Compared on 2026-11-11 with a secondary copy of the US FMCSA chart (August 2021) and Microsoft Purview's patterns. The official FMCSA PDFs were unreachable, and the chart may be out of date:

- **Match both sources:** ON, QC, SK, NL.
- **BC:** sources say exactly 7 digits. The owner's pattern allows 7–8.
- **AB:** FMCSA says 9 digits. Microsoft says 5–9, or 6 digits, hyphen, 3 digits. The letter-prefixed alternative has no outside source.
- **MB:** FMCSA says 13 characters. Microsoft describes 12. The `[A-Z]\d{12,13}` alternative has no outside source.
- **NB:** sources say 7 (FMCSA) or 5–7 (Microsoft). The owner's pattern allows 1–7.
- **NS:** FMCSA allows only letters and `*` in the first five characters. The owner's pattern also allows a space.
- **PE:** sources say 5 or 5–6. The owner's pattern allows 1–6.
- **YT:** FMCSA says 6. The owner's pattern allows 1–6.
- **NT:** FMCSA says 9. The owner's pattern allows 5–9.
- **NU:** FMCSA says 5. The owner's pattern allows 5–6.

## Known gaps and open items

- **Liveness:** no liveness model file exists. Dev runs skip it (advisory `LIVENESS_SKIPPED_DEV`). `OnnxLiveness` is untested on a real model and its parameters are unverified. Production start requires one.
- **Face threshold:** 0.35 (owner, 2026-11-12; was 0.40). AgeDB-30: 0 of 3,000 false accepts (highest impostor 0.268), 14.6% false rejects on ~30-year gaps. 3,000 impostor pairs cannot prove the 0.01% target; confirm on real licence-selfie pairs. Fine-tuning on CACD was tried and made it worse (docs/calibration).
- **Docker:** the owner decided (2026-11-11) NOT to use Docker until the module is handed to the main developers. Do not build, run or start Docker. Memory is measured locally with `scripts/bench.py` (Windows: steady about 334 MB, peak about 383 MB). The `Dockerfile` exists but has never been built or run; Linux memory is unmeasured until handover.
- **Plate OCR and detector:** the OCR reads 9 of 9 internet photos of Canadian plates exactly (tuned on nothing; the photos are in `$LOCALAPPDATA/komute/samples/plates`, stock images, never commit them). Plates smaller than about 5% of the frame width were not detected in a quick test. No real onboarding vehicle photo has been tested.
- **Docs:** `docs/superpowers/` holds the original spec and plan, which predate several decisions (see the ledger rulings). The other `docs/*.md` files describe the built system.
- **Not built:** S3 storage, multi-frame liveness, OCR fallback, age estimation, rate limiting (the host's job).

## Inputs still needed from the owner

- Real sample licences, to confirm each issuer prints a barcode.
- A decision on the 90% name rule versus an exact-surname rule, and on tightening the licence regexes.
- Legal review of the citations above, of the hand-controls rule, and of how long to keep the image-free audit record.
- A confirmed origin and SHA-256 for each ONNX file, recorded in `backend/models/models.lock`.
- A liveness model file, or approval of a candidate.
