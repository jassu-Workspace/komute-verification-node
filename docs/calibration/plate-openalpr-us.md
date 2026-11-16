# Plate benchmark: OpenALPR US

Generated 2026-11-12 06:20 UTC by `scripts/bench_public.py` on this machine. Data: OpenALPR US end-to-end set (222 real US car photos, one labelled plate each).

**Models:** `yolo-v9-t-384-license-plates-end2end.onnx` (detector), `cct_xs_v2_global.onnx` (OCR). Production functions and thresholds: detector score 0.35, OCR minimum confidence 0.6.
**Caveat:** US plates, not Canadian. Only the first labelled plate per photo is scored.

## End to end (what Stage 3 does)

- Plate found: 99.55% (221/222; 95% CI 97.52–99.99); found at the labelled place (IoU ≥ 0.5): 98.65% (219/222; 95% CI 96.10–99.72).
- Exact read: 87.39% (194/222; 95% CI 82.29–91.45).
- Passes the production plate rule with enough confidence (would match the claim): 88.74% (197/222; 95% CI 83.83–92.58).
- Read confident enough to use: 94.14% (209/222; 95% CI 90.19–96.85).

## OCR alone (on the labelled plate crop)

- Exact read: 88.74% (197/222; 95% CI 83.83–92.58).

## Small plates and speed

- Plates under 5% of the photo width: 18; found 17, read exactly 14.
- Detect + read time: median 61 ms, p95 127 ms per photo.
