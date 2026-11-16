# Face match benchmark: AgeDB-30

Generated 2026-11-12 10:14 UTC by `scripts/bench_public.py` on this machine. Data: AgeDB-30 (6000 pairs: 3000 same person about 30 years apart, 3000 different people; pre-aligned 112x112 crops).

**Model:** `w600k_r50.onnx`, production `embed()` (no flip augmentation, so slightly below published numbers).
**Not measured here:** face detection and alignment (YuNet), because the crops are already aligned; liveness.

## Standard AgeDB-30 accuracy (10-fold)

- Accuracy: **98.03%** ± 0.58 (per-fold thresholds 0.195–0.197).

## At the production threshold (0.40 cosine)

- Same person accepted (true accept): 74.13% (2224/3000; 95% CI 72.53–75.69).
- Same person rejected (false reject, ~30-year gap): 25.87% (776/3000; 95% CI 24.31–27.47).
- Different people accepted (false accept): 0.00% (0/3000; 95% CI 0.00–0.12).
- Overall accuracy at 0.40: 87.07%.

## Score distribution

- Same person: median 0.468, 5th percentile 0.252, lowest -0.108.
- Different people: median 0.014, 95th percentile 0.127, highest 0.268.

| Threshold | False reject (same person) | False accept (different people) |
|---|---|---|
| 0.25 | 4.90% (147/3000; 95% CI 4.16–5.73) | 0.07% (2/3000; 95% CI 0.01–0.24) |
| 0.30 | 8.43% (253/3000; 95% CI 7.46–9.49) | 0.00% (0/3000; 95% CI 0.00–0.12) |
| 0.35 | 14.63% (439/3000; 95% CI 13.39–15.95) | 0.00% (0/3000; 95% CI 0.00–0.12) |
| 0.40 | 25.87% (776/3000; 95% CI 24.31–27.47) | 0.00% (0/3000; 95% CI 0.00–0.12) |
| 0.45 | 43.47% (1304/3000; 95% CI 41.68–45.26) | 0.00% (0/3000; 95% CI 0.00–0.12) |
| 0.50 | 61.87% (1856/3000; 95% CI 60.10–63.61) | 0.00% (0/3000; 95% CI 0.00–0.12) |

- Speed: 4.1 faces/s using 1 worker processes x 2 ONNX thread(s) (49.4 min for 12000 faces). One production verification embeds 2 faces on 2 threads.
- Limit: 3000 impostor pairs can bound the false-accept rate only to about 0.12% (95% upper bound). The 0.01% target needs about 30,000 impostor trials.
