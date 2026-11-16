# Face re-run: AgeDB-30 false rejects

Generated 2026-11-12 15:03 UTC by `scripts/bench_face_folder.py` on this machine. Data: 776 same-person AgeDB-30 pairs (1552 images) from `false_rejects_0.40` (rejected at 0.40 in the full benchmark). Threshold tested: 0.35.

**Model:** `w600k_r50.onnx`, production `embed()`, 1 process x 10 ONNX thread(s).

## Result

- Re-run on the stored crops: 337 of 776 pairs now pass 0.35. Largest change from the first run: 0.0000 (the model is deterministic).
- Full production path (YuNet detect, align, embed): faces found in both images for 767 of 776 pairs; 251 of those pass 0.35.
- With test-time flip (not in production): 378 of 776 pairs pass 0.35.

## Score bands (pairs)

| Path | < 0.00 | 0.00–0.10 | 0.10–0.20 | 0.20–0.30 | 0.30–0.35 | 0.35–0.40 | ≥ 0.40 |
|---|---|---|---|---|---|---|---|
| Crop (production embed) | 20 | 31 | 49 | 153 | 186 | 337 | 0 |
| Full path (767 pairs) | 19 | 30 | 56 | 184 | 227 | 205 | 46 |
| Crop + flip | 21 | 29 | 44 | 135 | 169 | 299 | 79 |

## Pairs recovered by a lower threshold (crop path)

| Threshold | Pairs that would pass | Share of these 776 | False accepts in the full benchmark |
|---|---|---|---|
| 0.25 | 629 | 81.1% | 2 of 3,000 |
| 0.30 | 523 | 67.4% | 0 of 3,000 |
| 0.35 | 337 | 43.4% | 0 of 3,000 |
| 0.38 | 146 | 18.8% | 0 of 3,000 |

## Summary statistics

- Crop: median 0.341, lowest -0.108, highest 0.400.
- Full path: median 0.323, lowest -0.103, highest 0.458; mean change versus crop -0.011.
- Flip: median 0.348; mean change versus crop +0.008.
- Time: 17.4 min for 776 pairs.

**Limit:** these are AgeDB-30 pairs about 30 years apart. Licence-to-selfie gaps are usually under 10 years. Per-pair scores: `rerun_scores.csv` next to the images (outside the repository).
