# Stage 1 benchmark on fake licences

Generated 2026-11-12 05:58 UTC by `scripts/bench_barcode.py`. Data: 650 fake licence backs from `scripts/make_fake_licences.py` (seed 20261112, verification date 2026-11-12). No real personal data.

**What this measures:** the production decoder, parser and Stage 1 gates on synthetic barcodes rendered and degraded to mimic phone photos. **What it does not measure:** real cards. The barcodes come from our own generator, so a pass here proves the pipeline handles the AAMVA layout and these degradations, not that every issuer's real card decodes. 10 of 13 layouts are assumed (see the profile column).

## By region

| Group | Samples | Barcode decoded | Every field exact | Honest claim passes Stage 1 |
|---|---|---|---|---|
| AB (assumed) | 50 | 84.0% (42/50; 95% CI 70.9–92.8) | 84.0% (42/50; 95% CI 70.9–92.8) | 84.0% (42/50; 95% CI 70.9–92.8) |
| BC (assumed) | 50 | 88.0% (44/50; 95% CI 75.7–95.5) | 88.0% (44/50; 95% CI 75.7–95.5) | 88.0% (44/50; 95% CI 75.7–95.5) |
| MB (assumed) | 50 | 84.0% (42/50; 95% CI 70.9–92.8) | 80.0% (40/50; 95% CI 66.3–90.0) | 80.0% (40/50; 95% CI 66.3–90.0) |
| NB (assumed) | 50 | 92.0% (46/50; 95% CI 80.8–97.8) | 90.0% (45/50; 95% CI 78.2–96.7) | 90.0% (45/50; 95% CI 78.2–96.7) |
| NL (assumed) | 50 | 90.0% (45/50; 95% CI 78.2–96.7) | 88.0% (44/50; 95% CI 75.7–95.5) | 88.0% (44/50; 95% CI 75.7–95.5) |
| NS (observed) | 50 | 90.0% (45/50; 95% CI 78.2–96.7) | 88.0% (44/50; 95% CI 75.7–95.5) | 88.0% (44/50; 95% CI 75.7–95.5) |
| NT (assumed) | 50 | 88.0% (44/50; 95% CI 75.7–95.5) | 88.0% (44/50; 95% CI 75.7–95.5) | 88.0% (44/50; 95% CI 75.7–95.5) |
| NU (assumed) | 50 | 84.0% (42/50; 95% CI 70.9–92.8) | 82.0% (41/50; 95% CI 68.6–91.4) | 82.0% (41/50; 95% CI 68.6–91.4) |
| ON (observed) | 50 | 86.0% (43/50; 95% CI 73.3–94.2) | 86.0% (43/50; 95% CI 73.3–94.2) | 86.0% (43/50; 95% CI 73.3–94.2) |
| PE (assumed) | 50 | 86.0% (43/50; 95% CI 73.3–94.2) | 86.0% (43/50; 95% CI 73.3–94.2) | 86.0% (43/50; 95% CI 73.3–94.2) |
| QC (assumed) | 50 | 86.0% (43/50; 95% CI 73.3–94.2) | 82.0% (41/50; 95% CI 68.6–91.4) | 82.0% (41/50; 95% CI 68.6–91.4) |
| SK (observed) | 50 | 82.0% (41/50; 95% CI 68.6–91.4) | 80.0% (40/50; 95% CI 66.3–90.0) | 80.0% (40/50; 95% CI 66.3–90.0) |
| YT (assumed) | 50 | 88.0% (44/50; 95% CI 75.7–95.5) | 86.0% (43/50; 95% CI 73.3–94.2) | 86.0% (43/50; 95% CI 73.3–94.2) |
| **All** | 650 | 86.8% (564/650; 95% CI 83.9–89.3) | 85.2% (554/650; 95% CI 82.3–87.9) | 85.2% (554/650; 95% CI 82.3–87.9) |

## By photo condition

| Group | Samples | Barcode decoded | Every field exact | Honest claim passes Stage 1 |
|---|---|---|---|---|
| blur | 91 | 60.4% (55/91; 95% CI 49.6–70.5) | 60.4% (55/91; 95% CI 49.6–70.5) | 60.4% (55/91; 95% CI 49.6–70.5) |
| clean | 130 | 100.0% (130/130; 95% CI 97.2–100.0) | 100.0% (130/130; 95% CI 97.2–100.0) | 100.0% (130/130; 95% CI 97.2–100.0) |
| dark | 65 | 100.0% (65/65; 95% CI 94.5–100.0) | 100.0% (65/65; 95% CI 94.5–100.0) | 100.0% (65/65; 95% CI 94.5–100.0) |
| glare | 65 | 100.0% (65/65; 95% CI 94.5–100.0) | 100.0% (65/65; 95% CI 94.5–100.0) | 100.0% (65/65; 95% CI 94.5–100.0) |
| jpeg_low | 65 | 100.0% (65/65; 95% CI 94.5–100.0) | 100.0% (65/65; 95% CI 94.5–100.0) | 100.0% (65/65; 95% CI 94.5–100.0) |
| perspective | 91 | 100.0% (91/91; 95% CI 96.0–100.0) | 100.0% (91/91; 95% CI 96.0–100.0) | 100.0% (91/91; 95% CI 96.0–100.0) |
| rotated | 78 | 37.2% (29/78; 95% CI 26.5–48.9) | 24.4% (19/78; 95% CI 15.3–35.4) | 24.4% (19/78; 95% CI 15.3–35.4) |
| small | 65 | 98.5% (64/65; 95% CI 91.7–100.0) | 98.5% (64/65; 95% CI 91.7–100.0) | 98.5% (64/65; 95% CI 91.7–100.0) |

## Safety and speed

- Wrong name rejected (same card, another person's name from the same region): 100.0% (554/554; 95% CI 99.3–100.0). Highest impostor name score: 77.3 (pass mark 90).
- Lowest honest name score: 100.0.
- Province from the IIN correct: 100.0% (554/554; 95% CI 99.3–100.0).
- Decode time per image: median 45 ms, p95 1505 ms, max 2598 ms (budget 10 s; this machine).

## Failures

| Sample | Condition | What failed |
|---|---|---|
| AB-011 | blur | barcode not decoded |
| AB-012 | blur | barcode not decoded |
| AB-024 | rotated | barcode not decoded |
| AB-025 | rotated | barcode not decoded |
| AB-026 | rotated | barcode not decoded |
| AB-027 | rotated | barcode not decoded |
| AB-028 | rotated | barcode not decoded |
| AB-029 | rotated | barcode not decoded |
| BC-013 | blur | barcode not decoded |
| BC-014 | blur | barcode not decoded |
| BC-024 | rotated | barcode not decoded |
| BC-025 | rotated | barcode not decoded |
| BC-027 | rotated | barcode not decoded |
| BC-028 | rotated | barcode not decoded |
| MB-010 | blur | barcode not decoded |
| MB-011 | blur | barcode not decoded |
| MB-013 | blur | barcode not decoded |
| MB-015 | blur | barcode not decoded |
| MB-024 | rotated | not parsed |
| MB-025 | rotated | not parsed |
| MB-026 | rotated | barcode not decoded |
| MB-027 | rotated | barcode not decoded |
| MB-028 | rotated | barcode not decoded |
| MB-029 | rotated | barcode not decoded |
| NB-012 | blur | barcode not decoded |
| NB-024 | rotated | barcode not decoded |
| NB-026 | rotated | not parsed |
| NB-027 | rotated | barcode not decoded |
| NB-028 | rotated | barcode not decoded |
| NL-010 | blur | barcode not decoded |
| NL-024 | rotated | barcode not decoded |
| NL-025 | rotated | barcode not decoded |
| NL-027 | rotated | not parsed |
| NL-028 | rotated | barcode not decoded |
| NL-029 | rotated | barcode not decoded |
| NS-013 | blur | barcode not decoded |
| NS-024 | rotated | barcode not decoded |
| NS-025 | rotated | not parsed |
| NS-026 | rotated | barcode not decoded |
| NS-027 | rotated | barcode not decoded |
| NS-029 | rotated | barcode not decoded |
| NT-010 | blur | barcode not decoded |
| NT-012 | blur | barcode not decoded |
| NT-014 | blur | barcode not decoded |
| NT-024 | rotated | barcode not decoded |
| NT-027 | rotated | barcode not decoded |
| NT-029 | rotated | barcode not decoded |
| NU-012 | blur | barcode not decoded |
| NU-013 | blur | barcode not decoded |
| NU-015 | blur | barcode not decoded |
| NU-016 | blur | barcode not decoded |
| NU-024 | rotated | barcode not decoded |
| NU-025 | rotated | not parsed |
| NU-026 | rotated | barcode not decoded |
| NU-027 | rotated | barcode not decoded |
| NU-028 | rotated | barcode not decoded |
| ON-010 | blur | barcode not decoded |
| ON-015 | blur | barcode not decoded |
| ON-016 | blur | barcode not decoded |
| ON-025 | rotated | barcode not decoded |
| ON-026 | rotated | barcode not decoded |
| ON-028 | rotated | barcode not decoded |
| ON-037 | small | barcode not decoded |
| PE-010 | blur | barcode not decoded |
| PE-014 | blur | barcode not decoded |
| PE-025 | rotated | barcode not decoded |
| PE-026 | rotated | barcode not decoded |
| PE-027 | rotated | barcode not decoded |
| PE-028 | rotated | barcode not decoded |
| PE-029 | rotated | barcode not decoded |
| QC-010 | blur | barcode not decoded |
| QC-012 | blur | barcode not decoded |
| QC-014 | blur | barcode not decoded |
| QC-015 | blur | barcode not decoded |
| QC-025 | rotated | barcode not decoded |
| QC-026 | rotated | not parsed |
| QC-027 | rotated | not parsed |
| QC-028 | rotated | barcode not decoded |
| QC-029 | rotated | barcode not decoded |
| SK-012 | blur | barcode not decoded |
| SK-013 | blur | barcode not decoded |
| SK-014 | blur | barcode not decoded |
| SK-015 | blur | barcode not decoded |
| SK-016 | blur | barcode not decoded |
| SK-024 | rotated | barcode not decoded |
| SK-025 | rotated | barcode not decoded |
| SK-026 | rotated | not parsed |
| SK-028 | rotated | barcode not decoded |
| SK-029 | rotated | barcode not decoded |
| YT-010 | blur | barcode not decoded |
| YT-011 | blur | barcode not decoded |
| YT-012 | blur | barcode not decoded |
| YT-014 | blur | barcode not decoded |
| YT-024 | rotated | not parsed |
| YT-025 | rotated | barcode not decoded |
| YT-027 | rotated | barcode not decoded |
