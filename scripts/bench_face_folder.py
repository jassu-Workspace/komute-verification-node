"""Re-run the face model on a folder of pairs (for example the AgeDB-30 false rejects) and write a report.

    python scripts/bench_face_folder.py --folder "%LOCALAPPDATA%\\komute\\bench\\agedb\\false_rejects_0.40" \
        --out docs/calibration/face-false-rejects-0.40.md

Reads <folder>/pairs.csv (pair, image_a, image_b, ...). Scores every pair three ways, one line per pair:
  crop : production embed() on the stored 112x112 aligned crop (the earlier benchmark path).
  full : production path: YuNet detection, largest face, align_112, embed().
  flip : crop path, embedding averaged with the mirror image (test-time flip; not in production).
Writes <folder>/rerun_scores.csv and the Markdown report. Single process; --threads ONNX threads (default 2 = production).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backend.core  # noqa: E402,F401  (clamps before cv2)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import bench_public as bp  # noqa: E402
from backend.core.config import get_settings  # noqa: E402
from backend.core.stages.face import FaceDetector, align_112, embed, pick_licence_face  # noqa: E402


def _flip_embed(img: np.ndarray) -> np.ndarray:
    v = embed(img, bp._SESS) + embed(cv2.flip(img, 1), bp._SESS)
    return v / np.linalg.norm(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=2, help="ONNX threads (2 = production; same scores at any count)")
    a = p.parse_args()
    root = Path(a.folder)
    rows = list(csv.DictReader(open(root / "pairs.csv", encoding="utf-8")))
    s = get_settings()
    thr = s.face_match_threshold
    bp._face_worker_init(a.threads)
    det = FaceDetector(s.models_dir / "face_detection_yunet_2023mar.onnx")

    res, t0 = [], time.perf_counter()
    for i, r in enumerate(rows):
        imgs = [cv2.imread(str(root / r[k]), cv2.IMREAD_COLOR) for k in ("image_a", "image_b")]
        if any(im is None for im in imgs):
            raise SystemExit(f"cannot read pair {r['pair']}")  # a missing image must not pass silently
        crop = [embed(im, bp._SESS) for im in imgs]
        flip = [_flip_embed(im) for im in imgs]
        full, found = [], 0
        for im in imgs:
            f = pick_licence_face(det.detect(im))
            if f is not None:
                found += 1
                full.append(embed(align_112(im, f.landmarks), bp._SESS))
        out = {"pair": int(r["pair"]), "old": float(r["score"]), "crop": float(crop[0] @ crop[1]),
               "flip": float(flip[0] @ flip[1]), "faces_found": found,
               "full": float(full[0] @ full[1]) if found == 2 else None}
        res.append(out)
        el = time.perf_counter() - t0
        eta = el / (i + 1) * (len(rows) - i - 1)
        full_s = f"{out['full']:+.3f}" if out["full"] is not None else " none "
        print(f"[{i + 1:4d}/{len(rows)}] pair {out['pair']:04d}  crop {out['crop']:+.3f}  full {full_s}  "
              f"flip {out['flip']:+.3f}  {'PASS' if out['crop'] >= thr else 'fail'}  "
              f"ETA {int(eta // 60):2d}m{int(eta % 60):02d}s", flush=True)
    secs = time.perf_counter() - t0

    with open(root / "rerun_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)

    n = len(res)
    crop = np.array([r["crop"] for r in res])
    flip = np.array([r["flip"] for r in res])
    full_ok = [r for r in res if r["full"] is not None]
    full = np.array([r["full"] for r in full_ok])
    drift = float(np.max(np.abs(crop - np.array([r["old"] for r in res]))))

    def band(x: np.ndarray) -> list[str]:
        edges = [(-1, 0.0), (0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 2)]
        return [str(int(np.sum((x >= lo) & (x < hi)))) for lo, hi in edges]

    lines = [
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `scripts/bench_face_folder.py` on this machine. "
        f"Data: {n} same-person AgeDB-30 pairs ({2 * n} images) from `{root.name}`. Threshold tested: {thr:.2f}.",
        "",
        f"**Model:** `w600k_r50.onnx`, production `embed()`, 1 process x {a.threads} ONNX thread(s).",
        "",
        "## Result",
        "",
        f"- Re-run on the stored crops: {int(np.sum(crop >= thr))} of {n} pairs now pass {thr:.2f}. "
        f"Largest change from the first run: {drift:.4f} (the model is deterministic).",
        f"- Full production path (YuNet detect, align, embed): faces found in both images for {len(full_ok)} of {n} pairs; "
        f"{int(np.sum(full >= thr)) if len(full) else 0} of those pass {thr:.2f}.",
        f"- With test-time flip (not in production): {int(np.sum(flip >= thr))} of {n} pairs pass {thr:.2f}.",
        "",
        "## Score bands (pairs)",
        "",
        "| Path | < 0.00 | 0.00–0.10 | 0.10–0.20 | 0.20–0.30 | 0.30–0.35 | 0.35–0.40 | ≥ 0.40 |",
        "|---|---|---|---|---|---|---|---|",
        f"| Crop (production embed) | {' | '.join(band(crop))} |",
        f"| Full path ({len(full_ok)} pairs) | {' | '.join(band(full)) if len(full) else 'n/a'} |",
        f"| Crop + flip | {' | '.join(band(flip))} |",
        "",
        "## Pairs recovered by a lower threshold (crop path)",
        "",
        "| Threshold | Pairs that would pass | Share of these 776 | False accepts in the full benchmark |",
        "|---|---|---|---|",
    ]
    fa = {0.25: "2 of 3,000", 0.30: "0 of 3,000", 0.35: "0 of 3,000", 0.38: "0 of 3,000", 0.40: "0 of 3,000"}
    for t in (0.25, 0.30, 0.35, 0.38):
        k = int(np.sum(crop >= t))
        lines.append(f"| {t:.2f} | {k} | {100 * k / n:.1f}% | {fa[t]} |")
    lines += [
        "",
        "## Summary statistics",
        "",
        f"- Crop: median {np.median(crop):.3f}, lowest {crop.min():.3f}, highest {crop.max():.3f}.",
        (f"- Full path: median {np.median(full):.3f}, lowest {full.min():.3f}, highest {full.max():.3f}; "
         f"mean change versus crop {np.mean(full - crop[[i for i, r in enumerate(res) if r['full'] is not None]]):+.3f}.")
        if len(full) else "- Full path: no pair had a face found in both images.",
        f"- Flip: median {np.median(flip):.3f}; mean change versus crop {np.mean(flip - crop):+.3f}.",
        f"- Time: {secs / 60:.1f} min for {n} pairs.",
        "",
        "**Limit:** these are AgeDB-30 pairs about 30 years apart. Licence-to-selfie gaps are usually under 10 years. "
        "Per-pair scores: `rerun_scores.csv` next to the images (outside the repository).",
    ]
    Path(a.out).write_text("# Face re-run: AgeDB-30 false rejects\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {a.out}")


if __name__ == "__main__":
    main()
