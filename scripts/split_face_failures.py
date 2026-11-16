"""Move the same-person pairs the face model rejects into their own folder, in the same layout as agedb/.

    python scripts/split_face_failures.py --agedb "%LOCALAPPDATA%\\komute\\bench\\agedb" --threshold 0.40

Reads <agedb>/face_scores.json (from bench_public.py face). Moves images/NNNN_0.jpg and NNNN_1.jpg of every
same-person pair scoring below the threshold into <agedb>/false_rejects_<threshold>/images/, and writes a
pairs.csv there (pair, image_a, image_b, same_person, fold, score). The full set can be rebuilt from agedb_30.bin.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agedb", required=True)
    p.add_argument("--threshold", type=float, default=0.40)
    a = p.parse_args()
    root = Path(a.agedb)
    rows = json.loads((root / "face_scores.json").read_text(encoding="utf-8"))
    rejected = sorted((r for r in rows if r["same_person"] and r["score"] < a.threshold), key=lambda r: r["pair"])
    out = root / f"false_rejects_{a.threshold:.2f}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    moved = missing = 0
    for r in rejected:
        for j in (0, 1):
            name = f"{r['pair']:04d}_{j}.jpg"
            src, dst = root / "images" / name, out / "images" / name
            if src.exists():
                shutil.move(str(src), str(dst))
                moved += 1
            elif not dst.exists():
                missing += 1
    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "image_a", "image_b", "same_person", "fold", "score"])
        for r in rejected:
            w.writerow([r["pair"], f"images/{r['pair']:04d}_0.jpg", f"images/{r['pair']:04d}_1.jpg", 1,
                        r["pair"] // 600, r["score"]])
    present = len(list((out / "images").glob("*.jpg")))
    print(f"{len(rejected)} rejected same-person pairs below {a.threshold:.2f}; moved {moved} images now, "
          f"{present} in {out / 'images'}; missing {missing}")
    assert present == 2 * len(rejected), "image count does not match the pair list"


if __name__ == "__main__":
    main()
