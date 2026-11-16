"""Benchmark the face and plate models on public datasets, with the production code paths and thresholds.

    "$PY" scripts/bench_public.py face  --data "$LOCALAPPDATA/komute/bench/agedb/agedb_30.bin" --out docs/calibration/face-agedb30.md
    "$PY" scripts/bench_public.py plate --data "$LOCALAPPDATA/komute/bench/openalpr/repo/endtoend/us" --out docs/calibration/plate-openalpr-us.md

face:  AgeDB-30 (insightface .bin: 12,000 aligned 112x112 faces, 6,000 pairs, 3,000 same person, about 30 years apart).
       Reports the standard 10-fold accuracy and the accept/reject rates at the production threshold.
plate: OpenALPR US end-to-end set (one .txt per photo: file x y w h PLATE). Reports detection, exact read and the
       production plate rule, end to end and on the labelled crop.
Datasets stay outside the repository. Reports hold aggregate numbers only.
"""
from __future__ import annotations

import argparse
import os
import pickle
import statistics
import sys
import time
from multiprocessing import Pool
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backend.core  # noqa: E402,F401  (clamps before cv2)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.core.config import get_settings  # noqa: E402
from backend.core.registry import ModelRegistry  # noqa: E402
from evaluate import clopper_pearson  # noqa: E402

# The .bin is a third-party pickle: allow only the numpy pieces it needs, never arbitrary code.
_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
            ("numpy.core.multiarray", "scalar")}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def pct(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = clopper_pearson(k, n)
    return f"{100 * k / n:.2f}% ({k}/{n}; 95% CI {100 * lo:.2f}–{100 * hi:.2f})"


def _registry():
    s = get_settings()
    reg = ModelRegistry(s.models_dir, s.models_dir / "models.lock", None)
    reg.load()
    return s, reg


def _stamp(source: str) -> str:
    return f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `scripts/bench_public.py` on this machine. Data: {source}."


_SESS = None


def _face_worker_init(threads: int) -> None:
    """Same model file and preprocessing as production; only the thread count differs (bench speed, same numbers)."""
    global _SESS
    import onnxruntime as ort

    from backend.core.hardware import ort_session_options

    s = get_settings()
    so = ort_session_options()
    so.intra_op_num_threads = threads
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # many processes: do not burn CPU waiting
    cv2.setNumThreads(1)
    _SESS = ort.InferenceSession(str(s.models_dir / "w600k_r50.onnx"), so, providers=["CPUExecutionProvider"])


def _face_worker_embed(jpeg: bytes) -> tuple[np.ndarray, float]:
    from backend.core.stages.face import embed

    t = time.perf_counter()
    vec = embed(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), _SESS)
    return vec, 1000 * (time.perf_counter() - t)


def bench_face(data: str, out: str, workers: int = 1, threads: int = 2) -> None:
    s, _ = _registry()
    with open(data, "rb") as f:
        bins, same = _SafeUnpickler(f, encoding="bytes").load()
    same = np.asarray(same, bool)
    assert len(bins) == 2 * len(same), "expected two images per pair"
    thr = s.face_match_threshold
    t0 = time.perf_counter()
    emb, wrong = [], 0
    # Production embed() in a pool of processes, each with the production 2-thread ONNX session: all cores busy,
    # identical numbers. Results come back in order, one line per image, so the run can be watched live.
    pool = Pool(workers, initializer=_face_worker_init, initargs=(threads,))
    for k, (vec, ms) in enumerate(pool.imap(_face_worker_embed, (bytes(b) for b in bins), chunksize=4)):
        emb.append(vec)
        done, elapsed = k + 1, time.perf_counter() - t0
        eta = elapsed / done * (len(bins) - done)
        line = (f"[{done:5d}/{len(bins)}] {100 * done / len(bins):5.1f}%  image {k // 2:04d}_{k % 2}.jpg  "
                f"{ms:5.0f} ms  {done / elapsed:5.1f} img/s  ETA {int(eta // 60):3d}m{int(eta % 60):02d}s")
        if k % 2:
            score = float(np.dot(emb[-2], emb[-1]))
            ok = (score >= thr) == bool(same[k // 2])
            wrong += not ok
            line += (f"  | pair {k // 2:04d} {'same' if same[k // 2] else 'diff'} score {score:+.3f} "
                     f"{'OK   ' if ok else 'WRONG'} (wrong so far {wrong})")
        print(line, flush=True)
    pool.close()
    pool.join()
    emb = np.stack(emb)
    secs = time.perf_counter() - t0
    scores = np.sum(emb[0::2] * emb[1::2], axis=1)  # cosine: embeddings are L2-normalised
    gen, imp = scores[same], scores[~same]
    # Per-pair scores next to the data (outside the repo), for per-pair inspection.
    import json

    Path(data).with_name("face_scores.json").write_text(json.dumps(
        [{"pair": i, "same_person": bool(same[i]), "score": round(float(scores[i]), 6)} for i in range(len(same))]),
        encoding="utf-8")

    # Standard protocol: 10 folds; pick the best-accuracy threshold on 9, test on the held-out one.
    folds = np.array_split(np.arange(len(same)), 10)
    grid = np.arange(-1, 1, 0.0025)
    accs, thrs = [], []
    for k in range(10):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(10) if j != k])
        best = max(grid, key=lambda t: np.mean((scores[train] >= t) == same[train]))
        thrs.append(best)
        accs.append(np.mean((scores[test] >= best) == same[test]))

    thr = s.face_match_threshold
    tp, fp = int(np.sum(gen >= thr)), int(np.sum(imp >= thr))
    lines = [
        _stamp(f"AgeDB-30 ({len(same)} pairs: {len(gen)} same person about 30 years apart, {len(imp)} different people; "
               "pre-aligned 112x112 crops)"),
        "",
        "**Model:** `w600k_r50.onnx`, production `embed()` (no flip augmentation, so slightly below published numbers).",
        "**Not measured here:** face detection and alignment (YuNet), because the crops are already aligned; liveness.",
        "",
        "## Standard AgeDB-30 accuracy (10-fold)",
        "",
        f"- Accuracy: **{100 * np.mean(accs):.2f}%** ± {100 * np.std(accs):.2f} (per-fold thresholds {min(thrs):.3f}–{max(thrs):.3f}).",
        "",
        f"## At the production threshold ({thr:.2f} cosine)",
        "",
        f"- Same person accepted (true accept): {pct(tp, len(gen))}.",
        f"- Same person rejected (false reject, ~30-year gap): {pct(len(gen) - tp, len(gen))}.",
        f"- Different people accepted (false accept): {pct(fp, len(imp))}.",
        f"- Overall accuracy at {thr:.2f}: {100 * (tp + len(imp) - fp) / len(same):.2f}%.",
        "",
        "## Score distribution",
        "",
        f"- Same person: median {np.median(gen):.3f}, 5th percentile {np.percentile(gen, 5):.3f}, lowest {gen.min():.3f}.",
        f"- Different people: median {np.median(imp):.3f}, 95th percentile {np.percentile(imp, 95):.3f}, highest {imp.max():.3f}.",
        "",
        "| Threshold | False reject (same person) | False accept (different people) |",
        "|---|---|---|",
    ]
    for t in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        lines.append(f"| {t:.2f} | {pct(int(np.sum(gen < t)), len(gen))} | {pct(int(np.sum(imp >= t)), len(imp))} |")
    lines += [
        "",
        f"- Speed: {len(bins) / secs:.1f} faces/s using {workers} worker processes x {threads} ONNX thread(s) ({secs / 60:.1f} min for {len(bins)} faces). One production verification embeds 2 faces on 2 threads.",
        f"- Limit: {len(imp)} impostor pairs can bound the false-accept rate only to about "
        f"{100 * clopper_pearson(fp, len(imp))[1]:.2f}% (95% upper bound). The 0.01% target needs about 30,000 impostor trials.",
    ]
    _write(out, "Face match benchmark: AgeDB-30", lines)


def bench_plate(data: str, out: str) -> None:
    from backend.core.matching import normalise_plate, plates_match
    from backend.core.stages.vehicle import PlateBox, crop_with_margin, detect_plate, read_plate

    s, reg = _registry()
    det, ocr = reg.session("plate_detector"), reg.session("plate_ocr")
    rows = []
    for txt in sorted(Path(data).glob("*.txt")):
        parts = txt.read_text(encoding="utf-8").split()
        if len(parts) < 6:
            continue
        img = cv2.imread(str(txt.with_name(parts[0])))
        if img is None:
            continue
        x, y, w, h = map(int, parts[1:5])
        truth = parts[5]
        gt = PlateBox(x, y, x + w, y + h, 1.0)
        t0 = time.perf_counter()
        box = detect_plate(img, det, s.plate_detect_score)
        text, conf = read_plate(crop_with_margin(img, box), ocr) if box else ("", 0.0)
        ms = (time.perf_counter() - t0) * 1000
        iou = 0.0
        if box:
            ix = max(0, min(box.x2, gt.x2) - max(box.x1, gt.x1))
            iy = max(0, min(box.y2, gt.y2) - max(box.y1, gt.y1))
            inter = ix * iy
            iou = inter / (box.w * box.h + gt.w * gt.h - inter)
        gt_text, gt_conf = read_plate(crop_with_margin(img, gt), ocr)
        confident = bool(text) and conf >= s.plate_min_confidence
        rows.append({"found": box is not None, "iou": iou, "exact": normalise_plate(text) == normalise_plate(truth),
                     "rule": confident and plates_match(text, truth), "confident": confident,
                     "gt_exact": normalise_plate(gt_text) == normalise_plate(truth), "ms": ms,
                     "rel_width": w / img.shape[1]})
    n = len(rows)
    c = lambda key: sum(bool(r[key]) for r in rows)  # noqa: E731
    located = sum(r["iou"] >= 0.5 for r in rows)
    times = sorted(r["ms"] for r in rows)
    small = [r for r in rows if r["rel_width"] < 0.05]
    lines = [
        _stamp(f"OpenALPR US end-to-end set ({n} real US car photos, one labelled plate each)"),
        "",
        "**Models:** `yolo-v9-t-384-license-plates-end2end.onnx` (detector), `cct_xs_v2_global.onnx` (OCR). Production functions and "
        f"thresholds: detector score {s.plate_detect_score}, OCR minimum confidence {s.plate_min_confidence}.",
        "**Caveat:** US plates, not Canadian. Only the first labelled plate per photo is scored.",
        "",
        "## End to end (what Stage 3 does)",
        "",
        f"- Plate found: {pct(c('found'), n)}; found at the labelled place (IoU ≥ 0.5): {pct(located, n)}.",
        f"- Exact read: {pct(c('exact'), n)}.",
        f"- Passes the production plate rule with enough confidence (would match the claim): {pct(c('rule'), n)}.",
        f"- Read confident enough to use: {pct(c('confident'), n)}.",
        "",
        "## OCR alone (on the labelled plate crop)",
        "",
        f"- Exact read: {pct(c('gt_exact'), n)}.",
        "",
        "## Small plates and speed",
        "",
        f"- Plates under 5% of the photo width: {len(small)}; found {sum(r['found'] for r in small)}, read exactly {sum(r['exact'] for r in small)}.",
        f"- Detect + read time: median {statistics.median(times):.0f} ms, p95 {times[int(0.95 * (n - 1))]:.0f} ms per photo.",
    ]
    _write(out, "Plate benchmark: OpenALPR US", lines)


def _write(out: str, title: str, lines: list[str]) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(f"# {title}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["face", "plate"])
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=2, help="face only: ONNX threads per worker process (2 = production)")
    p.add_argument("--workers", type=int, default=1,
                   help="face only: worker processes; 1 keeps CPU use low (about 17%% here), more runs faster")
    a = p.parse_args()
    if a.mode == "face":
        bench_face(a.data, a.out, a.workers, a.threads)
    else:
        bench_plate(a.data, a.out)


if __name__ == "__main__":
    main()
