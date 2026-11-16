"""Benchmark Stage 1 (decode, parse, cross-check) on the fake set from make_fake_licences.py.

    "$PY" scripts/bench_barcode.py --data "$LOCALAPPDATA/komute/bench/barcodes" --out docs/calibration/barcode-synthetic.md

Per sample: decode the PDF417 with the production decoder, parse it, compare every parsed element to the ground
truth, run the Stage 1 gates with the honest claim (must pass) and with another sample's name (must fail).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backend.core  # noqa: E402,F401  (clamps before cv2)
import cv2  # noqa: E402

from backend.core.aamva import parse_aamva  # noqa: E402
from backend.core.contracts import Claimed  # noqa: E402
from backend.core.gates import licence_gates  # noqa: E402
from backend.core.policy import IIN_TO_PROVINCE  # noqa: E402
from backend.core.stages.barcode import decode_pdf417  # noqa: E402
from evaluate import clopper_pearson  # noqa: E402


def pct(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = clopper_pearson(k, n)
    return f"{100 * k / n:.1f}% ({k}/{n}; 95% CI {100 * lo:.1f}–{100 * hi:.1f})"


def to_claimed(c: dict) -> Claimed:
    d = date.fromisoformat
    return Claimed(c["full_name"], d(c["date_of_birth"]), c["licence_number"], c["issuing_province"], "ABC123", "Grey",
                   licence_class=c["licence_class"], issue_date=d(c["issue_date"]), expiry_date=d(c["expiry_date"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget-s", type=float, default=10.0)
    a = p.parse_args()
    gt = json.loads((Path(a.data) / "ground_truth.json").read_text(encoding="utf-8"))
    now = datetime.fromisoformat(gt["today"] + "T17:00:00+00:00").astimezone(timezone.utc)
    samples = gt["samples"]
    by_region = defaultdict(list)
    for s in samples:
        by_region[s["province"]].append(s)

    results = []
    for s in samples:
        img = cv2.imread(s["image"])
        t0 = time.perf_counter()
        text = decode_pdf417(img, budget_s=a.budget_s)
        ms = (time.perf_counter() - t0) * 1000
        rec = parse_aamva(text) if text else None
        r = {"id": s["id"], "province": s["province"], "distortion": s["distortion"], "profile": s["profile"],
             "decoded": text is not None, "parsed": rec is not None, "decode_ms": ms}
        if rec:
            r["iin_ok"] = IIN_TO_PROVINCE.get(rec.iin) == s["province"]
            wrong = [k for k, v in s["fields"].items() if rec.fields.get(k) != v]
            r["fields_ok"], r["wrong_fields"] = not wrong, wrong
            honest = to_claimed(s["claimed"])
            reasons, _, checks = licence_gates(rec, honest, now)
            r["honest_pass"], r["honest_codes"] = not reasons, [x.code.value for x in reasons]
            r["name_score"] = next(c["score"] for c in checks if c["field"] == "full_name")
            # Impostor: the same card claimed under another person's name from the same region.
            other = next(o for o in by_region[s["province"]] if o["claimed"]["full_name"].upper() != honest.full_name.upper())
            imp_reasons, _, imp_checks = licence_gates(rec, replace(honest, full_name=other["claimed"]["full_name"]), now)
            r["impostor_rejected"] = any(x.code.value == "NAME_MISMATCH" for x in imp_reasons)
            r["impostor_name_score"] = next(c["score"] for c in imp_checks if c["field"] == "full_name")
        results.append(r)
        print(f"{s['id']:8} {s['distortion']:11} decoded={r['decoded']!s:5} pass={r.get('honest_pass')!s:5} {ms:7.0f} ms", flush=True)

    def row(name: str, rs: list[dict]) -> str:
        n = len(rs)
        dec = sum(x["decoded"] for x in rs)
        ok_fields = sum(x.get("fields_ok", False) for x in rs)
        honest = sum(x.get("honest_pass", False) for x in rs)
        return f"| {name} | {n} | {pct(dec, n)} | {pct(ok_fields, n)} | {pct(honest, n)} |"

    head = ["| Group | Samples | Barcode decoded | Every field exact | Honest claim passes Stage 1 |", "|---|---|---|---|---|"]
    lines = [
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `scripts/bench_barcode.py`. Data: {len(samples)} fake licence backs "
        f"from `scripts/make_fake_licences.py` (seed {gt['seed']}, verification date {gt['today']}). No real personal data.",
        "",
        "**What this measures:** the production decoder, parser and Stage 1 gates on synthetic barcodes rendered and degraded to mimic phone photos. "
        "**What it does not measure:** real cards. The barcodes come from our own generator, so a pass here proves the pipeline handles the AAMVA "
        "layout and these degradations, not that every issuer's real card decodes. 10 of 13 layouts are assumed (see the profile column).",
        "",
        "## By region", "", *head,
    ]
    for prov in sorted(by_region):
        rs = [x for x in results if x["province"] == prov]
        lines.append(row(f"{prov} ({rs[0]['profile']})", rs))
    lines.append(row("**All**", results))
    lines += ["", "## By photo condition", "", *head]
    for kind in sorted({x["distortion"] for x in results}):
        lines.append(row(kind, [x for x in results if x["distortion"] == kind]))

    parsed = [x for x in results if x["parsed"]]
    imp_ok = sum(x["impostor_rejected"] for x in parsed)
    times = sorted(x["decode_ms"] for x in results)
    q = lambda f: times[min(len(times) - 1, int(f * len(times)))]  # noqa: E731
    lines += [
        "", "## Safety and speed", "",
        f"- Wrong name rejected (same card, another person's name from the same region): {pct(imp_ok, len(parsed))}. "
        f"Highest impostor name score: {max((x['impostor_name_score'] for x in parsed), default=0):.1f} (pass mark 90).",
        f"- Lowest honest name score: {min((x['name_score'] for x in parsed), default=0):.1f}.",
        f"- Province from the IIN correct: {pct(sum(x['iin_ok'] for x in parsed), len(parsed))}.",
        f"- Decode time per image: median {statistics.median(times):.0f} ms, p95 {q(0.95):.0f} ms, max {times[-1]:.0f} ms "
        f"(budget {a.budget_s:.0f} s; this machine).",
    ]
    fails = [x for x in results if not x.get("honest_pass")]
    if fails:
        lines += ["", "## Failures", "", "| Sample | Condition | What failed |", "|---|---|---|"]
        for x in fails:
            why = ("barcode not decoded" if not x["decoded"] else "not parsed" if not x["parsed"]
                   else ", ".join(x["honest_codes"]) + (f"; wrong fields {x['wrong_fields']}" if x["wrong_fields"] else ""))
            lines.append(f"| {x['id']} | {x['distortion']} | {why} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("# Stage 1 benchmark on fake licences\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    (Path(a.data) / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
