"""Generate fake licence-back images with AAMVA PDF417 barcodes for the 13 jurisdictions, plus ground truth.

    "$PY" scripts/make_fake_licences.py --per-region 50 --out "$LOCALAPPDATA/komute/bench/barcodes"

Every value is invented. The barcode layout copies what the real cards show, without any of their data:
  NS: version 09, DL subfile only.  ON: version 09, DL + ZO subfiles.  SK: version 03, DL only, subfile offset 0
  (the issuer quirk the parser handles). The other 10 jurisdictions are not observed: they use the AAMVA
  version 09 DL layout, which is an assumption until real cards confirm it.

Writes <out>/<REGION>/<id>.jpg, <out>/ground_truth.json and <out>/ground_truth.csv.
Keep the output outside the repository (it is large); the script, not the images, is the source of truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.core  # noqa: E402,F401  (clamps before cv2)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from backend.core.policy import IIN_TO_PROVINCE, LICENCE_PATTERNS  # noqa: E402
from tests.fixtures.aamva_factory import render_card  # noqa: E402

IIN = {v: k for k, v in IIN_TO_PROVINCE.items()}
FIRST = ["ALEX", "JORDAN", "TAYLOR", "MORGAN", "CASEY", "RILEY", "JAMIE", "AVERY", "QUINN", "ROWAN",
         "SAM", "DREW", "PARKER", "REESE", "SKYLER", "LOGAN", "EMERSON", "HAYDEN", "FINLEY", "SAGE"]
MIDDLE = ["LEE", "MARIE", "JAMES", "ANNE", "RAY", "JEAN", "", "", ""]
LAST = ["TESTER", "SAMPLE", "EXAMPLE", "FAKEMAN", "NOTREAL", "DUMMY", "PLACEHOLDER", "MOCKWELL",
        "SYNTHE", "FICTION", "TREMBLAY-COTE", "O'TESTER", "MACFAKE", "LEBLANC-FAUX", "ROY", "NGUYEN-TEST"]
CLASSES = {"ON": ["G", "G2", "G1", "D", "M2"], "BC": ["5", "7"], "AB": ["5", "7"], "QC": ["5", "4A"],
           "MB": ["5"], "SK": ["5"], "NB": ["5"], "NS": ["5"], "NL": ["5"], "PE": ["5"], "YT": ["5"],
           "NT": ["5"], "NU": ["5"]}
# Same 50-sample mix in every region: (distortion, count). Phone-photo problems the stage must survive.
MIX = [("clean", 10), ("blur", 7), ("perspective", 7), ("rotated", 6), ("dark", 5), ("small", 5),
       ("jpeg_low", 5), ("glare", 5)]


def _digits(r: random.Random, n: int) -> str:
    return "".join(r.choice("0123456789") for _ in range(n))


def _letters(r: random.Random, n: int, extra: str = "") -> str:
    return "".join(r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ" + extra) for _ in range(n))


def licence_number(prov: str, last: str, r: random.Random) -> str:
    initial = last[0]
    return {
        "ON": lambda: initial + _digits(r, 14), "QC": lambda: initial + _digits(r, 12), "BC": lambda: _digits(r, 7),
        "AB": lambda: _digits(r, 9), "SK": lambda: _digits(r, 8), "NB": lambda: _digits(r, 7),
        "NL": lambda: initial + _digits(r, 9), "PE": lambda: _digits(r, 6), "YT": lambda: _digits(r, 6),
        "NT": lambda: _digits(r, 9), "NU": lambda: _digits(r, 5),
        "NS": lambda: initial + _letters(r, 4, "*") + _digits(r, 9),
        "MB": lambda: initial + _letters(r, 4, "*") + _letters(r, 1) + _letters(r, 1, "*") + _letters(r, 5, "0123456789"),
    }[prov]()


def _d(x: date) -> str:
    return x.strftime("%Y%m%d")  # CAN order CCYYMMDD


def make_record(prov: str, i: int, r: random.Random, today: date) -> dict:
    first, middle, last = r.choice(FIRST), r.choice(MIDDLE), r.choice(LAST)
    dob = date(1950, 1, 1) + timedelta(days=r.randrange((date(2007, 12, 31) - date(1950, 1, 1)).days))
    issue = today - timedelta(days=r.randrange(30, 5 * 365))
    expiry = issue + timedelta(days=5 * 365 + r.randrange(0, 365))
    dl = {
        "DCA": r.choice(CLASSES[prov]), "DCB": "NONE", "DCD": "NONE", "DBA": _d(expiry),
        "DCS": last, "DAC": first, "DAD": middle or "NONE", "DBD": _d(issue), "DBB": _d(dob),
        "DBC": r.choice("12"), "DAY": r.choice(["BRO", "BLU", "GRN", "HAZ"]), "DAU": f"{r.randrange(150, 200)} cm",
        "DAG": f"{r.randrange(1, 9999)} FAKE ST", "DAI": "TESTVILLE", "DAJ": "NF" if prov == "NL" and i % 2 else prov,
        "DAK": f"{_letters(r, 1)}{_digits(r, 1)}{_letters(r, 1)}{_digits(r, 1)}{_letters(r, 1)}{_digits(r, 1)}",
        "DAQ": licence_number(prov, last, r), "DCF": "FAKE" + _digits(r, 8), "DCG": "CAN",
        "DDE": "N", "DDF": "N", "DDG": "N",
    }
    if prov in ("ON", "SK"):
        dl["DCK"] = "FAKE" + _digits(r, 10)
    if prov == "NS":
        dl["DAH"] = "UNIT " + _digits(r, 2)
    sub = {"ZO": {"ZOA": f"{last},{first}", "ZOB": "N", "ZOC": "N", "ZOD": "N", "ZOE": "N", "ZOZ": "FAKE"}} if prov == "ON" else {}
    version = "03" if prov == "SK" else "09"
    return {"dl": dl, "sub": sub, "version": version, "profile": "observed" if prov in ("NS", "ON", "SK") else "assumed"}


def aamva_text(iin: str, version: str, dl: dict, sub: dict, zero_offsets: bool) -> str:
    subfiles = [("DL", dl)] + list(sub.items())
    bodies = [t + "".join(f"{k}{v}\n" for k, v in f.items()) + "\r" for t, f in subfiles]
    juris = "00" if int(version) >= 2 else ""
    header_len = len("@\n\x1e\rANSI ") + 6 + 2 + len(juris) + 2 + 10 * len(subfiles)
    directory, offset = "", header_len
    for (t, _), body in zip(subfiles, bodies):
        directory += f"{t}{0 if zero_offsets else offset:04d}{len(body):04d}"
        offset += len(body)
    return f"@\n\x1e\rANSI {iin}{version}{juris}{len(subfiles):02d}{directory}" + "".join(bodies)


def degrade(img: np.ndarray, kind: str, r: random.Random) -> tuple[np.ndarray, int]:
    """Return the degraded image and the JPEG quality to save it with."""
    h, w = img.shape[:2]
    if kind == "blur":
        k = r.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (k, k), 0.4 * k), 90
    if kind == "perspective":
        j = lambda: r.uniform(0.0, 0.07)  # noqa: E731
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * j(), h * j()], [w * (1 - j()), h * j()], [w * (1 - j()), h * (1 - j())], [w * j(), h * (1 - j())]])
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(90, 80, 70)), 90
    if kind == "rotated":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-15, 15), 1.0)
        return cv2.warpAffine(img, m, (w, h), borderValue=(90, 80, 70)), 90
    if kind == "dark":
        f = r.uniform(0.25, 0.45)
        noise = np.random.default_rng(r.randrange(1 << 30)).normal(0, 6, img.shape)
        return np.clip(img.astype(np.float32) * f + noise, 0, 255).astype(np.uint8), 85
    if kind == "small":
        f = r.uniform(0.45, 0.7)
        return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), 90
    if kind == "jpeg_low":
        return img, r.randrange(25, 45)
    if kind == "glare":
        cx, cy, rad = r.randrange(w), r.randrange(h), r.randrange(min(h, w) // 5, min(h, w) // 2)
        yy, xx = np.ogrid[:h, :w]
        mask = np.clip(1 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad, 0, 1)[..., None] * r.uniform(0.5, 0.85)
        return np.clip(img * (1 - mask) + 255 * mask, 0, 255).astype(np.uint8), 90
    return img, 92


def claim(dl: dict, prov: str) -> dict:
    """What an honest driver types: names as on the card, ISO dates, the claimed province."""
    iso = lambda v: f"{v[:4]}-{v[4:6]}-{v[6:]}"  # noqa: E731
    middle = "" if dl["DAD"] == "NONE" else dl["DAD"]
    return {"full_name": " ".join(x for x in (dl["DAC"].title(), middle.title(), dl["DCS"].title()) if x),
            "date_of_birth": iso(dl["DBB"]), "licence_number": dl["DAQ"], "issuing_province": prov,
            "licence_class": dl["DCA"], "issue_date": iso(dl["DBD"]), "expiry_date": iso(dl["DBA"])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-region", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--today", default="2026-11-12")
    a = p.parse_args()
    out, today = Path(a.out), date.fromisoformat(a.today)
    kinds = [k for k, n in MIX for _ in range(n)]
    kinds = (kinds * (a.per_region // len(kinds) + 1))[: a.per_region]
    rows = []
    for prov in sorted(IIN):
        (out / prov).mkdir(parents=True, exist_ok=True)
        for i in range(a.per_region):
            r = random.Random(f"{a.seed}-{prov}-{i}")
            rec = make_record(prov, i, r, today)
            dl = rec["dl"]
            assert LICENCE_PATTERNS[prov].match(dl["DAQ"]), (prov, dl["DAQ"])
            text = aamva_text(IIN[prov], rec["version"], dl, rec["sub"], zero_offsets=prov == "SK")
            card = render_card(text, scale=r.choice([2, 3, 3, 4]), canvas=(1000, 1600), seed=r.randrange(1 << 30))
            img, quality = degrade(card, kinds[i], r)
            sid = f"{prov}-{i:03d}"
            path = out / prov / f"{sid}.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            rows.append({"id": sid, "image": str(path), "province": prov, "iin": IIN[prov], "profile": rec["profile"],
                         "aamva_version": rec["version"], "distortion": kinds[i], "jpeg_quality": quality,
                         "fields": {**dl, **{k: v for s in rec["sub"].values() for k, v in s.items()}},
                         "claimed": claim(dl, prov)})
    (out / "ground_truth.json").write_text(json.dumps({"today": a.today, "seed": a.seed, "samples": rows}, indent=1), encoding="utf-8")
    with open(out / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "province", "iin", "profile", "distortion", "full_name", "date_of_birth",
                    "licence_number", "licence_class", "issue_date", "expiry_date"])
        for s in rows:
            c = s["claimed"]
            w.writerow([s["id"], s["image"], s["province"], s["iin"], s["profile"], s["distortion"], c["full_name"],
                        c["date_of_birth"], c["licence_number"], c["licence_class"], c["issue_date"], c["expiry_date"]])
    print(f"wrote {len(rows)} fake samples to {out}")


if __name__ == "__main__":
    main()
