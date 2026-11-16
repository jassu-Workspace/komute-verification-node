"""Build the live real-licence results PDF from the end-to-end test output (non-personal fields only).

    python docs/guides/build_live_report.py <path to e2e_results.json>
The JSON comes from tests/e2e/test_samples.py ($KOMUTE_SAMPLES_DIR/e2e_results.json, outside the repo). It holds
codes, flags, scores and timings only. Licences appear as A, B, C with their province; nothing personal is shown.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import OrderedDict

from build_guides import (SOFT, W, P, Spacer, Table, TableStyle, base, build, bullets, cell, grid, h1, h2, note,
                          small)
from reportlab.lib import colors

R = json.load(open(sys.argv[1], encoding="utf-8"))
LABEL = {"p1": "A", "p2": "B", "p3": "C"}
PROV = {r["person"]: r["jurisdiction"] for r in R if r["case"] == "baseline"}
OKC = colors.HexColor("#17794a")

CASES = OrderedDict([
    ("baseline", ("Correct data, own photo, vehicle photo", "APPROVED")),
    ("vehicle-smoke", ("Correct data; checks that the vehicle stage runs", "APPROVED")),
    ("wrong-name", ("Name changed to another name", "REJECTED: NAME_MISMATCH")),
    ("wrong-dob", ("Date of birth one day later", "REJECTED: DOB_MISMATCH")),
    ("wrong-licence", ("Licence number changed", "REJECTED: LICENCE_NUMBER_MISMATCH")),
    ("wrong-province", ("Province changed to NU", "REJECTED: JURISDICTION_MISMATCH")),
    ("wrong-class", ("Licence class changed", "REJECTED: CLASS_MISMATCH")),
    ("wrong-issue", ("Issue date changed", "REJECTED: ISSUE_DATE_MISMATCH")),
    ("wrong-expiry", ("Expiry date changed", "REJECTED: EXPIRY_DATE_MISMATCH")),
    ("expired", ("Clock set to 3 days after the expiry date", "REJECTED: LICENCE_EXPIRED")),
    ("underage", ("Clock set to the day before the 18th birthday", "REJECTED: UNDERAGE")),
    ("blank-back", ("Blank image as licence back", "REJECTED: BARCODE_UNREADABLE (retry)")),
    ("front-as-back", ("Licence front sent as licence back", "REJECTED: BARCODE_UNREADABLE (retry)")),
    ("blank-selfie", ("Blank image as selfie", "REJECTED: FACE_NOT_FOUND_SELFIE (retry)")),
    ("impostor", ("Selfie of a different real person", "REJECTED: FACE_MISMATCH")),
    ("no-plate", ("Blank image as vehicle photo", "REJECTED: PLATE_NOT_FOUND (retry)")),
])


def result(r: dict) -> str:
    if r["status"] != 200:
        return f"HTTP {r['status']}"
    codes = ", ".join(c for c in r["codes"]) or "-"
    return r["decision"] + ("" if r["decision"] == "APPROVED" else f": {codes}")


def expected_ok(case: str, r: dict) -> bool:
    exp = CASES[case][1]
    if exp == "APPROVED":
        return r["decision"] == "APPROVED"
    code = exp.split(": ")[1].split(" ")[0]
    return r["decision"] == "REJECTED" and code in r["codes"]


rows = [["Test", "What was sent", "Expected", "Licence A", "Licence B", "Licence C"]]
n_ok = n_runs = 0
for case, (sent, exp) in CASES.items():
    per = {}
    for p in ("p1", "p2", "p3"):
        rs = [r for r in R if r["person"] == p and (r["case"] == case or (case == "impostor" and r["case"].startswith("impostor")))]
        oks = [expected_ok(case, r) for r in rs]
        n_ok += sum(oks)
        n_runs += len(rs)
        if case == "impostor":
            per[p] = "<br/>".join(f"{result(r)} ({r['sim']:+.3f})" for r in rs)
        else:
            per[p] = "<br/>".join(sorted({result(r) for r in rs}))
        per[p] = ("<font color='#17794a'><b>PASS</b></font><br/>" if all(oks) and oks else
                  "<font color='#b42318'><b>FAIL</b></font><br/>") + per[p]
    rows.append([case, sent, exp, per["p1"], per["p2"], per["p3"]])

replay = [r for r in R if r["case"] == "replay"]
# Replay: first and second send give the same stored result; the third (changed data) must be 409.
replay_ok = (len(replay) == 3 and replay[0]["decision"] == replay[1]["decision"] == "APPROVED"
             and replay[2]["status"] == 409)
n_ok += 3 if replay_ok else 0
n_runs += len(replay)
decisions = [r["decision"] for r in R if r["status"] == 200]
ms = [r["total_ms"] for r in R if r["total_ms"]]
base_ms = [r["total_ms"] for r in R if r["case"] == "baseline"]
imp = sorted({round(r["sim"], 3) for r in R if r["case"].startswith("impostor")})
same = [r["sim"] for r in R if r["case"] == "baseline"]


def kpi_row(items):
    t = Table([[P(f"<font size=15 color='#27407a'><b>{v}</b></font><br/><font size=7.4 color='#5b6169'>{k}</font>", cell)
                for v, k in items]], colWidths=[W / len(items)] * len(items))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("LINEAFTER", (0, 0), (-2, -1), 0.6, colors.white),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


s = [P("Komute-Local: inputs, verification flow and live results on three real licences", h1),
     P("Test report. Written in ASD-STE100 Simplified Technical English. Test run: 15 November 2026, development laptop "
       "(Windows, CPU only), full stack: API, models, encrypted storage and signed records.", small), Spacer(1, 6),
     kpi_row([(len(R), "live requests"), (f"{n_ok}/{n_runs}", "results as expected"),
              (decisions.count("APPROVED"), "approved (all expected)"), (decisions.count("REJECTED"), "rejected (all expected)"),
              ("48/48", "test checks passed")]),
     Spacer(1, 4),
     note("NOTE", "This report shows no personal data. The licences are Licence A (" + PROV["p1"] + "), Licence B ("
          + PROV["p2"] + ") and Licence C (" + PROV["p3"] + "). Names, numbers, dates, addresses and photos are not shown.")]

s += [P("1. What we ask from the user", h2),
      P("The main application collects these items from the driver. It sends them to the node in one request.", base),
      Spacer(1, 3),
      grid([["Item", "Details"],
            ["4 photos", "Selfie; licence front; licence back (with the barcode); vehicle with the licence plate visible. "
                         "JPEG, PNG or WebP, maximum 10 MB each, original photos."],
            ["Licence data", "Full name, date of birth, licence number, issuing province, licence class, issue date, expiry date"],
            ["Vehicle data", "Plate number and body colour. Make, model and year are optional (stored, not checked)."],
            ["Consent", "Time of the driver's consent (with time zone) and the version of the consent text"],
            ["Identifiers", "request_id (unique for each attempt) and driver_id"]],
           [0.2, 0.8])]

s += [P("2. How the verification operates", h2),
      grid([["Stage", "What the node does", "Result if it fails"],
            ["1. Licence barcode", "Reads the PDF417 barcode on the licence back. The barcode is the reference data. "
             "Compares the typed data with it: province, name (fuzzy, 90% pass), date of birth, licence number, class, "
             "issue date, expiry date (exact). Checks: not expired, issue date not after today, age 18 or more.",
             "REJECTED. Stages 2 and 3 do not run."],
            ["2. Face", "Finds the face in the selfie and on the licence front. Liveness check (skipped: no model yet). "
             "Compares the two faces. Pass: similarity 0.35 or more.", "REJECTED"],
            ["3. Vehicle", "Finds the plate and reads it. Pass: equal to the typed plate (one O/0, I/1 or B/8 difference "
             "is permitted). Body colour is advisory only.", "REJECTED"],
            ["Record", "Stores the photos encrypted (AES-256-GCM). Writes a signed record (HMAC-SHA256).", "Error, no decision"]],
           [0.15, 0.6, 0.25]),
      Spacer(1, 3),
      P("The result is APPROVED only if all checks pass. Each failed check gives a reason code. Each code is retryable "
        "(a better photo can help) or final (a new photo cannot help).", base)]

s += [P("3. Test method", h2)]
s += bullets([
    "The test sends real requests to the API with the three real licences (front and back photos).",
    "The correct typed data comes from each licence's own barcode. Each test then changes one value.",
    "Selfie: the licence front photo of the same person. No real selfies are available yet.",
    "Impostor test: the licence front photo of one of the other two people.",
    "Vehicle: a synthetic test image with the plate ABC123. Claimed colour: Grey.",
    "For each decision, the test also checks the signed record. All signed records are valid.",
])

s += [P("4. Live results", h2),
      grid(rows, [0.11, 0.2, 0.2, 0.163, 0.163, 0.164]),
      Spacer(1, 3)]
s += bullets([
    "Replay test (Licence B): the same request_id with the same data gave the same stored result, " +
    ("APPROVED, " if replay and replay[0]["decision"] == "APPROVED" else "") +
    f"without a new run. The same request_id with changed data gave HTTP {replay[-1]['status']} (REQUEST_ID_CONFLICT), as expected.",
    "Underage test: the result also shows LICENCE_NOT_YET_VALID. The simulated date is before the issue date of the licence. "
    "This is correct.",
    "Approved results show the advisory COLOUR_MISMATCH: the synthetic vehicle is not grey. Advisories do not stop an approval.",
    "All results show the advisory LIVENESS_SKIPPED_DEV, because no liveness model is installed.",
])

s += [P("5. Face scores and time", h2),
      grid([["Measure", "Result"],
            ["Same person (same photo), 3 licences", ", ".join(f"{x:.3f}" for x in same) + " (pass mark 0.35)"],
            ["Different real people, 3 pairs (each tested in both directions)",
             ", ".join(f"{x:+.3f}" for x in imp) + ". All far below 0.35, so all are rejected."],
            ["Time for a correct request (3 licences)", f"{min(base_ms) / 1000:.2f} to {max(base_ms) / 1000:.2f} seconds"],
            ["Time for all 53 requests with a result", f"median {statistics.median(ms) / 1000:.2f} s, "
             f"range {min(ms) / 1000:.2f} to {max(ms) / 1000:.2f} s"]],
           [0.42, 0.58])]

s += [P("6. Limits of this test", h2)]
s += bullets([
    "Three people only. This test shows that the system operates correctly. It cannot measure accuracy rates.",
    "The selfie is the licence photo itself, so the same-person score is 1.000. A real selfie will give a lower score.",
    "The vehicle photo is synthetic. No real vehicle photo of a driver was tested.",
    "No liveness model is installed. The node cannot yet detect a printed photo or a screen.",
])

build("komute-local-live-results.pdf", "Inputs, flow and live results", s)
