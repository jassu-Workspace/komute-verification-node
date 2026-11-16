"""End-to-end on the owner's real licences, through the full stack (API, engine, models, storage, audit).

Privacy: claimed data is derived inside the process from each card's own barcode. Nothing read from the
samples is printed, asserted against a literal, or written to the repo. The summary written to
$KOMUTE_SAMPLES_DIR/e2e_report.md holds counts, codes and similarity scores only.

Limits (see docs): the selfie is the licence-front photo of the same person (a smoke test, NOT a biometric
accuracy measure); the other persons' fronts serve as real impostors; the vehicle is a synthetic scene.
"""
import base64
import json
import os
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.core  # noqa: F401  (clamps before cv2)
import cv2

from backend.app.main import create_app
from backend.core import policy
from backend.core.aamva import parse_aamva
from backend.core.audit import session_prefix, verify_session
from backend.core.config import Settings
from backend.core.pipeline import build_engine
from backend.core.policy import IIN_TO_PROVINCE
from backend.core.stages.barcode import decode_pdf417

pytestmark = [pytest.mark.samples, pytest.mark.models]
KEY = "e2e-key"
VEHICLE = Path("tests/fixtures/synthetic_vehicle.png").read_bytes()
BLANK = cv2.imencode(".png", cv2.cvtColor(__import__("numpy").full((300, 500), 220, "uint8"), cv2.COLOR_GRAY2BGR))[1].tobytes()
RESULTS: list[dict] = []


def b64(b):
    return base64.b64encode(b).decode()


def _cards():
    root = Path(os.environ["KOMUTE_SAMPLES_DIR"], "raw")
    cards = {}
    for person in sorted(root.glob("p*")):
        files = sorted(person.iterdir())
        decoded = [(f, decode_pdf417(cv2.imread(str(f)))) for f in files]
        back = next(((f, t) for f, t in decoded if t), None)
        front = next((f for f, t in decoded if not t), None)
        if back and front and parse_aamva(back[1]):
            cards[person.name] = {"back": back[0].read_bytes(), "front": front.read_bytes(), "rec": parse_aamva(back[1])}
    return cards


@pytest.fixture(scope="module")
def cards():
    c = _cards()
    assert len(c) >= 3, "expected three readable real licences"
    return c


@pytest.fixture(scope="module")
def env(tmp_path_factory, cards):
    clock = {"now": datetime.now(timezone.utc)}
    s = Settings(
        api_key=KEY, storage_root=tmp_path_factory.mktemp("e2e") / "store", storage_key_b64=b64(b"s" * 32),
        storage_key_id="s1", manifest_key_b64=b64(b"m" * 32), manifest_key_id="m1", dev_skip_liveness=True,
    )
    engine = build_engine(s, clock=lambda: clock["now"])
    with TestClient(create_app(s, engine)) as client:
        yield client, engine, clock


def claim(rec, **over):
    dob = rec.date("DBB")
    f = {
        "request_id": "x", "driver_id": "d", "consent_at": "2026-11-11T12:00:00Z", "consent_policy_version": "v1",
        "claimed.full_name": rec.full_name(), "claimed.date_of_birth": dob.isoformat(),
        "claimed.licence_number": rec.fields["DAQ"], "claimed.issuing_province": IIN_TO_PROVINCE[rec.iin],
        "claimed.plate_number": "ABC123", "claimed.vehicle_colour": "Grey",
        "claimed.licence_class": rec.fields["DCA"], "claimed.issue_date": rec.date("DBD").isoformat(),
        "claimed.expiry_date": rec.date("DBA").isoformat(),
    }
    f.update(over)
    return f


def run(env, name, card, case, fields=None, files=None, now=None, others=None):
    client, engine, clock = env
    clock["now"] = now or datetime.now(timezone.utc)
    f = claim(card["rec"], **(fields or {}))
    f["request_id"], f["driver_id"] = f"{name}-{case}", name
    imgs = {"selfie": card["front"], "licence_front": card["front"], "licence_back": card["back"], "vehicle": VEHICLE}
    imgs.update(files or {})
    r = client.post("/v1/verify", data=f, files={k: (f"{k}.img", v) for k, v in imgs.items()},
                    headers={"X-API-Key": KEY})
    body = r.json()
    codes = [x["code"] for x in body.get("reasons", [])]
    st = body.get("stages", {})
    # Non-personal fields only: codes, flags, scores, timings. Never names, numbers, dates or barcode text.
    RESULTS.append({"case": case, "person": name, "status": r.status_code, "decision": body.get("decision"),
                    "codes": codes, "retryable": [x["retryable"] for x in body.get("reasons", [])],
                    "sim": st.get("face", {}).get("similarity"),
                    "barcode_readable": st.get("barcode", {}).get("readable"),
                    "jurisdiction": st.get("barcode", {}).get("jurisdiction"),
                    "face_ran": st.get("face", {}).get("ran"), "vehicle_ran": st.get("vehicle", {}).get("ran"),
                    "plate_found": st.get("vehicle", {}).get("plate_found"),
                    "plate_matched": st.get("vehicle", {}).get("plate_matched"),
                    "advisories": body.get("advisories", []), "total_ms": body.get("timings_ms", {}).get("total")})
    if r.status_code == 200:
        report = verify_session(engine.store, session_prefix(name, f["request_id"]), b"m" * 32)
        assert report.ok, f"{name}/{case}: session does not verify"
    return r, codes, body


PEOPLE = ["p1", "p2", "p3"]


@pytest.mark.parametrize("p", PEOPLE)
def test_baseline_real_card_with_own_photo_passes_stages_1_and_2(env, cards, p):
    r, codes, body = run(env, p, cards[p], "baseline")
    assert r.status_code == 200
    assert body["stages"]["barcode"]["readable"] and body["stages"]["barcode"]["gates_failed"] == []
    assert body["stages"]["face"]["similarity"] > 0.9, "same photo must match itself"
    assert "FACE_MISMATCH" not in codes and "LIVENESS_FAILED" not in codes
    assert "LIVENESS_SKIPPED_DEV" in body["advisories"]


@pytest.mark.parametrize("p", PEOPLE)
def test_baseline_vehicle_stage_runs_on_synthetic_scene(env, cards, p):
    _, codes, body = run(env, p, cards[p], "vehicle-smoke")
    assert body["stages"]["vehicle"]["ran"] and body["stages"]["vehicle"]["plate_found"]


@pytest.mark.parametrize("p", PEOPLE)
@pytest.mark.parametrize("case,over,code", [
    ("wrong-name", {"claimed.full_name": "Zed Nobody"}, "NAME_MISMATCH"),
    ("wrong-licence", {"claimed.licence_number": "99999999"}, "LICENCE_NUMBER_MISMATCH"),
    ("wrong-province", {"claimed.issuing_province": "NU"}, "JURISDICTION_MISMATCH"),
    ("wrong-class", {"claimed.licence_class": "ZZ9"}, "CLASS_MISMATCH"),
    ("wrong-issue", {"claimed.issue_date": "1901-01-01"}, "ISSUE_DATE_MISMATCH"),
    ("wrong-expiry", {"claimed.expiry_date": "2199-01-01"}, "EXPIRY_DATE_MISMATCH"),
])
def test_wrong_claims_are_rejected_with_final_reasons(env, cards, p, case, over, code):
    r, codes, body = run(env, p, cards[p], case, fields=over)
    assert body["decision"] == "REJECTED" and code in codes
    assert all(not x["retryable"] for x in body["reasons"] if x["code"] == code)
    assert body["stages"]["face"] == {"ran": False}, "stage 1 failure must stop the pipeline"


@pytest.mark.parametrize("p", PEOPLE)
def test_wrong_date_of_birth_is_rejected(env, cards, p):
    dob = cards[p]["rec"].date("DBB") + timedelta(days=1)
    _, codes, body = run(env, p, cards[p], "wrong-dob", fields={"claimed.date_of_birth": dob.isoformat()})
    assert body["decision"] == "REJECTED" and "DOB_MISMATCH" in codes


@pytest.mark.parametrize("p", PEOPLE)
def test_expired_licence_is_rejected(env, cards, p):
    now = datetime.combine(cards[p]["rec"].date("DBA") + timedelta(days=3), datetime.min.time(), timezone.utc) + timedelta(hours=18)
    _, codes, body = run(env, p, cards[p], "expired", now=now)
    assert body["decision"] == "REJECTED" and "LICENCE_EXPIRED" in codes


@pytest.mark.parametrize("p", PEOPLE)
def test_underage_on_the_day_before_18th_birthday_is_rejected(env, cards, p):
    dob = cards[p]["rec"].date("DBB")
    day = date(dob.year + 18, dob.month, 28 if (dob.month, dob.day) == (2, 29) else dob.day) - timedelta(days=1)
    now = datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=18)
    _, codes, body = run(env, p, cards[p], "underage", now=now)
    assert "UNDERAGE" in codes


@pytest.mark.parametrize("p", PEOPLE)
def test_unreadable_barcode_is_retryable_with_the_retake_message(env, cards, p):
    for case, back in (("blank-back", BLANK), ("front-as-back", cards[p]["front"])):
        r, codes, body = run(env, p, cards[p], case, files={"licence_back": back})
        assert body["decision"] == "REJECTED" and codes == ["BARCODE_UNREADABLE"]
        assert body["reasons"][0]["retryable"] and "not clearly visible" in body["reasons"][0]["message"]


@pytest.mark.parametrize("p", PEOPLE)
def test_no_face_in_selfie_is_retryable(env, cards, p):
    _, codes, body = run(env, p, cards[p], "blank-selfie", files={"selfie": BLANK})
    assert "FACE_NOT_FOUND_SELFIE" in codes
    assert next(x for x in body["reasons"] if x["code"] == "FACE_NOT_FOUND_SELFIE")["retryable"]


@pytest.mark.parametrize("p", PEOPLE)
def test_impostor_selfie_from_other_real_licences_is_rejected(env, cards, p):
    for other in PEOPLE:
        if other == p:
            continue
        _, codes, body = run(env, p, cards[p], f"impostor-{other}", files={"selfie": cards[other]["front"]})
        assert body["decision"] == "REJECTED" and "FACE_MISMATCH" in codes, "a different person's face must not match"


@pytest.mark.parametrize("p", PEOPLE)
def test_missing_plate_is_retryable(env, cards, p):
    _, codes, body = run(env, p, cards[p], "no-plate", files={"vehicle": BLANK})
    assert "PLATE_NOT_FOUND" in codes and next(x for x in body["reasons"] if x["code"] == "PLATE_NOT_FOUND")["retryable"]


def test_replay_returns_stored_result_and_changed_payload_conflicts(env, cards):
    client, _, _ = env
    r1, _, b1 = run(env, "p2", cards["p2"], "replay")
    r2, _, b2 = run(env, "p2", cards["p2"], "replay")
    assert b1 == b2 and r2.status_code == 200
    r3, _, _ = run(env, "p2", cards["p2"], "replay", fields={"claimed.plate_number": "ZZZ999"})
    assert r3.status_code == 409
    got = client.get("/v1/sessions/p2-replay", headers={"X-API-Key": KEY})
    assert got.status_code == 200 and got.json() == b1


def test_health_after_loading_real_models(env):
    client, _, _ = env
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and KEY not in r.text


def test_zz_write_private_summary(env):
    decisions = Counter((x["decision"] or f"HTTP-{x['status']}") for x in RESULTS)
    reasons = Counter(c for x in RESULTS for c in x["codes"])
    sims = sorted(x["sim"] for x in RESULTS if x["case"].startswith("impostor") and x["sim"] is not None)
    selfs = [x["sim"] for x in RESULTS if x["case"] == "baseline" and x["sim"] is not None]
    lines = ["# Real-licence end-to-end summary (counts, codes, scores only)", f"- Requests: {len(RESULTS)}",
             f"- Decisions: {dict(decisions)}", f"- Reason codes: {dict(reasons)}",
             f"- Impostor similarities (different real people, {len(sims)} pairs): "
             + (f"min {sims[0]:.3f}, max {sims[-1]:.3f}, mean {sum(sims) / len(sims):.3f}" if sims else "none"),
             f"- Same-photo similarity: {[round(s, 3) for s in selfs]}"]
    Path(os.environ["KOMUTE_SAMPLES_DIR"], "e2e_report.md").write_text("\n".join(lines), encoding="utf-8")
    Path(os.environ["KOMUTE_SAMPLES_DIR"], "e2e_results.json").write_text(json.dumps(RESULTS), encoding="utf-8")
    assert sims and max(sims) < 0.4, "impostor similarity must stay below the provisional 0.40 threshold"
