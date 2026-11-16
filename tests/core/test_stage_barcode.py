import json
from datetime import date, datetime, timezone

import numpy as np

import backend.core  # noqa: F401
from backend.core.contracts import MESSAGES, Claimed, ReasonCode
from backend.core.stages.barcode import run_barcode_stage
from tests.fixtures.aamva_factory import FAKE_FIELDS, make_aamva_text, render_card

NOW = datetime(2026, 11, 11, 15, 0, tzinfo=timezone.utc)
CLAIMED = Claimed(
    full_name="Alex Jamie Tester", date_of_birth=date(1990, 3, 15), licence_number="T12345678901234",
    issuing_province="ON", plate_number="ABC123", vehicle_colour="Grey",
    licence_class="G", issue_date=date(2023, 9, 14), expiry_date=date(2030, 12, 31),
)


def test_unreadable_gives_retryable_barcode_unreadable():
    result, record = run_barcode_stage(np.full((500, 800, 3), 200, np.uint8), CLAIMED, NOW, budget_s=3)
    assert record is None and not result.details["readable"]
    [reason] = result.reasons
    assert reason.code == ReasonCode.BARCODE_UNREADABLE and reason.retryable
    assert reason.message == MESSAGES[ReasonCode.BARCODE_UNREADABLE]
    assert "not clearly visible" in reason.message and "retake" in reason.message.lower()


def test_pdf417_not_aamva_gives_barcode_not_aamva():
    result, record = run_barcode_stage(render_card("hello world, not a licence"), CLAIMED, NOW)
    assert record is None
    assert [r.code for r in result.reasons] == [ReasonCode.BARCODE_NOT_AAMVA] and result.reasons[0].retryable


def test_valid_card_passes():
    result, record = run_barcode_stage(render_card(make_aamva_text()), CLAIMED, NOW)
    assert result.reasons == [] and record is not None
    assert result.details["jurisdiction"] == "ON" and result.details["aamva"]
    assert set(result.timings_ms) == {"decode", "parse", "gates"}


def test_failed_gates_listed_in_details():
    card = render_card(make_aamva_text(fields=dict(FAKE_FIELDS, DBA="20200101")))
    result, _ = run_barcode_stage(card, CLAIMED, NOW)
    assert result.details["gates_failed"] == ["EXPIRY_DATE_MISMATCH", "LICENCE_EXPIRED"]


def test_personal_data_only_in_checks():
    result, _ = run_barcode_stage(render_card(make_aamva_text()), CLAIMED, NOW)
    checks = result.details.pop("checks")
    assert len(checks) == 7 and all(c["passed"] for c in checks)
    blob = json.dumps(result.details)
    for secret in ("TESTER", "ALEX", "JAMIE", "19900315", "1990-03-15", FAKE_FIELDS["DAQ"], "1 TEST ST"):
        assert secret not in blob
    assert "1 TEST ST" not in json.dumps(checks)  # the address is never compared, so never in a check row


def test_garbage_pdf417_read_does_not_stop_the_search(monkeypatch):
    """A rotated photo can give a garbage PDF417 read on one variant; the decoder must keep looking for AAMVA."""
    import zxingcpp

    from backend.core.stages import barcode

    class Hit:
        def __init__(self, text):
            self.bytes = text.encode("latin-1")

    calls = iter([[Hit('HNA "\t21,:*')], [], [Hit(make_aamva_text())]])
    monkeypatch.setattr(zxingcpp, "read_barcodes", lambda *a, **k: next(calls, []))
    assert barcode.decode_pdf417(np.full((300, 400, 3), 200, np.uint8)).startswith("@")
    calls = iter([[Hit("hello")]])
    assert barcode.decode_pdf417(np.full((300, 400, 3), 200, np.uint8)) == "hello"  # non-AAMVA still reported
