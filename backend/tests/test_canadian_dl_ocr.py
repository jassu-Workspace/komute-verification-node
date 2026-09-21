import pytest
import numpy as np
from core.canadian_dl_grammar import (
    repair_provincial_dl,
    format_canadian_dl,
    normalize_province_code,
    clean_dl_string,
)
from core.image_utils import assess_image_quality
from core.dl_ocr import dl_ocr_engine
from app.schemas import PersonalInfo, LicenseDetails


def test_ontario_dl_grammar_and_repair():
    """Test Ontario: 1 letter matching surname + 14 digits."""
    # Test OCR confusion: 'O' instead of '0', 'I' instead of '1', 'S' instead of '5'
    raw_ocr_ontario = "D1234-56789-O123S"
    repaired, conf, is_valid = repair_provincial_dl(raw_ocr_ontario, province="ON", surname="Doe")
    assert is_valid is True
    assert repaired == "D12345678901235"
    assert repaired[0] == "D"
    assert repaired[1:].isdigit()
    assert len(repaired) == 15

    formatted = format_canadian_dl(repaired, "ON")
    assert formatted == "D1234-56789-01235"


def test_bc_dl_grammar_and_repair():
    """Test British Columbia: 7 digits with character confusion repair."""
    raw_ocr_bc = "I234S67"  # 'I' -> '1', 'S' -> '5'
    repaired, conf, is_valid = repair_provincial_dl(raw_ocr_bc, province="BC")
    assert is_valid is True
    assert repaired == "1234567"
    assert len(repaired) == 7


def test_alberta_dl_grammar_and_repair():
    """Test Alberta: 7-9 numeric digits."""
    raw_ocr_ab = "123456-789"
    repaired, conf, is_valid = repair_provincial_dl(raw_ocr_ab, province="AB")
    assert is_valid is True
    assert repaired == "123456789"
    assert len(repaired) == 9


def test_quebec_dl_grammar_and_repair():
    """Test Quebec: 1 letter + 12 digits (13 chars)."""
    raw_ocr_qc = "C123456789012"
    repaired, conf, is_valid = repair_provincial_dl(raw_ocr_qc, province="QC", surname="Curie")
    assert is_valid is True
    assert repaired == "C123456789012"
    assert len(repaired) == 13


def test_manitoba_and_saskatchewan_repair():
    """Test Manitoba (1 letter + 13 digits) and Saskatchewan (8 digits)."""
    mb_raw = "M1234567890123"
    repaired_mb, _, valid_mb = repair_provincial_dl(mb_raw, province="MB", surname="Miller")
    assert valid_mb is True
    assert len(repaired_mb) == 14

    sk_raw = "12345678"
    repaired_sk, _, valid_sk = repair_provincial_dl(sk_raw, province="SK")
    assert valid_sk is True
    assert len(repaired_sk) == 8


def test_province_normalization():
    """Test full province name mapping to canonical 2-letter codes."""
    assert normalize_province_code("Ontario") == "ON"
    assert normalize_province_code("British Columbia") == "BC"
    assert normalize_province_code("Alberta") == "AB"
    assert normalize_province_code("Quebec") == "QC"
    assert normalize_province_code("ON") == "ON"


def test_preflight_quality_gate_extreme_blur():
    """Test that severely blurred images are flagged by the quality gate."""
    # Create black or completely flat image (variance 0)
    flat_img = np.full((300, 500, 3), 128, dtype=np.uint8)
    is_usable, msg, metrics = assess_image_quality(flat_img)
    assert is_usable is False
    assert metrics["blur_variance"] < 40.0


def test_canadian_dl_matching_in_engine():
    """Test end-to-end Canadian DL number matching with OCR confusion in dl_ocr_engine."""
    lines = [
        "ONTARIO DRIVER LICENCE",
        "1. DOE",
        "2. JOHN",
        "4d D6101-40706-609OS",  # 'O' -> '0', 'S' -> '5'
        "3. 1966-09-05",
        "4b. 2029-04-23",
    ]
    target_dl = "D61014070660905"
    score, matched = dl_ocr_engine._match_dl_number(lines, target_dl, province="ON", surname="Doe")
    assert score >= 99.0
    assert matched is not None
