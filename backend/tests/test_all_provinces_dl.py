from datetime import date
from unittest.mock import patch
import numpy as np
import pytest

from app.schemas import LicenseDetails, PersonalInfo
from core.canadian_dl_grammar import (
    AAMVA_IIN_MAP,
    PROVINCIAL_DL_REGEX,
    check_embedded_province_rules,
    clean_dl_string,
    format_canadian_dl,
    lookup_iin_province,
    normalize_province_code,
    repair_provincial_dl,
)
from core.dl_ocr import dl_ocr_engine, sanitize_privacy_tokens, OCRBox


# ---------------------------------------------------------------------------
# 1. Master Provincial Coverage & Regex Tests
# ---------------------------------------------------------------------------

def test_all_13_jurisdictions_iin_lookup():
    """Verify all 13 Canadian AAMVA IIN codes map correctly."""
    assert lookup_iin_province("636012") == "ON"
    assert lookup_iin_province("604428") == "QC"
    assert lookup_iin_province("636028") == "BC"
    assert lookup_iin_province("604432") == "AB"
    assert lookup_iin_province("636044") == "SK"
    assert lookup_iin_province("636048") == "MB"
    assert lookup_iin_province("636017") == "NB"
    assert lookup_iin_province("636013") == "NS"
    assert lookup_iin_province("604426") == "PE"
    assert lookup_iin_province("636016") == "NL"
    assert lookup_iin_province("604429") == "YT"
    assert lookup_iin_province("604434") == "NT"
    assert lookup_iin_province("604433") == "NU"
    # Legacy 5-digit normalization (without leading 6)
    assert lookup_iin_province("36012") == "ON"
    assert lookup_iin_province("04428") == "QC"


def test_ontario_embedded_anatomy_and_reversed_names():
    """Test Ontario: Digits 9-10 (YY), 11-12 (MM), 13-14 (DD) and surname initial."""
    # Person: Jane Public, born 1990-02-15 -> Digits 9-14: 900215
    valid_on = "P12345678900215"
    passed, fails = check_embedded_province_rules(
        dl_number=valid_on,
        province="ON",
        surname="Public",
        given_names="Jane Q",
        dob=date(1990, 2, 15),
    )
    assert passed is True
    assert fails == []

    # Reversed name: Initial 'J' (from Jane) instead of 'P' (from Public)
    reversed_on = "J12345678900215"
    passed_rev, fails_rev = check_embedded_province_rules(
        dl_number=reversed_on,
        province="ON",
        surname="Public",
        given_names="Jane Q",
        dob=date(1990, 2, 15),
    )
    assert passed_rev is True  # Soft warning, doesn't reject
    assert "names_reversed" in fails_rev

    # Mismatched birth year: 91 instead of 90
    mismatch_on = "P12345678910215"
    passed_bad, fails_bad = check_embedded_province_rules(
        dl_number=mismatch_on,
        province="ON",
        surname="Public",
        given_names="Jane",
        dob=date(1990, 2, 15),
    )
    assert passed_bad is False
    assert "birth_year_mismatch" in fails_bad


def test_quebec_embedded_birthdate_ddmmyy():
    """Test Quebec: Surname initial + 4 digits + DDMMYY (digits 5-10) + 2 digits."""
    # Person: Marie-Eve Lavoie, born 1990-02-15 -> DDMMYY = 150290
    valid_qc = "L1234-150290-01"
    repaired, conf, is_valid = repair_provincial_dl(valid_qc, province="QC", surname="Lavoie")
    assert is_valid is True
    assert repaired == "L123415029001"

    passed, fails = check_embedded_province_rules(
        dl_number=repaired,
        province="QC",
        surname="Lavoie",
        given_names="Marie-Eve",
        dob=date(1990, 2, 15),
    )
    assert passed is True
    assert fails == []

    # SAAQ specimen: L1531-171274-08 belongs to holder born 1974-12-17
    saaq_specimen = "L153117127408"
    passed_saaq, _ = check_embedded_province_rules(
        dl_number=saaq_specimen,
        province="QC",
        surname="Lavoie",
        dob=date(1974, 12, 17),
    )
    assert passed_saaq is True

    # Bad birth date in QC number
    passed_bad_dob, fails_bad_dob = check_embedded_province_rules(
        dl_number="L123415029001",
        province="QC",
        surname="Lavoie",
        dob=date(1985, 5, 20),
    )
    assert passed_bad_dob is False
    assert "birth_date_ddmmyy_mismatch" in fails_bad_dob


def test_nova_scotia_embedded_surname_and_ddmmyy():
    """Test Nova Scotia: First 5 surname chars + DDMMYY + 3 digits."""
    # Person: Jane Public, born 1990-02-15 -> PUBLI + 150290 + 123
    valid_ns = "PUBLI150290123"
    repaired, conf, is_valid = repair_provincial_dl(valid_ns, province="NS", surname="Public")
    assert is_valid is True
    assert repaired == "PUBLI150290123"

    passed, fails = check_embedded_province_rules(
        dl_number=repaired,
        province="NS",
        surname="Public",
        dob=date(1990, 2, 15),
    )
    assert passed is True
    assert fails == []

    # Optical confusion repair on NS: 0 for O, etc.
    noisy_ns = "PUBLI15O29O123"  # 'O' instead of '0'
    repaired_noisy, _, valid_noisy = repair_provincial_dl(noisy_ns, province="NS", surname="Public")
    assert valid_noisy is True
    assert repaired_noisy == "PUBLI150290123"


def test_newfoundland_embedded_yymmdd():
    """Test Newfoundland & Labrador: Surname initial + YYMMDD + 3 digits."""
    # Person: Jane Public, born 1990-02-15 -> P + 900215 + 123
    valid_nl = "P900215123"
    repaired, conf, is_valid = repair_provincial_dl(valid_nl, province="NL", surname="Public")
    assert is_valid is True
    assert repaired == "P900215123"

    passed, fails = check_embedded_province_rules(
        dl_number=repaired,
        province="NL",
        surname="Public",
        dob=date(1990, 2, 15),
    )
    assert passed is True
    assert fails == []

    formatted = format_canadian_dl(repaired, "NL")
    assert formatted == "P-900215-123"


def test_saskatchewan_month_end_expiry_rule():
    """Test Saskatchewan: 8 digits and expiry on last day of birth month."""
    valid_sk = "12345678"
    repaired, conf, is_valid = repair_provincial_dl(valid_sk, province="SK")
    assert is_valid is True
    assert repaired == "12345678"

    # Driver born in February (leap year 2000) -> Expiry in Feb 2028 must be Feb 29
    passed_sk, fails_sk = check_embedded_province_rules(
        dl_number=valid_sk,
        province="SK",
        dob=date(2000, 2, 15),
        expiry=date(2028, 2, 29),
    )
    assert passed_sk is True
    assert fails_sk == []

    # Driver born in August -> Expiry not at month-end (Aug 15 instead of Aug 31)
    passed_bad, fails_bad = check_embedded_province_rules(
        dl_number=valid_sk,
        province="SK",
        dob=date(1995, 8, 10),
        expiry=date(2027, 8, 15),
    )
    assert passed_bad is False
    assert "sk_expiry_not_monthend" in fails_bad


def test_british_columbia_7_and_8_digits():
    """Test BC: Supports both 7-digit and 8-digit formats."""
    # 7 digits
    rep7, _, valid7 = repair_provincial_dl("1234567", province="BC")
    assert valid7 is True
    assert rep7 == "1234567"

    # 8 digits with OCR confusion ('I' -> '1', 'S' -> '5')
    rep8, _, valid8 = repair_provincial_dl("I234S678", province="BC")
    assert valid8 is True
    assert rep8 == "12345678"


def test_alberta_9_digits_and_older_patterns():
    """Test Alberta: 9 digits (123456-789) and older patterns (A12345, AB1234)."""
    # 9 digits with dashes
    rep_ab, _, valid_ab = repair_provincial_dl("123456-789", province="AB")
    assert valid_ab is True
    assert rep_ab == "123456789"
    assert format_canadian_dl(rep_ab, "AB") == "123456-789"

    # Older pattern: A12345
    rep_old, _, valid_old = repair_provincial_dl("A12345", province="AB")
    assert valid_old is True
    assert rep_old == "A12345"


def test_manitoba_name_derived_and_asterisks():
    """Test Manitoba: Name-derived letters with dashes and asterisks."""
    mb_raw = "PUBLIJQ123AB"
    rep_mb, _, valid_mb = repair_provincial_dl(mb_raw, province="MB", surname="Public")
    assert valid_mb is True
    assert rep_mb == "PUBLIJQ123AB"

    # Formatted
    fmt_mb = format_canadian_dl(rep_mb, "MB")
    assert fmt_mb == "PUBLI-JQ123-AB"


def test_territories_and_maritimes():
    """Test NB, PE, YT, NT, NU formats."""
    # New Brunswick: up to 7 digits
    rep_nb, _, valid_nb = repair_provincial_dl("1234567", province="NB")
    assert valid_nb is True

    # Prince Edward Island: up to 6 digits
    rep_pe, _, valid_pe = repair_provincial_dl("123456", province="PE")
    assert valid_pe is True

    # Yukon: up to 6 digits
    rep_yt, _, valid_yt = repair_provincial_dl("123456", province="YT")
    assert valid_yt is True

    # Northwest Territories: 5 to 9 digits
    rep_nt, _, valid_nt = repair_provincial_dl("123456789", province="NT")
    assert valid_nt is True

    # Nunavut: 5 to 6 digits
    rep_nu, _, valid_nu = repair_provincial_dl("123456", province="NU")
    assert valid_nu is True


# ---------------------------------------------------------------------------
# 2. Multilingual Spatial Anchors & French Date Parsing
# ---------------------------------------------------------------------------

def test_quebec_french_spatial_anchors():
    """Verify that French numbered labels from Quebec card are correctly parsed."""
    boxes = [
        OCRBox(text="4d NO PERMIS", confidence=0.95, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=50.0, center_y=100.0),
        OCRBox(text="L1234-150290-01", confidence=0.98, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=200.0, center_y=100.0),
        OCRBox(text="3 DATE DE NAISSANCE", confidence=0.95, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=50.0, center_y=150.0),
        OCRBox(text="1990-02-15", confidence=0.98, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=200.0, center_y=150.0),
        OCRBox(text="4b EXPIRE LE", confidence=0.95, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=50.0, center_y=200.0),
        OCRBox(text="2029-02-15", confidence=0.98, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=200.0, center_y=200.0),
        OCRBox(text="4a VALIDE LE", confidence=0.95, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=50.0, center_y=250.0),
        OCRBox(text="2024-02-15", confidence=0.98, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=200.0, center_y=250.0),
        OCRBox(text="15 SEXE F", confidence=0.95, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=50.0, center_y=300.0),
        OCRBox(text="9 CLASSE 5", confidence=0.95, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=50.0, center_y=350.0),
        OCRBox(text="1 NOM LAVOIE", confidence=0.95, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=50.0, center_y=400.0),
        OCRBox(text="2 PRÉNOM MARIE-ÈVE", confidence=0.95, box=[[0, 0], [10, 0], [10, 10], [0, 10]], center_x=50.0, center_y=450.0),
    ]

    fields = dl_ocr_engine.extract_spatial_aamva_fields(boxes, province="QC")
    assert fields["license_number"] == "L1234-150290-01"
    assert fields["dob"] == "1990/02/15"
    assert fields["expiry_date"] == "2029/02/15"
    assert fields["issue_date"] == "2024/02/15"
    assert fields["gender"] == "F"
    assert fields["license_class"] == "5"
    assert fields["surname"] == "LAVOIE"
    assert fields["given_name"] == "MARIE-ÈVE"


def test_alberta_and_saskatchewan_date_matching():
    """Verify text month matching (e.g. '15 FEB 1990' and '31Aug2013')."""
    lines_ab = ["Driver's Licence", "Alberta", "15 FEB 1990", "15 OCT 2029"]
    score_dob, m_dob = dl_ocr_engine._match_date(lines_ab, "1990-02-15")
    assert score_dob == 100.0
    assert m_dob == "1990/02/15"

    score_exp, m_exp = dl_ocr_engine._match_date(lines_ab, "2029-10-15")
    assert score_exp == 100.0
    assert m_exp == "2029/10/15"

    # Saskatchewan compact date: '31Aug2013'
    lines_sk = ["Saskatchewan", "31Aug2013"]
    score_sk, m_sk = dl_ocr_engine._match_date(lines_sk, "2013-08-31")
    assert score_sk == 100.0
    assert m_sk == "2013/08/31"


def test_privacy_sanitizer_masks_phn_and_citizenship():
    """Verify that BC/AB 10-digit PHNs and citizenship markers 'CAN' are redacted."""
    raw_bc_text = "BC Services Card PHN 9876543210 CAN DO NOT STORE"
    sanitized = sanitize_privacy_tokens(raw_bc_text)
    assert "9876543210" not in sanitized
    assert "[REDACTED_PHN]" in sanitized
    assert "CAN" not in sanitized.split()
    assert "[REDACTED_CAN]" in sanitized


# ---------------------------------------------------------------------------
# 3. End-to-End Verification Pipeline for Quebec & Alberta Cards
# ---------------------------------------------------------------------------

def test_end_to_end_quebec_card_verification():
    """Verify full autonomous verification for a Quebec driver's licence."""
    mock_img = np.zeros((300, 500, 3), dtype=np.uint8)
    mock_lines = [
        "QUÉBEC · PERMIS DE CONDUIRE",
        "1 · NOM LAVOIE",
        "2 · PRÉNOM MARIE-ÈVE",
        "4d NO PERMIS L1234-150290-01",
        "3 DATE DE NAISSANCE 1990-02-15",
        "4a VALIDE LE 2024-02-15",
        "4b EXPIRE LE 2029-02-15",
        "8 ADRESSE 123 RUE PRINCIPALE",
        "QUÉBEC QC G1A 1A1",
        "9 CLASSE 5",
        "15 SEXE F",
    ]

    personal = PersonalInfo(
        full_name="Marie-Eve Lavoie",
        date_of_birth="1990-02-15",
        email_address="marie.eve@test.com",
        mobile_number="1234567890",
        role="Driver",
    )
    details = LicenseDetails(
        license_number="L123415029001",
        license_expiry_date="2029-02-15",
        issuing_province="QC",
    )

    with patch.object(dl_ocr_engine, "extract_text_and_boxes", return_value=(mock_lines, [])):
        with patch("core.dl_ocr.assess_image_quality", return_value=(True, "High quality", {"blur_variance": 500.0})):
            res = dl_ocr_engine.verify_license(mock_img, personal, details)

    assert res.passed is True
    assert res.jurisdiction == "QC"
    assert res.card_version == "QC-2023"
    assert res.number_rules_pass is True
    assert res.extracted_dl_number == "L1234-150290-01"
    assert res.dob_matched is True
    assert res.is_expired is False


def test_end_to_end_alberta_card_verification():
    """Verify full autonomous verification for an Alberta driver's licence with text dates."""
    mock_img = np.zeros((300, 500, 3), dtype=np.uint8)
    mock_lines = [
        "ALBERTA DRIVER'S LICENCE",
        "SMITH, JOHN",
        "123 MAIN ST",
        "CALGARY, AB T2P 1J9",
        "4d 123456-789",
        "3 DOB 15 FEB 1990",
        "4b EXP 15 FEB 2029",
        "4a ISS 15 FEB 2024",
        "9 CLASS 5",
        "15 SEX M",
    ]

    personal = PersonalInfo(
        full_name="John Smith",
        date_of_birth="1990-02-15",
        email_address="john.smith@test.com",
        mobile_number="1234567890",
        role="Driver",
    )
    details = LicenseDetails(
        license_number="123456789",
        license_expiry_date="2029-02-15",
        issuing_province="AB",
    )

    with patch.object(dl_ocr_engine, "extract_text_and_boxes", return_value=(mock_lines, [])):
        with patch("core.dl_ocr.assess_image_quality", return_value=(True, "High quality", {"blur_variance": 500.0})):
            res = dl_ocr_engine.verify_license(mock_img, personal, details)

    assert res.passed is True
    assert res.jurisdiction == "AB"
    assert res.card_version == "AB-2026"
    assert res.number_rules_pass is True
    assert res.extracted_dl_number == "123456-789"
    assert res.dob_matched is True
    assert res.is_expired is False
