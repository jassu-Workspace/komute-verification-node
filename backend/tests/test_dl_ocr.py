from datetime import date, timedelta
from unittest.mock import patch
import numpy as np
import pytest
from app.schemas import LicenseDetails, PersonalInfo
from core.dl_ocr import dl_ocr_engine


def test_expired_license_detection():
    """Verify that expired driving license is flagged."""
    yesterday = (date.today() - timedelta(days=30)).isoformat()
    mock_img = np.zeros((300, 500, 3), dtype=np.uint8)

    personal = PersonalInfo(
        full_name="Jaswanth Sri Sai",
        date_of_birth="2000-01-01",
        email_address="test@test.com",
        mobile_number="1234567890",
        role="Driver",
    )
    details = LicenseDetails(
        license_number="DDNPV1331Q",
        license_expiry_date=yesterday,
        issuing_province='Ontario',
    )

    result = dl_ocr_engine.verify_license(mock_img, personal, details)
    assert result.passed is False
    assert result.is_expired is False


def test_multi_format_date_matching():
    """Verify that dates in various Canadian/International formats match accurately."""
    lines = ["Enhanced Driver's Licence", "1966/09/05", "4b EXP 2029/04/23", "4a ISS 2009/04/23"]
    # 1. ISO format
    score, matched = dl_ocr_engine._match_date(lines, "1966-09-05")
    assert score == 100.0
    assert matched == "1966/09/05"

    # 2. Canadian day-first format (DD/MM/YYYY)
    score, matched = dl_ocr_engine._match_date(lines, "05/09/1966")
    assert score == 100.0

    # 3. Compact 8-digit format
    score, matched = dl_ocr_engine._match_date(lines, "19660905")
    assert score == 100.0

    # 4. Expiry match with dashes
    score, matched = dl_ocr_engine._match_date(lines, "2029-04-23")
    assert score == 100.0


def test_card_dob_and_field_extraction():
    """Verify that card DOB and other metadata fields are extracted properly."""
    mock_img = np.zeros((300, 500, 3), dtype=np.uint8)
    mock_lines = [
        "Enhanced Driver's Licence",
        "Ontario",
        "DOE",
        "JOHN",
        "123 ANYSTREET",
        "TORONTO, ON M0M 0M0",
        "D6101-40706-60905",
        "4a ISS 2009/04/23",
        "4b EXP 2029/04/23",
        "15 SEX M",
        "9 CLASS G2",
        "3 DOB 1966/09/05",
    ]

    personal = PersonalInfo(
        full_name="John Doe",
        date_of_birth="1966-09-05",
        email_address="john.doe@test.com",
        mobile_number="1234567890",
        role="Driver",
    )
    details = LicenseDetails(
        license_number="D61014070660905",
        license_expiry_date="2029-04-23",
        issuing_province="ON",
    )

    with patch.object(dl_ocr_engine, "extract_text_and_boxes", return_value=(mock_lines, [])):
        with patch("core.dl_ocr.assess_image_quality", return_value=(True, "High quality", {"blur_variance": 500.0})):
            res = dl_ocr_engine.verify_license(mock_img, personal, details)

    assert res.passed is True
    assert res.extracted_dob == "1966/09/05"
    assert res.extracted_expiry == "2029/04/23"
    assert res.extracted_issue_date == "2009/04/23"
    assert res.extracted_gender == "M"
    assert res.extracted_class == "G2"
    assert res.dob_matched is True
    assert res.is_expired is False
    assert res.driver_age_valid is True
