import os
import sys
import numpy as np
import pytest

# Ensure backend root is on path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import cv2
from core.vehicle_alpr import (
    vehicle_alpr_engine,
    calculate_plate_match_score,
    normalize_plate_string,
    group_horizontal_tokens,
    is_slogan_noise,
)
from app.schemas import VehicleDetails


def _find_specimen(filename: str) -> str:
    candidates = [
        os.path.join(backend_dir, "tests", "plate_verification", filename),
        os.path.join(backend_dir, "..", "test_cases_extracted", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return ""


def test_alberta_ckz_3449_full_extraction():
    """
    Jira INK-7 Regression Test:
    Verifies that Alberta license plate 'CKZ-3449' is extracted in its entirety
    (not truncated to 'CKZ') with 100% similarity score and passed verification.
    """
    test_img_path = _find_specimen("tony_vehicle_photo.png")
    if not test_img_path or not os.path.exists(test_img_path):
        pytest.skip("Test specimen tony_vehicle_photo.png not found")

    img = cv2.imread(test_img_path)
    assert img is not None

    target = "CKZ-3449"
    plate, matched, score, box_det, crop, raw_texts = vehicle_alpr_engine.extract_plate_number(img, target)

    assert matched is True
    assert score >= 0.90
    assert plate in ("CKZ-3449", "CKZ3449")
    assert crop is not None and crop.size > 0

    # Also test through high-level verify_vehicle
    details = VehicleDetails(
        make="Standard",
        model="Vehicle",
        year="2022",
        plate=target,
        color="White",
    )
    res = vehicle_alpr_engine.verify_vehicle(img, details)
    assert res.passed is True
    assert res.extracted_plate in ("CKZ-3449", "CKZ3449")
    assert res.plate_matched is True
    assert res.plate_similarity >= 0.90


def test_horizontal_token_grouping_split_plate():
    """
    Verifies that horizontally adjacent word tokens separated by spaces,
    dashes, or provincial symbols (e.g. ['CKZ', '3449']) are reconstructed into composite plate candidates.
    """
    synthetic_ocr_lines = [
        ([[0, 20], [60, 20], [60, 60], [0, 60]], "CKZ", 0.99),
        ([[75, 22], [145, 22], [145, 62], [75, 62]], "3449", 0.99),
        ([[10, 75], [130, 75], [130, 90], [10, 90]], "Wild Rose Country", 0.95),
    ]

    candidates = group_horizontal_tokens(synthetic_ocr_lines)
    candidate_texts = [c[0] for c in candidates]

    assert "CKZ" in candidate_texts
    assert "3449" in candidate_texts
    assert "CKZ-3449" in candidate_texts or "CKZ 3449" in candidate_texts or "CKZ3449" in candidate_texts

    # Verify matching against CKZ-3449
    scores = [calculate_plate_match_score(c[0], "CKZ-3449") for c in candidates]
    best_match = any(m is True and s >= 0.90 for m, s in scores)
    assert best_match is True


def test_slogan_noise_filtering():
    """
    Verifies that provincial headers/slogans ('ALBERTA', 'WILD ROSE COUNTRY', etc.)
    are flagged as slogan noise so they do not out-score actual plates.
    """
    assert is_slogan_noise("Wild Rose Country") is True
    assert is_slogan_noise("Alberta") is True
    assert is_slogan_noise("Yours to Discover") is True
    assert is_slogan_noise("Beautiful British Columbia") is True
    assert is_slogan_noise("Friendly Manitoba") is True

    # Real plates containing digits MUST NOT be flagged as slogan noise
    assert is_slogan_noise("CKZ-3449") is False
    assert is_slogan_noise("DL 7CQ 1939") is False
    assert is_slogan_noise("A99-3BV") is False


def test_plate_match_score_ambiguity_and_spacing():
    """
    Verifies plate match score calculation across spacing, hyphens, and OCR ambiguities.
    """
    # Exact match with hyphens/spaces
    matched, score = calculate_plate_match_score("CKZ-3449", "CKZ-3449")
    assert matched is True and score == 1.0

    matched, score = calculate_plate_match_score("CKZ 3449", "CKZ-3449")
    assert matched is True and score == 1.0

    matched, score = calculate_plate_match_score("CKZ3449", "CKZ-3449")
    assert matched is True and score == 1.0

    # Incomplete prefix only (the original bug) must NOT trigger a match
    matched, score = calculate_plate_match_score("CKZ", "CKZ-3449")
    assert matched is False
    assert score < 0.70


def test_manitoba_fbf_951_full_extraction():
    """
    Verifies that Manitoba license plate 'FBF 951' (from Test Cases.docx)
    is properly recognized even in close-up cropped shots without failing as 'None'.
    """
    test_img_path = _find_specimen("manitoba_plate_specimen.png")
    if not test_img_path or not os.path.exists(test_img_path):
        pytest.skip("Test specimen manitoba_plate_specimen.png not found")

    img = cv2.imread(test_img_path)
    assert img is not None

    target = "FBF 951"
    plate, matched, score, box_det, crop, raw_texts = vehicle_alpr_engine.extract_plate_number(img, target)

    assert matched is True
    assert score >= 0.90
    assert plate in ("FBF 951", "FBF951")


if __name__ == "__main__":
    print("Running test_alberta_ckz_3449_full_extraction...")
    test_alberta_ckz_3449_full_extraction()
    print("  [PASS]")

    print("Running test_manitoba_fbf_951_full_extraction...")
    test_manitoba_fbf_951_full_extraction()
    print("  [PASS]")

    print("Running test_horizontal_token_grouping_split_plate...")
    test_horizontal_token_grouping_split_plate()
    print("  [PASS]")

    print("Running test_slogan_noise_filtering...")
    test_slogan_noise_filtering()
    print("  [PASS]")

    print("Running test_plate_match_score_ambiguity_and_spacing...")
    test_plate_match_score_ambiguity_and_spacing()
    print("  [PASS]")

    print("\nALL 5 TEST CASES PASSED SUCCESSFULLY!")


