import cv2
import numpy as np
import pytest
from app.schemas import VehicleDetails
from core.vehicle_alpr import (
    calculate_plate_match_score,
    normalize_plate_string,
    vehicle_alpr_engine,
)


def test_plate_normalization():
    """Verify plate normalization and confusion character mapping."""
    raw = "DL 7CQ 1939"
    assert normalize_plate_string(raw) == "DL7CQ1939"
    # Canonical mapping: Q/O -> 0
    assert normalize_plate_string(raw, apply_confusion=True) == "DL7C01939"


def test_plate_character_confusion_matching():
    """Verify OCR ambiguity handling: 'O' vs '0', 'Q' vs '0'."""
    target = "DL 7CQ 1939"
    ocr_result_ambiguous = "DL7C01939"
    matched, score = calculate_plate_match_score(ocr_result_ambiguous, target)
    assert matched is True
    assert score >= 0.85

    ocr_exact = "DL7CQ1939"
    matched_exact, score_exact = calculate_plate_match_score(ocr_exact, target)
    assert matched_exact is True
    assert score_exact == 1.0

    ocr_mismatch = "MH12AB1234"
    matched_mismatch, score_mismatch = calculate_plate_match_score(ocr_mismatch, target)
    assert matched_mismatch is False
    assert score_mismatch < 0.40


def test_vehicle_color_detection():
    """Verify dominant color classification in HSV space."""
    # Create black vehicle image (low intensity)
    black_img = np.full((300, 400, 3), 20, dtype=np.uint8)
    matched, detected_color, conf = vehicle_alpr_engine.detect_vehicle_color(black_img, "Black")
    assert matched is True
    assert detected_color == "Black"

    # Create red vehicle image (BGR: [0, 0, 220])
    red_img = np.zeros((300, 400, 3), dtype=np.uint8)
    red_img[:, :] = (20, 20, 220)
    matched_red, detected_color_red, conf_red = vehicle_alpr_engine.detect_vehicle_color(red_img, "Red")
    assert matched_red is True
    assert detected_color_red == "Red"


def test_full_stage3_vehicle_verification():
    """Verify complete Stage 3 execution with plate OCR and color detection."""
    vehicle_img = np.full((400, 600, 3), 20, dtype=np.uint8)
    # Add a white plate rectangle with high-contrast text
    cv2.rectangle(vehicle_img, (200, 240), (420, 300), (255, 255, 255), -1)
    cv2.putText(vehicle_img, "DL7CQ1939", (210, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    details = VehicleDetails(
        make="Hyundai",
        model="Venue",
        year="2020",
        color="Black",
        plate="DL 7CQ 1939",
    )

    res = vehicle_alpr_engine.verify_vehicle(vehicle_img, details)
    assert res.color_matched is True
    assert res.plate_matched is True
    assert res.passed is True
