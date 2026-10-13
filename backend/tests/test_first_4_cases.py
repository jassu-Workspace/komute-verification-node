import os
import sys
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.vehicle_alpr import vehicle_alpr_engine
from app.schemas import VehicleDetails

specimens_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "plate_verification"))

@pytest.mark.parametrize("province,ticket,filename,plate,expected_color", [
    ("Alberta", "INK-7", "tony_vehicle_photo.png", "CKZ-3449", "White"),
    ("Ontario", "Ontario Test Case", "1.webp", "CMBY-815", "White"),
    ("British Columbia", "BC Test Case", "2.webp", "627-BEJ", "White"),
    ("Manitoba", "Manitoba Test Case", "manitoba_plate_specimen.png", "FBF 951", "White"),
])
def test_first_4_provincial_cases_end_to_end(province, ticket, filename, plate, expected_color):
    """
    Validates end-to-end Stage 3 verification on the first 4 test cases from Test Cases.docx:
    - Alberta (Tony Thomas INK-7 regression: CKZ-3449)
    - Ontario (CMBY-815)
    - British Columbia (627-BEJ)
    - Manitoba (FBF 951)
    """
    img_path = os.path.join(specimens_dir, filename)
    assert os.path.exists(img_path), f"Specimen file missing: {img_path}"
    
    img = cv2.imread(img_path)
    assert img is not None, f"Failed to load image: {img_path}"
    
    details = VehicleDetails(
        make="Standard",
        model="Vehicle",
        year="2022",
        plate=plate,
        color=expected_color
    )
    
    res = vehicle_alpr_engine.verify_vehicle(img, details)
    
    assert res.passed is True, f"{province} ({ticket}) verification failed: {res.details}"
    assert res.plate_matched is True, f"{province} plate mismatch: extracted '{res.extracted_plate}', expected '{plate}'"
    assert res.plate_similarity >= 0.90, f"{province} plate similarity {res.plate_similarity} below 0.90"
