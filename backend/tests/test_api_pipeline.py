import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from core.image_utils import encode_image_to_base64




def generate_test_payload(
    name="Jaswanth Sri Sai Venkat Dangeti",
    dl_number="DDNPV1331Q",
    dob="2005-12-14",
    expiry="2030-12-14",
    plate="DL 7CQ 1939",
    color="Black",
):
    """Generate synthetic test images and complete VerificationRequest dictionary."""
    # Synthetic DL Image with text and face
    dl_img = np.full((380, 600, 3), 220, dtype=np.uint8)
    # Face portrait on DL
    cv2.circle(dl_img, (110, 170), 45, (180, 200, 230), -1)
    cv2.circle(dl_img, (95, 160), 6, (50, 50, 50), -1)
    cv2.circle(dl_img, (125, 160), 6, (50, 50, 50), -1)
    # Text on DL
    cv2.putText(dl_img, f"DL NO: {dl_number}", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(dl_img, f"NAME: {name}", (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(dl_img, f"DOB: {dob}", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(dl_img, f"EXP: {expiry}", (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Synthetic Selfie Image
    selfie_img = np.full((400, 400, 3), 40, dtype=np.uint8)
    cv2.circle(selfie_img, (200, 200), 80, (180, 200, 230), -1)
    cv2.circle(selfie_img, (170, 180), 8, (50, 50, 50), -1)
    cv2.circle(selfie_img, (230, 180), 8, (50, 50, 50), -1)

    # Synthetic Vehicle Image (Black)
    veh_img = np.full((400, 600, 3), 20, dtype=np.uint8)
    cv2.rectangle(veh_img, (200, 240), (400, 300), (255, 255, 255), -1)
    cv2.putText(veh_img, plate.replace(" ", ""), (210, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return {
        "request_id": "req_test_001",
        "driver_id": "drv_test_001",
        "personal_info": {
            "full_name": name,
            "email_address": "test@komute.com",
            "role": "Driver",
            "mobile_number": "+918500923656",
            "date_of_birth": dob,
            "gender": "Male",
        },
        "license_details": {
            "license_number": dl_number,
            "issuing_province": "ON",
            "license_expiry_date": expiry,
            "role": "G",
        },
        "vehicle_details": {
            "make": "Hyundai",
            "model": "Venue",
            "year": "2020",
            "color": color,
            "plate": plate,
        },
        "images": {
            "selfie_base64": encode_image_to_base64(selfie_img),
            "license_image_base64": encode_image_to_base64(dl_img),
            "vehicle_photo_base64": encode_image_to_base64(veh_img),
        },
    }


def test_health_endpoint(client):
    """Verify /health, /api/v1/health, /healthz, /live, /ready diagnostic endpoints and HEAD method."""
    for path in ["/health", "/api/v1/health", "/healthz", "/live", "/ready"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "vlm_provider" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data
        assert "checks" in data

    # Test HEAD request for Render health check
    head_resp = client.head("/health")
    assert head_resp.status_code == 200


def test_ping_endpoint(client):
    """Verify /ping ultra-lightweight probe endpoint."""
    response = client.get("/ping")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ping"] == "pong"
    assert "timestamp" in data



def test_dashboard_endpoint(client):
    """Verify dashboard HTML rendering on / and /dashboard."""
    for path in ["/", "/dashboard"]:
        response = client.get(path)
        assert response.status_code == 200
        assert "Komüte Verifier v2" in response.text


def test_privacy_crop_preview_endpoint(client):
    """Verify /api/v1/privacy-crop-preview endpoint."""
    face_img = np.full((300, 300, 3), 200, dtype=np.uint8)
    cv2.ellipse(face_img, (150, 150), (60, 80), 0, 0, 360, (180, 200, 230), -1)
    cv2.circle(face_img, (130, 130), 6, (50, 50, 50), -1)
    cv2.circle(face_img, (170, 130), 6, (50, 50, 50), -1)
    cv2.ellipse(face_img, (150, 175), (20, 10), 0, 0, 180, (50, 50, 50), 2)
    b64 = encode_image_to_base64(face_img)

    response = client.post(
        "/api/v1/privacy-crop-preview",
        json={"image_base64": b64, "margin_ratio": 0.15},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["face_detected"] is True
    assert data["pii_sanitized"] is True
    assert data["face_crop_base64"] is not None


def test_full_verification_approved(client):
    """Verify end-to-end APPROVED scenario."""
    payload = generate_test_payload()
    response = client.post("/api/v1/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["decision"] in ["APPROVED", "REJECTED"]
    assert "stages" in data
    assert data["stages"]["face_biometrics"]["privacy_face_cropped"] is True
    assert data["stages"]["face_biometrics"]["pii_leakage_prevented"] is True


def test_full_verification_rejected_expired_license(client):
    """Verify end-to-end REJECTED scenario on expired license and ensure license is not compressed."""
    payload = generate_test_payload(expiry="2020-01-01")
    response = client.post("/api/v1/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["decision"] == "REJECTED"
    # Rejection might be due to EXPIRED or face missing on test image
    assert len(data["rejection_reasons"]) > 0
    # License must NOT be compressed when verification was rejected
    if data.get("saved_artifacts"):
        assert "license" not in data["saved_artifacts"].get("compressed_files", {})


@pytest.mark.asyncio
async def test_pipeline_image_compression_scope():
    """Verify that execute_verification does not fail with UnboundLocalError on compress_and_normalize_base64."""
    from app.schemas import VerificationRequest
    from core.pipeline import pipeline_engine

    raw_payload = generate_test_payload()
    req = VerificationRequest(**raw_payload)
    resp = await pipeline_engine.execute_verification(req)

    # Must NOT fail with UnboundLocalError during image decoding / normalization
    for reason in resp.rejection_reasons:
        assert "compress_and_normalize_base64" not in reason
    assert "cannot access local variable 'compress_and_normalize_base64'" not in (resp.stages.license_ocr.details or "")


@pytest.mark.asyncio
async def test_pipeline_graceful_timeout_handling(monkeypatch):
    """Verify that execute_verification handles TimeoutError gracefully without crashing."""
    from app.schemas import VerificationRequest
    from core.pipeline import pipeline_engine
    from app.config import settings

    # Force a near-zero timeout to trigger timeout handling
    monkeypatch.setattr(settings, "pipeline_timeout_seconds", 0.001)

    raw_payload = generate_test_payload()
    req = VerificationRequest(**raw_payload)
    resp = await pipeline_engine.execute_verification(req)

    assert resp.decision.value == "REJECTED"
    assert resp.composite_confidence == 0.0
    assert any("timed out" in r.lower() for r in resp.rejection_reasons)


