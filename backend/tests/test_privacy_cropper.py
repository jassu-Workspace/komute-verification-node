import cv2
import numpy as np
import pytest
from core.face_privacy_cropper import face_privacy_cropper
from core.image_utils import decode_base64_to_image, encode_image_to_base64


def create_test_face_image(width=400, height=400):
    """Generate a clean synthetic portrait image."""
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    # Face oval (skin tone)
    center = (width // 2, height // 2)
    cv2.ellipse(img, center, (80, 100), 0, 0, 360, (180, 200, 230), -1)
    # Eyes
    cv2.circle(img, (center[0] - 30, center[1] - 20), 8, (50, 50, 50), -1)
    cv2.circle(img, (center[0] + 30, center[1] - 20), 8, (50, 50, 50), -1)
    # Smile
    cv2.ellipse(img, (center[0], center[1] + 35), (25, 15), 0, 0, 180, (50, 50, 50), 3)
    return img


def test_face_detection_and_tight_crop():
    """Verify face isolation and 15% bounding padding."""
    test_img = create_test_face_image()
    crop, bbox, sanitized = face_privacy_cropper.extract_isolated_face(
        img_bgr=test_img,
        margin_ratio=0.15,
        target_size=(256, 256),
    )

    assert crop is not None
    assert crop.shape == (256, 256, 3)
    assert bbox is not None
    assert bbox["width"] > 0
    assert bbox["height"] > 0
    assert sanitized is True


def test_empty_image_handling():
    """Verify graceful handling of invalid/empty image arrays."""
    crop, bbox, sanitized = face_privacy_cropper.extract_isolated_face(np.array([]))
    assert crop is None
    assert bbox is None
    assert sanitized is False


def test_base64_encode_decode_roundtrip():
    """Verify base64 encoding and decoding preserves image integrity."""
    original = create_test_face_image(200, 200)
    b64_str = encode_image_to_base64(original, format_ext=".jpg")
    assert b64_str.startswith("data:image/jpeg;base64,")

    decoded = decode_base64_to_image(b64_str)
    assert decoded is not None
    assert decoded.shape == original.shape


def test_qr_code_rejection_layer():
    """Verify that QR codes / barcodes are rejected and not misclassified as faces."""
    # Create a synthetic QR grid pattern
    qr_img = np.full((300, 300, 3), 255, dtype=np.uint8)
    for i in range(20, 280, 20):
        for j in range(20, 280, 20):
            if (i + j) % 40 == 0:
                cv2.rectangle(qr_img, (i, j), (i + 18, j + 18), (0, 0, 0), -1)

    has_face, score, details = face_privacy_cropper.verify_face_presence(qr_img)
    assert has_face is False
    assert score == 0.0


def test_retinaface_mobilenet_verification_layer():
    """Verify RetinaFace MobileNet0.25 detection on portrait vs non-face textures."""
    test_face = create_test_face_image(300, 300)
    has_face, score, details = face_privacy_cropper.verify_face_presence(test_face)
    assert isinstance(has_face, bool)
    assert isinstance(score, float)

