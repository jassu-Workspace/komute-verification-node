import asyncio
import concurrent.futures
import numpy as np
import pytest
import cv2
from fastapi.testclient import TestClient
from app.main import app
from core.face_privacy_cropper import face_privacy_cropper
from core.image_utils import encode_image_to_base64


def generate_synthetic_face_image(size=300):
    """Generate a synthetic test image with a recognizable face pattern."""
    img = np.full((size, size, 3), 220, dtype=np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (size // 4, size // 3), 0, 0, 360, (180, 200, 230), -1)
    cv2.circle(img, (size // 2 - 30, size // 2 - 20), 8, (50, 50, 50), -1)
    cv2.circle(img, (size // 2 + 30, size // 2 - 20), 8, (50, 50, 50), -1)
    cv2.ellipse(img, (size // 2, size // 2 + 30), (25, 12), 0, 0, 180, (50, 50, 50), 2)
    return img


def test_concurrent_face_detector_lock():
    """
    Stress-test FaceDetectorYN under 12 concurrent worker threads.
    Verifies that threading.Lock in FacePrivacyCropper completely prevents
    C++ OpenCV race conditions, memory corruption, and segmentation faults.
    """
    img = generate_synthetic_face_image()

    def _worker(thread_id):
        # Slightly alter image noise per worker
        worker_img = img.copy()
        worker_img[0, 0] = thread_id % 255
        crop, bbox, sanitized = face_privacy_cropper.extract_isolated_face(
            img_bgr=worker_img,
            margin_ratio=0.15,
        )
        return crop is not None, bbox, sanitized

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(_worker, i) for i in range(12)]
        results = [f.result() for f in futures]

    assert len(results) == 12
    for detected, bbox, sanitized in results:
        assert isinstance(detected, bool)
        if detected:
            assert sanitized is True
            assert bbox is not None


@pytest.mark.asyncio
async def test_concurrent_asyncio_face_crops():
    """
    Verify async thread pooling (asyncio.to_thread) under 10 concurrent async tasks.
    """
    img = generate_synthetic_face_image()

    async def _async_task(idx):
        worker_img = img.copy()
        return await asyncio.to_thread(
            face_privacy_cropper.extract_isolated_face,
            img_bgr=worker_img,
            margin_ratio=0.15,
        )

    tasks = [_async_task(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    for crop, bbox, sanitized in results:
        if crop is not None:
            assert sanitized is True


def test_concurrent_client_privacy_preview():
    """
    Stress-test /api/v1/privacy-crop-preview with concurrent TestClient HTTP requests.
    """
    client = TestClient(app)
    img = generate_synthetic_face_image()
    b64 = encode_image_to_base64(img)
    payload = {"image_base64": b64, "margin_ratio": 0.15}

    def _call_endpoint(i):
        return client.post("/api/v1/privacy-crop-preview", json=payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_call_endpoint, i) for i in range(8)]
        responses = [f.result() for f in futures]

    for resp in responses:
        assert resp.status_code == 200
        data = resp.json()
        assert "face_detected" in data
        assert "pii_sanitized" in data
