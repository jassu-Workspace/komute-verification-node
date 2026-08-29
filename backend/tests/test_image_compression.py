import io
import cv2
import numpy as np
import pytest
from PIL import Image
from core.image_utils import (
    compress_and_normalize_base64,
    compress_image_bytes,
    encode_image_to_base64,
)


def create_test_rgb_image(width=1600, height=1200):
    """Generate a test high-resolution RGB image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (800, 800), (0, 120, 255), -1)
    cv2.putText(img, "TEST IMAGE COMPRESSION", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return img


def test_compress_image_bytes_webp():
    """Verify raw byte compression to WebP, downsizing and size reduction."""
    img_bgr = create_test_rgb_image(1800, 1400)
    _, encoded = cv2.imencode(".png", img_bgr)
    raw_bytes = encoded.tobytes()

    compressed_bytes, mime_type, orig_sz, comp_sz = compress_image_bytes(
        input_bytes=raw_bytes,
        out_format="webp",
        max_dim=1000,
        quality=80,
    )

    assert mime_type == "image/webp"
    assert comp_sz < orig_sz
    assert len(compressed_bytes) == comp_sz

    # Verify decompressed dimension <= 1000
    pil_img = Image.open(io.BytesIO(compressed_bytes))
    assert max(pil_img.size) <= 1000


def test_compress_and_normalize_base64():
    """Verify base64 in-flight conversion to normalized WebP."""
    img_bgr = create_test_rgb_image(800, 600)
    raw_b64 = encode_image_to_base64(img_bgr, format_ext=".png")

    bgr_out, data_uri_out = compress_and_normalize_base64(
        base64_data=raw_b64,
        out_format="webp",
        max_dim=600,
        quality=85,
    )

    assert bgr_out is not None
    assert bgr_out.shape[0] <= 600 and bgr_out.shape[1] <= 600
    assert data_uri_out.startswith("data:image/webp;base64,")


def test_compress_endpoint(client):
    """Verify the /api/v1/compress endpoint accepts multipart uploads."""
    img_bgr = create_test_rgb_image(600, 400)
    _, encoded = cv2.imencode(".jpg", img_bgr)

    files = {"file": ("test_upload.jpg", encoded.tobytes(), "image/jpeg")}
    data = {"compressionPercentage": "70", "format": "webp", "maxWidth": "500"}

    response = client.post("/api/v1/compress", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert "X-Saved-Bytes" in response.headers
