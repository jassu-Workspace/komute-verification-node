import cv2
import numpy as np
import pytest
from core.vlm_face_verifier import VLMFaceVerifier, vlm_face_verifier


def test_vlm_json_extractor():
    """Verify robust JSON extraction from LLM/VLM text outputs."""
    verifier = VLMFaceVerifier()

    raw_json_str = '{"is_match": true, "confidence_score": 0.95, "is_live_selfie": true, "verdict": "MATCH_CONFIRMED"}'
    parsed1 = verifier._extract_json_from_text(raw_json_str)
    assert parsed1["is_match"] is True
    assert parsed1["confidence_score"] == 0.95

    # Markdown fence
    markdown_str = """Here is the biometric verification analysis:
```json
{
  "is_match": true,
  "confidence_score": 0.94,
  "is_live_selfie": true,
  "estimated_age_delta_years": 3,
  "facial_feature_notes": "Identical nasal bridge.",
  "reasoning": "High confidence match.",
  "verdict": "MATCH_CONFIRMED"
}
```
Thank you."""
    parsed2 = verifier._extract_json_from_text(markdown_str)
    assert parsed2["is_match"] is True
    assert parsed2["confidence_score"] == 0.94
    assert parsed2["estimated_age_delta_years"] == 3


@pytest.mark.asyncio
async def test_synthetic_biometric_verification():
    """Verify fail-safe biometric handling when cloud VLM provider is unconfigured."""
    result_dict = vlm_face_verifier._fail_safe_biometric_fallback("Offline test")
    assert result_dict["is_match"] is False
    assert result_dict["confidence_score"] == 0.0
    assert result_dict["verdict"] == "SERVICE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_missing_face_handling():
    """Verify handling when face is missing from one or both inputs."""
    face1 = np.full((200, 200, 3), 150, dtype=np.uint8)
    result = await vlm_face_verifier.verify_biometrics(face1, None)
    assert result.passed is False
    assert result.is_match is False
    assert result.dl_face_detected is False
