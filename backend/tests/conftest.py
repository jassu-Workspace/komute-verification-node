import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure the backend package root is importable regardless of invocation directory.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def client():
    headers = {}
    if settings.enable_api_key_auth and settings.api_secret_key:
        headers["X-API-Key"] = settings.api_secret_key
    return TestClient(app, headers=headers)


@pytest.fixture(autouse=True)
def mock_vlm_provider(request, monkeypatch):
    """
    Phase 6: guarantee no test performs live cloud VLM inference.

    Previously `test_full_verification_approved` issued a real cloud VLM request,
    so the suite was network-dependent, slow, and non-deterministic.

    Tests that need a specific verdict override `_mock_state`.
    Opt in to real calls with `@pytest.mark.live_vlm`.
    """
    if request.node.get_closest_marker("live_vlm"):
        yield
        return

    from app.biometric_schemas import BiometricVerdict
    from core import vlm_face_verifier as vfv
    from core.vlm_providers import ProviderResult, VLMProvider

    state = {
        "is_match": True,
        "confidence_score": 0.94,
        "is_live_selfie": True,
        "verdict": "MATCH_CONFIRMED",
        "configured": True,
    }

    class _StubProvider(VLMProvider):
        name = "mock"

        @property
        def model(self) -> str:
            return "mock"

        def is_configured(self) -> bool:
            return state["configured"]

        async def connectivity_check(self) -> None:
            return None

        async def _call_once(self, selfie_b64, dl_b64, repair_text=""):
            return state.get(
                "raw_text",
                '{"is_match": %s, "confidence_score": %s, "is_live_selfie": %s, '
                '"estimated_age_delta_years": 0, "facial_feature_notes": "mock", '
                '"reasoning": "mock provider", "verdict": "%s"}'
                % (
                    "true" if state["is_match"] else "false",
                    state["confidence_score"],
                    "true" if state["is_live_selfie"] else "false",
                    state["verdict"],
                ),
            )

    monkeypatch.setattr(vfv, "build_model_candidates", lambda: [_StubProvider()] if state["configured"] else [])

    class _Handle:
        def configure(self, **kwargs):
            state.update(kwargs)

        @property
        def state(self):
            return state

    yield _Handle()


@pytest.fixture(autouse=True)
def clear_telemetry_cache():
    """Telemetry now uses a 30s TTL; force recomputation between tests."""
    from core.telemetry_service import telemetry_service

    telemetry_service.invalidate_caches()
    yield
    telemetry_service.invalidate_caches()


@pytest.fixture
def detected_face_crops(monkeypatch):
    """
    Make the Zero-PII cropper treat the uploaded image as an already-isolated
    face crop.

    YuNet + the RetinaFace verification layer correctly reject the flat synthetic
    ellipses used in tests, which previously short-circuited Stage 2 to
    REJECTED_NO_FACE before the VLM was ever reached. This keeps end-to-end
    response-shape tests focused on the VLM/degraded-reporting contract rather
    than on face-detection accuracy.
    """
    from core import pipeline as pipeline_module
    from core.face_privacy_cropper import face_privacy_cropper

    def _fake_extract(img_bgr, margin_ratio=0.15):
        return img_bgr, (0, 0, img_bgr.shape[1], img_bgr.shape[0]), True

    monkeypatch.setattr(face_privacy_cropper, "extract_isolated_face", _fake_extract)
    yield
