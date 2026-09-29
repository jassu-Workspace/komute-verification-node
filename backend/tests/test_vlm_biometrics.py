"""
Phase 3/6 regression tests for the cloud VLM biometric verifier.

These lock in the fail-closed behaviour. The most important test is
`test_prose_containing_match_is_not_a_pass` — a regression guard against the
historical authentication bypass where any non-JSON reply containing the word
"match" produced is_match=True / confidence 0.85 / is_live=True and therefore a
false APPROVED decision.
"""

import asyncio
import json

import cv2
import numpy as np
import pytest

from app.biometric_schemas import BiometricVerdict
from app.config import settings
from core.vlm_providers import (
    BiometricVerdictValidationError,
    OpenAICompatibleProvider,
    VLMProvider,
    build_model_candidates,
    build_provider,
)
from core.vlm_face_verifier import (
    REJECTED_NO_FACE,
    SERVICE_UNAVAILABLE,
    VLMFaceVerifier,
    vlm_face_verifier,
)


def _parse(text: str) -> dict:
    """Call the shared fail-closed parser."""
    return VLMProvider._parse(VLMProvider, text).to_provider_dict()


# ---------------------------------------------------------------------------
# Fail-closed parsing
# ---------------------------------------------------------------------------


def test_valid_json_verdict_parses():
    parsed = _parse(
        '{"is_match": true, "confidence_score": 0.95, "is_live_selfie": true, '
        '"facial_feature_notes": "x", "reasoning": "y", "verdict": "MATCH_CONFIRMED"}'
    )
    assert parsed["is_match"] is True
    assert parsed["confidence_score"] == 0.95
    assert parsed["verdict"] == "MATCH_CONFIRMED"


def test_markdown_fenced_json_parses():
    parsed = _parse(
        'Here you go:\n```json\n{"is_match": true, "confidence_score": 0.9, '
        '"is_live_selfie": true, "verdict": "MATCH_CONFIRMED"}\n```\nThanks!'
    )
    assert parsed["is_match"] is True
    assert parsed["confidence_score"] == 0.9


@pytest.mark.parametrize(
    "prose",
    [
        "I cannot determine if the images match.",
        "The faces do not match at all.",
        "Sorry, I am unable to compare these images.",
        "",
        "   ",
    ],
)
def test_prose_is_never_parsed_as_a_match(prose):
    """REGRESSION LOCK: prose must never yield is_match=True."""
    with pytest.raises(BiometricVerdictValidationError):
        _parse(prose)


def test_prose_containing_match_does_not_produce_a_pass():
    """The exact historical bypass: the word 'match' in prose."""
    with pytest.raises(BiometricVerdictValidationError):
        _parse("I cannot determine if the two faces match.")

    # And via the legacy helper surface, which must also raise.
    with pytest.raises(BiometricVerdictValidationError):
        vlm_face_verifier._extract_json_from_text("I cannot determine if the two faces match.")


def test_json_array_response_is_rejected():
    with pytest.raises(BiometricVerdictValidationError):
        _parse("[1, 2, 3]")


def test_out_of_range_confidence_is_rejected_by_schema():
    with pytest.raises(BiometricVerdictValidationError):
        _parse(
            '{"is_match": true, "confidence_score": 1.4, "is_live_selfie": true, '
            '"verdict": "MATCH_CONFIRMED"}'
        )


def test_nan_confidence_is_rejected():
    with pytest.raises(BiometricVerdictValidationError):
        _parse(
            '{"is_match": true, "confidence_score": NaN, "is_live_selfie": true, '
            '"verdict": "MATCH_CONFIRMED"}'
        )


def test_verdict_label_aliases_are_normalised():
    assert _parse(
        '{"is_match": true, "confidence_score": 0.9, "is_live_selfie": true, "verdict": "MATCH"}'
    )["verdict"] == "MATCH_CONFIRMED"
    assert _parse(
        '{"is_match": false, "confidence_score": 0.1, "is_live_selfie": true, "verdict": "NO_MATCH"}'
    )["verdict"] == "MISMATCH"


def test_omitted_verdict_label_cannot_smuggle_a_weak_pass():
    """An omitted label with a sub-threshold score must NOT become MATCH_CONFIRMED."""
    parsed = _parse(
        '{"is_match": true, "confidence_score": 0.55, "is_live_selfie": true}'
    )
    assert parsed["verdict"] == "MISMATCH"


def test_omitted_liveness_defaults_to_false():
    """Fail-closed: an omitted liveness result must never count as live."""
    parsed = _parse(
        '{"is_match": true, "confidence_score": 0.99, "verdict": "MATCH_CONFIRMED"}'
    )
    assert parsed["is_live_selfie"] is False


def test_gateway_style_keys_are_aliased_fail_closed():
    """Small gateway models may answer with their own key names.

    Observed live from openrouter/stealth/space-bunny-alpha:
    {"verdict":"INCONCLUSIVE","match_confidence":0.0,"is_live_selfie":null,
     "reason":"..."} — must validate as a rejection, not raise.
    """
    parsed = _parse(
        '{"verdict": "INCONCLUSIVE", "match_confidence": 0.0, '
        '"is_live_selfie": null, "reason": "pixels unavailable"}'
    )
    assert parsed["verdict"] == "INCONCLUSIVE"
    assert parsed["is_match"] is False
    assert parsed["confidence_score"] == 0.0
    assert parsed["is_live_selfie"] is False
    assert parsed["reasoning"] == "pixels unavailable"


def test_null_and_garbage_fields_cannot_smuggle_a_pass():
    parsed = _parse(
        '{"is_match": null, "confidence_score": "high", "is_live_selfie": null, '
        '"verdict": "MATCH_CONFIRMED"}'
    )
    assert parsed["is_match"] is False
    assert parsed["confidence_score"] == 0.0
    assert parsed["is_live_selfie"] is False


def test_gemini_format_alias_normalization():
    """Verify models using 'confidence' alias and omitting 'is_match' normalize correctly on MATCH_CONFIRMED."""
    parsed = _parse(
        '{"verdict": "MATCH_CONFIRMED", "confidence": 0.98, "is_live_selfie": true, '
        '"reasoning": "Identical craniofacial landmarks."}'
    )
    assert parsed["verdict"] == "MATCH_CONFIRMED"
    assert parsed["is_match"] is True
    assert parsed["confidence_score"] == 0.98
    assert parsed["is_live_selfie"] is True
    assert "Identical" in parsed["reasoning"]


def test_percentage_string_confidence():
    """Verify percentage strings like '95%' parse properly into floats."""
    parsed = _parse(
        '{"verdict": "MATCH_CONFIRMED", "confidence": "95%", "is_live_selfie": true}'
    )
    assert parsed["confidence_score"] == 0.95
    assert parsed["is_match"] is True


# ---------------------------------------------------------------------------
# Field hardening in the result mapper
# ---------------------------------------------------------------------------


def _result_from(verdict: BiometricVerdict) -> "object":
    verifier = VLMFaceVerifier()
    img = np.full((40, 40, 3), 120, dtype=np.uint8)
    vlm_data = verdict.to_provider_dict()
    vlm_data.update({"provider": "mock", "model": "mock"})
    return verifier._build_result(vlm_data, img, img, [])


def test_confidence_is_clamped_to_one():
    result = _result_from(
        BiometricVerdict(
            is_match=True,
            confidence_score=0.99,
            is_live_selfie=True,
            verdict="MATCH_CONFIRMED",
        )
    )
    assert 0.0 <= result.vlm_confidence <= 1.0


def test_pass_requires_verdict_agreement():
    """is_match=True with a MISMATCH verdict must not pass."""
    result = _result_from(
        BiometricVerdict(
            is_match=True,
            confidence_score=0.95,
            is_live_selfie=True,
            verdict="MISMATCH",
        )
    )
    assert result.passed is False
    assert result.is_match is True
    assert result.verdict == "MISMATCH"


def test_pass_requires_liveness():
    result = _result_from(
        BiometricVerdict(
            is_match=True,
            confidence_score=0.95,
            is_live_selfie=False,
            verdict="MATCH_CONFIRMED",
        )
    )
    assert result.passed is False


def test_pass_requires_threshold():
    result = _result_from(
        BiometricVerdict(
            is_match=True,
            confidence_score=settings.vlm_min_match_confidence - 0.05,
            is_live_selfie=True,
            verdict="MATCH_CONFIRMED",
        )
    )
    assert result.passed is False


def test_strong_match_passes():
    result = _result_from(
        BiometricVerdict(
            is_match=True,
            confidence_score=0.94,
            is_live_selfie=True,
            verdict="MATCH_CONFIRMED",
        )
    )
    assert result.passed is True
    assert result.vlm_confidence == 0.94


# ---------------------------------------------------------------------------
# Fail-safe fallback + face handling
# ---------------------------------------------------------------------------


def test_fail_safe_fallback_is_strict():
    result = vlm_face_verifier._fail_safe_biometric_fallback("Offline test")
    assert result["is_match"] is False
    assert result["confidence_score"] == 0.0
    assert result["is_live_selfie"] is False
    assert result["verdict"] == SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_missing_face_handling():
    face = np.full((200, 200, 3), 150, dtype=np.uint8)
    result = await vlm_face_verifier.verify_biometrics(face, None)
    assert result.passed is False
    assert result.is_match is False
    assert result.dl_face_detected is False
    assert result.verdict == REJECTED_NO_FACE


@pytest.mark.asyncio
async def test_no_credentials_yields_service_unavailable(mock_vlm_provider):
    """No provider configured -> strict fail-safe, not a match."""
    mock_vlm_provider.configure(configured=False)

    selfie = np.full((120, 120, 3), 140, dtype=np.uint8)
    result = await vlm_face_verifier.verify_biometrics(selfie, selfie)

    assert result.verdict == SERVICE_UNAVAILABLE
    assert result.passed is False
    assert result.is_match is False
    assert result.vlm_confidence == 0.0
    assert result.is_live is False
    assert "No VLM API credentials configured" in result.reasoning


@pytest.mark.asyncio
async def test_unparseable_provider_output_fails_closed(mock_vlm_provider):
    """A provider that returns prose must degrade to SERVICE_UNAVAILABLE."""
    mock_vlm_provider.configure(
        raw_text="I am not able to determine whether these faces match."
    )
    mock_vlm_provider.state["configured"] = True

    selfie = np.full((120, 120, 3), 140, dtype=np.uint8)
    result = await vlm_face_verifier.verify_biometrics(selfie, selfie)

    assert result.verdict == SERVICE_UNAVAILABLE
    assert result.passed is False
    assert result.is_match is False
    assert any(
        entry.get("reason") == "unparseable_response" for entry in result.vlm_attempt_chain
    )


@pytest.mark.asyncio
async def test_mock_provider_success_populates_provenance(mock_vlm_provider):
    selfie = np.full((120, 120, 3), 140, dtype=np.uint8)
    result = await vlm_face_verifier.verify_biometrics(selfie, selfie)

    assert result.verdict == "MATCH_CONFIRMED"
    assert result.passed is True
    assert result.vlm_provider_used == "mock"
    assert result.vlm_model_used == "mock"
    assert result.vlm_attempt_chain and result.vlm_attempt_chain[0]["ok"] is True


# ---------------------------------------------------------------------------
# Provider fallback chain (Phase 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_falls_back_to_next_provider_on_failure():
    """Provider order must be honoured and a failure must advance the chain."""
    order = []

    class _Failing(VLMProvider):
        name = "openai_compatible"

        @property
        def model(self):
            return "gateway-primary"

        def is_configured(self):
            return True

        async def connectivity_check(self):
            return None

        async def _call_once(self, *a, **k):
            order.append("openai_compatible")
            raise RuntimeError("simulated provider outage")

    class _Succeeding(VLMProvider):
        name = "anthropic"

        @property
        def model(self):
            return "claude-sonnet-5"

        def is_configured(self):
            return True

        async def connectivity_check(self):
            return None

        async def _call_once(self, *a, **k):
            order.append("anthropic")
            return json.dumps(
                {
                    "is_match": True,
                    "confidence_score": 0.91,
                    "is_live_selfie": True,
                    "verdict": "MATCH_CONFIRMED",
                }
            )

    from core import vlm_face_verifier as vfv

    original = vfv.build_model_candidates
    vfv.build_model_candidates = lambda: [_Failing(), _Succeeding()]
    try:
        selfie = np.full((120, 120, 3), 140, dtype=np.uint8)
        result = await vfv.vlm_face_verifier.verify_biometrics(selfie, selfie)
    finally:
        vfv.build_model_candidates = original

    assert order == ["openai_compatible", "anthropic"]
    assert result.passed is True
    assert result.vlm_provider_used == "anthropic"
    assert len(result.vlm_attempt_chain) == 2
    assert result.vlm_attempt_chain[0]["ok"] is False
    assert result.vlm_attempt_chain[1]["ok"] is True


@pytest.mark.asyncio
async def test_all_providers_failing_yields_service_unavailable():
    class _AlwaysFails(VLMProvider):
        name = "openai_compatible"

        @property
        def model(self):
            return "gateway-primary"

        def is_configured(self):
            return True

        async def connectivity_check(self):
            return None

        async def _call_once(self, *a, **k):
            raise TimeoutError("simulated timeout")

    from core import vlm_face_verifier as vfv

    original = vfv.build_model_candidates
    vfv.build_model_candidates = lambda: [_AlwaysFails()]
    try:
        selfie = np.full((120, 120, 3), 140, dtype=np.uint8)
        result = await vfv.vlm_face_verifier.verify_biometrics(selfie, selfie)
    finally:
        vfv.build_model_candidates = original

    assert result.verdict == SERVICE_UNAVAILABLE
    assert result.passed is False
    assert result.vlm_attempt_chain[0]["reason"] == "timeout"


def test_gateway_contributes_one_candidate_per_model_fallback():
    """A bad model ID must be skipped individually, not poison the provider."""
    original = settings.vlm_model_fallbacks
    original_key = settings.vlm_api_key
    original_base = settings.vlm_api_base_url
    original_model = settings.vlm_openai_model
    try:
        settings.vlm_api_key = "sk-stub-for-tests-only"
        settings.vlm_api_base_url = "https://our-llm.onrender.com/v1"
        settings.vlm_openai_model = "primary-model"
        settings.vlm_model_fallbacks = "primary-model,secondary-model"
        candidates = build_model_candidates()
    finally:
        settings.vlm_model_fallbacks = original
        settings.vlm_api_key = original_key
        settings.vlm_api_base_url = original_base
        settings.vlm_openai_model = original_model

    gateway_models = [
        c.model for c in candidates if isinstance(c, OpenAICompatibleProvider)
    ]
    assert gateway_models == ["primary-model", "secondary-model"]


def test_anthropic_uses_a_live_model_id():
    """claude-3-5-sonnet-20241022 was retired in Oct 2025 and can never work."""
    assert "claude-3-5-sonnet" not in settings.anthropic_model
    assert settings.anthropic_model == "claude-sonnet-5"


def test_provider_registry_covers_all_configured_values():
    for name in ("openai_compatible", "anthropic", "mock"):
        assert build_provider(name) is not None


def test_mock_provider_refuses_production():
    from core.vlm_providers import MockProvider

    original = settings.environment
    try:
        settings.environment = "production"
        with pytest.raises(RuntimeError):
            MockProvider().is_configured()
    finally:
        settings.environment = original


# ---------------------------------------------------------------------------
# Failure classification (Phase 0/1)
# ---------------------------------------------------------------------------


class _FakeGoogleError(Exception):
    """Mirrors google.genai.errors.ServerError, which is what 503 surfaces as."""

    def __init__(self, message, code=None, status=None):
        super().__init__(message)
        self.code = code
        self.status = status


@pytest.mark.parametrize(
    "exc,expected",
    [
        (
            _FakeGoogleError(
                "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is "
                "currently experiencing high demand. Spikes in demand are usually temporary.'}}",
                code=503,
                status="UNAVAILABLE",
            ),
            "model_overloaded",
        ),
        (_FakeGoogleError("429 Too Many Requests: rate limit reached", code=429), "model_overloaded"),
        (_FakeGoogleError("RESOURCE_EXHAUSTED: You exceeded your current quota", code=429), "quota_exhausted"),
        (_FakeGoogleError("API key not valid. Please pass a valid API key.", code=400), "invalid_credentials"),
        (_FakeGoogleError("PERMISSION_DENIED", code=403, status="PERMISSION_DENIED"), "invalid_credentials"),
        (_FakeGoogleError("models/xyz-1 is not found for API version v1", code=404), "model_unavailable"),
        (_FakeGoogleError("Connection refused", code=None), "provider_unreachable"),
        (TimeoutError("deadline exceeded"), "timeout"),
        (ImportError("No module named 'google.genai'"), "sdk_missing"),
        (ValueError("VLM_API_KEY is not configured"), "config_error"),
    ],
)
def test_classify_error_maps_to_actionable_reasons(exc, expected):
    from core.vlm_diagnostics import classify_error

    assert classify_error(exc) == expected


def test_capacity_errors_are_flagged_as_retryable_elsewhere():
    from core.vlm_diagnostics import CAPACITY_REASONS, REASON_MODEL_OVERLOADED

    assert REASON_MODEL_OVERLOADED in CAPACITY_REASONS


@pytest.mark.asyncio
async def test_overloaded_provider_skips_its_remaining_model_fallbacks():
    """
    A 503/429 is a property of the account or region, not of one model ID, so
    the remaining model fallbacks of that provider must be skipped rather than
    consuming the whole VLM budget.
    """
    attempted: list = []

    class _Overloaded(VLMProvider):
        name = "openai_compatible"

        def __init__(self, model):
            self._model = model

        @property
        def model(self):
            return self._model

        def is_configured(self):
            return True

        async def connectivity_check(self):
            return None

        async def _call_once(self, *a, **k):
            attempted.append(self._model)
            raise _FakeGoogleError(
                "503 UNAVAILABLE. This model is currently experiencing high demand.",
                code=503,
                status="UNAVAILABLE",
            )

    class _Backup(VLMProvider):
        name = "anthropic"

        @property
        def model(self):
            return "claude-sonnet-5"

        def is_configured(self):
            return True

        async def connectivity_check(self):
            return None

        async def _call_once(self, *a, **k):
            return json.dumps(
                {
                    "is_match": True,
                    "confidence_score": 0.92,
                    "is_live_selfie": True,
                    "verdict": "MATCH_CONFIRMED",
                }
            )

    from core import vlm_face_verifier as vfv

    original = vfv.build_model_candidates
    vfv.build_model_candidates = lambda: [
_Overloaded("gateway-primary"),
        _Overloaded("gateway-secondary"),
        _Backup(),
    ]
    try:
        selfie = np.full((120, 120, 3), 140, dtype=np.uint8)
        result = await vfv.vlm_face_verifier.verify_biometrics(selfie, selfie)
    finally:
        vfv.build_model_candidates = original

# Only the FIRST gateway model was actually called; the second was skipped.
    assert attempted == ["gateway-primary"]
    assert any(a.get("skipped") for a in result.vlm_attempt_chain)
    # Failover still succeeded via the backup provider.
    assert result.passed is True
    assert result.vlm_provider_used == "anthropic"
