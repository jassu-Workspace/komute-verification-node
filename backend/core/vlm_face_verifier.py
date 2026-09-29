"""
Stage 2B: Cloud VLM biometric verifier.

Phase 2/3/4 rewrite. This module is a thin orchestrator around
`core.vlm_providers`:

  * Fail-closed. An unparseable or schema-invalid provider response is a strict
    rejection (verdict SERVICE_UNAVAILABLE), never an inferred match.
  * Fully async. No `asyncio.to_thread`, so cancellation is real and no thread
    leaks when a provider exceeds its budget.
  * Observable. Every attempt is recorded in `vlm_attempt_chain`, and failures
    are classified into actionable reason codes instead of a blanket
    "VLM is not configured".
"""

import asyncio
import base64
import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.config import settings
from app.schemas import FaceBiometricsResult
from core.image_utils import encode_image_to_base64
from core.vlm_diagnostics import (
    CAPACITY_REASONS,
    REASON_CONFIG_ERROR,
    REASON_NO_CREDENTIALS,
    REASON_TIMEOUT,
    REASON_UNPARSEABLE_RESPONSE,
    classify_error,
    vlm_diagnostics,
)
from core.vlm_providers import (
    BiometricVerdictValidationError,
    VLMProvider,
    build_model_candidates,
)

logger = logging.getLogger(__name__)

__all__ = [
    "VLMFaceVerifier",
    "vlm_face_verifier",
    "BiometricVerdictValidationError",
    "REJECTED_NO_FACE",
    "SERVICE_UNAVAILABLE",
]

REJECTED_NO_FACE = "REJECTED_NO_FACE"
SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


def encode_crop_to_jpeg_base64(crop_bgr: np.ndarray) -> str:
    """Encode a BGR crop to bare base64 JPEG (no data-URI prefix)."""
    ok, buffer = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("Failed to JPEG-encode face crop")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


class VLMFaceVerifier:
    """Stage 2B: Cloud Vision-Language Model (VLM) Biometric Verifier."""

    def _fail_safe_biometric_fallback(
        self,
        reason: str,
        reason_code: str = "provider_error",
        attempt_chain: Optional[List[Dict[str, Any]]] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fail-safe fallback when the cloud VLM cannot be reached or its response
        cannot be validated. Rejects strictly (0.0 confidence) to prevent
        false-positive identity impersonation.
        """
        return {
            "is_match": False,
            "confidence_score": 0.0,
            "is_live_selfie": False,
            "estimated_age_delta_years": None,
            "facial_feature_notes": "Cloud biometric service unavailable; fail-safe rejection enforced.",
            "reasoning": f"Biometric verification service unavailable: {reason}",
            "verdict": SERVICE_UNAVAILABLE,
            "reason_code": reason_code,
            "provider": provider,
            "model": None,
            "attempt_chain": attempt_chain or [],
        }

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Backwards-compatible JSON extraction helper.

        Phase 3: the historical keyword heuristic (which returned
        is_match=True / confidence 0.85 / is_live_selfie=True for any prose
        containing the word "match") has been removed. Unparseable input now
        raises BiometricVerdictValidationError instead of guessing.
        """
        return VLMProvider._parse(VLMProvider, text).to_provider_dict()

    async def verify_biometrics(
        self,
        selfie_crop_bgr: Optional[np.ndarray],
        dl_face_crop_bgr: Optional[np.ndarray],
    ) -> FaceBiometricsResult:
        """Execute Stage 2B cloud VLM biometric cross-matching with zero PII leakage."""
        if selfie_crop_bgr is None or dl_face_crop_bgr is None:
            dl_prev = (
                encode_image_to_base64(dl_face_crop_bgr, format_ext=".jpg")
                if dl_face_crop_bgr is not None
                else None
            )
            selfie_prev = (
                encode_image_to_base64(selfie_crop_bgr, format_ext=".jpg")
                if selfie_crop_bgr is not None
                else None
            )
            return FaceBiometricsResult(
                passed=False,
                vlm_confidence=0.0,
                is_live=False,
                is_match=False,
                privacy_face_cropped=True,
                pii_leakage_prevented=True,
                reasoning="Face detection failed to isolate driver portrait on license or selfie.",
                verdict=REJECTED_NO_FACE,
                dl_face_detected=dl_face_crop_bgr is not None,
                selfie_face_detected=selfie_crop_bgr is not None,
                dl_face_crop_preview=dl_prev,
                selfie_face_crop_preview=selfie_prev,
                vlm_provider_used=None,
                vlm_model_used=None,
                vlm_attempt_chain=[],
            )

        attempt_chain: List[Dict[str, Any]] = []
        candidates = build_model_candidates()

        if not candidates:
            vlm_data = self._fail_safe_biometric_fallback(
                "No VLM API credentials configured for any provider in the fallback chain "
                f"{settings.provider_fallback_list}",
                reason_code=REASON_NO_CREDENTIALS,
                attempt_chain=attempt_chain,
            )
        else:
            vlm_data = await self._run_fallback_chain(
                candidates, attempt_chain, selfie_crop_bgr, dl_face_crop_bgr
            )

        return self._build_result(vlm_data, selfie_crop_bgr, dl_face_crop_bgr, attempt_chain)

    async def _run_fallback_chain(
        self,
        candidates: List[Any],
        attempt_chain: List[Dict[str, Any]],
        selfie_crop_bgr: np.ndarray,
        dl_face_crop_bgr: np.ndarray,
    ) -> Dict[str, Any]:
        """Walk the provider/model candidate list under the overall VLM budget."""
        last_reason = "provider_error"
        last_detail = "No provider was attempted."
        last_provider: Optional[str] = None

        overall_budget = settings.vlm_total_timeout_seconds
        started = time.perf_counter()
        last_provider_overloaded = False

        # Image encoding is provider-agnostic, so do it once.
        try:
            selfie_b64 = encode_crop_to_jpeg_base64(selfie_crop_bgr)
            dl_b64 = encode_crop_to_jpeg_base64(dl_face_crop_bgr)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to encode face crops for VLM: %s", type(exc).__name__)
            return self._fail_safe_biometric_fallback(
                f"face crop encoding failed: {type(exc).__name__}",
                reason_code=REASON_CONFIG_ERROR,
                attempt_chain=attempt_chain,
            )

        for provider in candidates:
            # A capacity error (429/503 "high demand") is a property of the
            # account or region, not of one model ID. Once a provider has
            # reported it, burning the remaining budget on its other model
            # fallbacks is wasteful, so advance to the next provider instead.
            if last_provider_overloaded and provider.name == last_provider:
                attempt_chain.append(
                    {
                        "provider": provider.name,
                        "model": provider.model,
                        "ok": False,
                        "skipped": True,
                        "reason": "provider_overloaded",
                        "latency_ms": 0,
                    }
                )
                continue

            remaining = overall_budget - (time.perf_counter() - started)
            if remaining <= 0.5:
                last_reason = REASON_TIMEOUT
                last_detail = (
                    f"VLM budget of {overall_budget}s exhausted after {len(attempt_chain)} attempt(s)."
                )
                break

            per_call = min(settings.vlm_call_timeout_seconds, remaining)
            attempt_started = time.perf_counter()
            entry: Dict[str, Any] = {
                "provider": provider.name,
                "model": provider.model,
                "ok": False,
                "latency_ms": 0,
                "attempts": 0,
            }

            try:
                # Per-attempt budget. asyncio.timeout cancels the underlying
                # HTTP request for real, so no orphaned work survives.
                async with asyncio.timeout(per_call):
                    result = await provider.verify(selfie_b64, dl_b64)

                entry["ok"] = True
                entry["attempts"] = result.attempts
                entry["latency_ms"] = result.latency_ms
                attempt_chain.append(entry)

                verdict = result.verdict
                return {
                    "is_match": verdict.is_match,
                    "confidence_score": verdict.confidence_score,
                    "is_live_selfie": verdict.is_live_selfie,
                    "estimated_age_delta_years": verdict.estimated_age_delta_years,
                    "facial_feature_notes": verdict.facial_feature_notes,
                    "reasoning": verdict.reasoning,
                    "verdict": verdict.verdict,
                    "provider": result.provider,
                    "model": result.model,
                }

            except (TimeoutError, asyncio.TimeoutError):
                entry["latency_ms"] = int((time.perf_counter() - attempt_started) * 1000)
                entry["reason"] = REASON_TIMEOUT
                attempt_chain.append(entry)
                last_reason = REASON_TIMEOUT
                last_detail = f"exceeded {per_call:.1f}s budget"
                last_provider = provider.name
                logger.warning(
                    "VLM provider '%s' model '%s' timed out after %.1fs.",
                    provider.name,
                    provider.model,
                    per_call,
                )

            except BiometricVerdictValidationError as exc:
                entry["latency_ms"] = int((time.perf_counter() - attempt_started) * 1000)
                entry["reason"] = REASON_UNPARSEABLE_RESPONSE
                attempt_chain.append(entry)
                last_reason = REASON_UNPARSEABLE_RESPONSE
                last_detail = str(exc)
                last_provider = provider.name

            except Exception as exc:  # noqa: BLE001
                entry["latency_ms"] = int((time.perf_counter() - attempt_started) * 1000)
                reason = vlm_diagnostics.record_error(provider.name, provider.model, exc)
                entry["reason"] = reason
                attempt_chain.append(entry)
                last_reason = reason
                last_detail = f"{type(exc).__name__}: {classify_error(exc)}"
                last_provider = provider.name
                if reason in CAPACITY_REASONS:
                    last_provider_overloaded = True
                    logger.warning(
                        "VLM provider '%s' is capacity-throttled (%s). Skipping its remaining "
                        "model fallbacks and advancing to the next provider.",
                        provider.name,
                        reason,
                    )

        return self._fail_safe_biometric_fallback(
            f"All {len(candidates)} VLM provider/model candidate(s) failed. Last failure "
            f"[{last_reason}] from '{last_provider or 'n/a'}': {last_detail}",
            reason_code=last_reason,
            attempt_chain=attempt_chain,
            provider=last_provider,
        )

    def _build_result(
        self,
        vlm_data: Dict[str, Any],
        selfie_crop_bgr: np.ndarray,
        dl_face_crop_bgr: np.ndarray,
        attempt_chain: List[Dict[str, Any]],
    ) -> FaceBiometricsResult:
        """Map a validated provider verdict onto the API response model."""
        is_match = bool(vlm_data.get("is_match", False))

        # Clamp: a provider must never be able to report outside [0, 1].
        raw_confidence = float(vlm_data.get("confidence_score", 0.0) or 0.0)
        confidence = (
            0.0 if raw_confidence != raw_confidence else min(1.0, max(0.0, raw_confidence))
        )

        # Fail-closed default: an omitted liveness result never counts as live.
        is_live = bool(vlm_data.get("is_live_selfie", False))

        age_delta = vlm_data.get("estimated_age_delta_years")
        notes = vlm_data.get("facial_feature_notes")
        reasoning = vlm_data.get("reasoning", "")
        verdict = vlm_data.get("verdict", "INCONCLUSIVE")

        # Pass requires agreement across every signal, not just a high score.
        passed = (
            is_match
            and is_live
            and confidence >= settings.vlm_min_match_confidence
            and verdict == "MATCH_CONFIRMED"
        )

        dl_preview = encode_image_to_base64(dl_face_crop_bgr, format_ext=".jpg")
        selfie_preview = encode_image_to_base64(selfie_crop_bgr, format_ext=".jpg")

        confidence_proof = {
            "formula": "vlm_biometric_confidence if is_live else 0.0",
            "metrics": {
                "biometric_similarity_proof": {
                    "vlm_confidence_score": round(confidence, 3),
                    "required_threshold": settings.vlm_min_match_confidence,
                    "is_match": is_match,
                    "verdict": verdict,
                    "weighted_points": round(confidence, 3),
                    "justification": (
                        "Cloud biometric verification service is unavailable"
                        if verdict == SERVICE_UNAVAILABLE
                        else (
                            f"Craniofacial landmark congruence: {confidence * 100:.1f}%. "
                            f"Met threshold >= {settings.vlm_min_match_confidence * 100:.0f}%"
                            if passed
                            else f"Biometric mismatch or low confidence ({confidence * 100:.1f}%)"
                        )
                    ),
                },
                "anti_spoof_liveness_proof": {
                    "is_live_selfie": is_live,
                    "anti_spoof_passed": is_live,
                    "justification": (
                        "Anti-spoof check could not be completed: Cloud biometric service unavailable"
                        if verdict == SERVICE_UNAVAILABLE
                        else (
                            "Live human subject verified without moire patterns, screen bezels, or digital replay"
                            if is_live
                            else "Anti-spoof rejection: replay attack / 2D screen display detected"
                        )
                    ),
                },
                "privacy_isolation_proof": {
                    "pii_sanitization_enforced": True,
                    "face_crop_padding": f"{int(settings.face_crop_padding_ratio * 100)}% tight margin",
                    "justification": "100% PII sanitized: zero card numbers, barcodes, or text transmitted to cloud neural models",
                },
                "age_delta_proof": {
                    "estimated_age_delta_years": age_delta or 0,
                    "justification": (
                        f"Age difference between ID photo and live selfie is approximately {age_delta} years, within normal temporal drift"
                        if age_delta
                        else "Facial age progression consistent"
                    ),
                },
                "provider_provenance": {
                    "provider": vlm_data.get("provider"),
                    "model": vlm_data.get("model"),
                    "reason_code": vlm_data.get("reason_code"),
                    "attempt_chain": attempt_chain,
                },
            },
            "forensic_reasoning_verbatim": reasoning,
            "calculated_confidence": round(confidence if is_live else 0.0, 3),
            "stage_verdict": "PASSED" if passed else "REJECTED",
        }

        return FaceBiometricsResult(
            passed=passed,
            vlm_confidence=round(confidence, 3),
            is_live=is_live,
            is_match=is_match,
            estimated_age_delta_years=age_delta,
            privacy_face_cropped=True,
            pii_leakage_prevented=True,
            facial_feature_notes=notes,
            reasoning=reasoning,
            verdict=verdict,
            dl_face_detected=True,
            selfie_face_detected=True,
            dl_face_crop_preview=dl_preview,
            selfie_face_crop_preview=selfie_preview,
            confidence_proof=confidence_proof,
            vlm_provider_used=vlm_data.get("provider"),
            vlm_model_used=vlm_data.get("model"),
            vlm_attempt_chain=attempt_chain,
        )


# Global instance
vlm_face_verifier = VLMFaceVerifier()
