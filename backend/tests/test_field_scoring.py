import pytest
from app.config import Settings, settings
from app.schemas import (
    DecisionEnum,
    FaceBiometricsResult,
    LicenseOcrResult,
    VehicleVerificationResult,
)
from core.pipeline import VerificationPipeline
from core.vlm_providers import VLMProvider, OpenAICompatibleProvider


def test_field_average_scoring_approved():
    pipeline = VerificationPipeline()

    s1 = LicenseOcrResult(
        passed=True,
        confidence=1.0,
        number_matched=True,
        number_similarity=1.0,
        name_matched=True,
        name_similarity=1.0,
        dob_matched=True,
        dob_similarity=1.0,
        expiry_similarity=1.0,
    )
    s2 = FaceBiometricsResult(
        passed=True,
        vlm_confidence=0.90,
        is_live=True,
        is_match=True,
        dl_face_detected=True,
        selfie_face_detected=True,
        verdict="MATCH_CONFIRMED",
    )
    s3 = VehicleVerificationResult(
        passed=True,
        confidence=0.85,
        plate_matched=True,
        plate_similarity=0.85,
        color_matched=True,
        color_confidence=0.90,
    )

    decision, score, rejections, manual, proof, degraded = pipeline._evaluate_composite_decision(s1, s2, s3)
    assert decision == DecisionEnum.APPROVED
    assert score >= settings.composite_approval_threshold
    assert proof["scoring_mode"] == "field_average"
    assert len(proof["field_scores"]) == 7


def test_field_average_scoring_approved_at_70_percent_boundary():
    """Verify that a composite score around ~72% (between 70% and 80%) is APPROVED under the 70% threshold."""
    pipeline = VerificationPipeline()

    # 4 fields at 0.8, 3 fields at 0.65 -> average = (3.2 + 1.95) / 7 = 5.15 / 7 = ~0.736 (> 0.70)
    s1 = LicenseOcrResult(
        passed=True,
        confidence=0.80,
        number_matched=True,
        number_similarity=0.80,
        name_matched=True,
        name_similarity=0.80,
        dob_matched=True,
        dob_similarity=0.80,
        expiry_similarity=0.80,
    )
    s2 = FaceBiometricsResult(
        passed=True,
        vlm_confidence=0.65,
        is_live=True,
        is_match=True,
        dl_face_detected=True,
        selfie_face_detected=True,
        verdict="MATCH_CONFIRMED",
    )
    s3 = VehicleVerificationResult(
        passed=True,
        confidence=0.65,
        plate_matched=True,
        plate_similarity=0.65,
        color_matched=True,
        color_confidence=0.65,
    )

    decision, score, rejections, manual, proof, degraded = pipeline._evaluate_composite_decision(s1, s2, s3)
    assert 0.70 <= score < 0.80
    assert decision == DecisionEnum.APPROVED
    assert proof["approval_threshold"] == 0.70
    assert len(rejections) == 0


def test_field_average_scoring_rejected():
    pipeline = VerificationPipeline()

    s1 = LicenseOcrResult(
        passed=False,
        confidence=0.30,
        number_matched=False,
        number_similarity=0.30,
        name_matched=False,
        name_similarity=0.20,
        dob_matched=False,
        dob_similarity=0.0,
        expiry_similarity=0.50,
    )
    s2 = FaceBiometricsResult(
        passed=False,
        vlm_confidence=0.20,
        is_live=True,
        is_match=False,
        dl_face_detected=True,
        selfie_face_detected=True,
        verdict="MISMATCH",
    )
    s3 = VehicleVerificationResult(
        passed=False,
        confidence=0.20,
        plate_matched=False,
        plate_similarity=0.20,
        color_matched=False,
        color_confidence=0.10,
    )

    decision, score, rejections, manual, proof, degraded = pipeline._evaluate_composite_decision(s1, s2, s3)
    assert decision == DecisionEnum.REJECTED
    assert score < settings.composite_approval_threshold
    assert len(rejections) > 0


def test_field_weights_customization(monkeypatch):
    pipeline = VerificationPipeline()
    monkeypatch.setattr(settings, "field_weight_face", 10.0)
    monkeypatch.setattr(settings, "field_weight_name", 1.0)
    monkeypatch.setattr(settings, "field_weight_dl_number", 1.0)
    monkeypatch.setattr(settings, "field_weight_dob", 1.0)
    monkeypatch.setattr(settings, "field_weight_expiry", 1.0)
    monkeypatch.setattr(settings, "field_weight_plate", 1.0)
    monkeypatch.setattr(settings, "field_weight_color", 1.0)

    # Face has low score, but high weight
    s1 = LicenseOcrResult(passed=True, confidence=1.0, number_similarity=1.0, name_similarity=1.0, dob_similarity=1.0, expiry_similarity=1.0)
    s2 = FaceBiometricsResult(passed=False, vlm_confidence=0.10, is_live=True, is_match=False, dl_face_detected=True, selfie_face_detected=True, verdict="MISMATCH")
    s3 = VehicleVerificationResult(passed=True, confidence=1.0, plate_similarity=1.0, color_matched=True, color_confidence=1.0)

    decision, score, rejections, manual, proof, degraded = pipeline._evaluate_composite_decision(s1, s2, s3)
    # 6 fields at 1.0 * weight 1.0 = 6.0 points. Face at 0.1 * weight 10.0 = 1.0 point.
    # Total points = 7.0, total weight = 16.0 -> 7/16 = 0.4375 (< 0.80) -> REJECTED
    assert score < 0.50
    assert decision == DecisionEnum.REJECTED


def test_stage_weighted_mode(monkeypatch):
    pipeline = VerificationPipeline()
    monkeypatch.setattr(settings, "scoring_mode", "stage_weighted")
    monkeypatch.setattr(settings, "stage1_weight", 0.40)
    monkeypatch.setattr(settings, "stage2_weight", 0.40)
    monkeypatch.setattr(settings, "stage3_weight", 0.20)

    s1 = LicenseOcrResult(passed=True, confidence=0.90, number_similarity=0.90, name_similarity=0.90)
    s2 = FaceBiometricsResult(passed=True, vlm_confidence=0.90, is_live=True, is_match=True, dl_face_detected=True, selfie_face_detected=True, verdict="MATCH_CONFIRMED")
    s3 = VehicleVerificationResult(passed=True, confidence=0.90, plate_matched=True, plate_similarity=0.90, color_matched=True, color_confidence=0.90)

    decision, score, rejections, manual, proof, degraded = pipeline._evaluate_composite_decision(s1, s2, s3)
    assert decision == DecisionEnum.APPROVED
    assert proof["scoring_mode"] == "stage_weighted"


def test_vlm_liveness_alias_handling():
    raw_response = '{"match": true, "confidence": 0.95, "verdict": "MATCH_CONFIRMED", "is_live": true}'
    verdict = VLMProvider._parse(VLMProvider, raw_response)
    assert verdict.is_live_selfie is True
    assert verdict.is_match is True
    assert verdict.confidence_score == 0.95


def test_vlm_liveness_fail_closed_on_none():
    raw_response = '{"match": true, "confidence": 0.95, "verdict": "MATCH_CONFIRMED", "is_live": null}'
    verdict = VLMProvider._parse(VLMProvider, raw_response)
    assert verdict.is_live_selfie is False
