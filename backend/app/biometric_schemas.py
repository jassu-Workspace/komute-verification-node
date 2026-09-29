"""
Phase 3: strict biometric verdict contract.

This module is the single source of truth for what a cloud VLM is allowed to say
about a biometric comparison. Enforcing it with a validated schema (rather than
trusting free-text prose) is what removes the historical fail-open path where a
non-JSON reply containing the word "match" produced is_match=True / confidence
0.85 / is_live_selfie=True and therefore a false APPROVED decision.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

VerdictLabel = Literal["MATCH_CONFIRMED", "MISMATCH", "INCONCLUSIVE"]


class BiometricVerdict(BaseModel):
    """Validated provider response. Anything that does not satisfy this is rejected."""

    model_config = ConfigDict(extra="ignore")

    is_match: bool = Field(..., description="Both crops depict the same individual")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Craniofacial similarity confidence (0.0 - 1.0)",
    )
    is_live_selfie: bool = Field(
        ...,
        description="Anti-spoof result for the live selfie crop",
    )
    estimated_age_delta_years: Optional[int] = Field(
        default=None,
        description="Estimated age gap between the ID portrait and the live selfie",
    )
    facial_feature_notes: str = Field("", description="Landmark-level geometry notes")
    reasoning: str = Field("", description="Forensic reasoning summary")
    verdict: VerdictLabel = Field(..., description="Forensic verdict label")

    @field_validator("confidence_score")
    @classmethod
    def _reject_non_finite(cls, value: float) -> float:
        # NaN / inf must never be coerced into a pass.
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("confidence_score must be a finite number")
        return value

    @field_validator("facial_feature_notes", "reasoning", mode="before")
    @classmethod
    def _coerce_none_to_empty(cls, value: object) -> object:
        return "" if value is None else value

    def to_provider_dict(self) -> dict:
        return self.model_dump()


# Explicit schema description handed to providers that support structured output.
BIOMETRIC_VERDICT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "is_match": {"type": "boolean"},
        "confidence_score": {"type": "number"},
        "is_live_selfie": {"type": "boolean"},
        "estimated_age_delta_years": {"type": "integer"},
        "facial_feature_notes": {"type": "string"},
        "reasoning": {"type": "string"},
        "verdict": {
            "type": "string",
            "enum": list(VerdictLabel.__args__),
        },
    },
    "required": [
        "is_match",
        "confidence_score",
        "is_live_selfie",
        "facial_feature_notes",
        "reasoning",
        "verdict",
    ],
    "propertyOrdering": [
        "is_match",
        "confidence_score",
        "is_live_selfie",
        "estimated_age_delta_years",
        "facial_feature_notes",
        "reasoning",
        "verdict",
    ],
}
