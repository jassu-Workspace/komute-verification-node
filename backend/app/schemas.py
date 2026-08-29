from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class DecisionEnum(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class PersonalInfo(BaseModel):
    full_name: str = Field(..., description="Full Name")
    email_address: Optional[str] = Field(default="driver@komute.com", description="Email Address")
    mobile_number: Optional[str] = Field(default="+10000000000", description="Mobile Number")
    date_of_birth: str = Field(..., description="Date of Birth in MM-DD-YYYY or YYYY-MM-DD")
    gender: Optional[str] = Field(None, description="Gender")
    role: Optional[str] = Field(default="Driver", description="Role (e.g. Driver)")
    password: Optional[str] = Field(None, description="Password")
    self_description: Optional[str] = Field(None, description="Self Description")


class LicenseDetails(BaseModel):
    license_number: str = Field(..., description="License Number (e.g., DL1234567)")
    issuing_province: Optional[str] = Field(default="ON", description="Issuing Province/State")
    license_expiry_date: Optional[str] = Field(default=None, description="License Expiry Date (MM-DD-YYYY or YYYY-MM-DD)")
    expiry_date: Optional[str] = Field(default=None, description="Alias for license_expiry_date")

    @model_validator(mode="before")
    @classmethod
    def resolve_expiry(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if not values.get("license_expiry_date") and values.get("expiry_date"):
                values["license_expiry_date"] = values["expiry_date"]
            elif not values.get("expiry_date") and values.get("license_expiry_date"):
                values["expiry_date"] = values["license_expiry_date"]
            if not values.get("license_expiry_date"):
                values["license_expiry_date"] = "2030-01-01"
            if not values.get("issuing_province"):
                values["issuing_province"] = "ON"
        return values


class VehicleDetails(BaseModel):
    make: Optional[str] = Field(default="Standard", description="Vehicle Make")
    model: Optional[str] = Field(default="Vehicle", description="Vehicle Model")
    year: Optional[Union[str, int]] = Field(default=2022, description="Vehicle Year")
    color: Optional[str] = Field(default="Unknown", description="Vehicle Color")
    plate: Optional[str] = Field(default=None, description="License Plate (e.g., DL01AB1234)")
    plate_number: Optional[str] = Field(default=None, description="Alias for plate")

    @model_validator(mode="before")
    @classmethod
    def resolve_plate(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if not values.get("plate") and values.get("plate_number"):
                values["plate"] = values["plate_number"]
            elif not values.get("plate_number") and values.get("plate"):
                values["plate_number"] = values["plate"]
            if not values.get("plate"):
                values["plate"] = ""
            if not values.get("color"):
                values["color"] = "Unknown"
        return values



class VerificationImages(BaseModel):
    selfie_base64: str = Field(..., description="Base64 encoded live user selfie image")
    license_image_base64: str = Field(..., description="Base64 encoded driving license card image")
    vehicle_photo_base64: str = Field(..., description="Base64 encoded vehicle exterior photo")


class VerificationRequest(BaseModel):
    request_id: str = Field(..., description="Unique request tracing ID")
    driver_id: str = Field(..., description="Unique driver identifier")
    personal_info: PersonalInfo
    license_details: LicenseDetails
    vehicle_details: VehicleDetails
    images: VerificationImages


# Stage 1 Schema
class LicenseOcrResult(BaseModel):
    passed: bool = Field(..., description="Whether DL OCR validation criteria passed")
    extracted_dl_number: Optional[str] = Field(None, description="DL number extracted from card")
    extracted_name: Optional[str] = Field(None, description="Name extracted from card")
    extracted_dob: Optional[str] = Field(None, description="DOB extracted from card")
    extracted_expiry: Optional[str] = Field(None, description="Expiry date extracted from card")
    extracted_province: Optional[str] = Field(None, description="Issuing state/province extracted")
    extracted_address: Optional[str] = Field(None, description="Address extracted from card")
    extracted_gender: Optional[str] = Field(None, description="Gender extracted from card")
    extracted_class: Optional[str] = Field(None, description="License class extracted from card")
    extracted_issue_date: Optional[str] = Field(None, description="Issue date extracted from card")
    extracted_conditions: Optional[str] = Field(None, description="Conditions or endorsements extracted from card")
    number_matched: bool = Field(False, description="Whether DL number matched registration")
    number_similarity: float = Field(0.0, description="Levenshtein similarity for DL number")
    name_matched: bool = Field(False, description="Whether name matched registration")
    name_similarity: float = Field(0.0, description="Token sort similarity for full name")
    dob_matched: bool = Field(False, description="Whether DOB matches registered DOB")
    is_expired: bool = Field(False, description="Whether driving license has expired")
    driver_age_valid: bool = Field(True, description="Whether driver is >= 18 years old")
    confidence: float = Field(0.0, description="Stage 1 confidence score")
    details: str = Field("", description="Diagnostic summary of Stage 1")
    raw_ocr_lines: List[str] = Field(default_factory=list, description="Raw detected text lines")
    missing_fields: List[str] = Field(default_factory=list, description="Fields entered by user but not found in DL image")
    confidence_proof: Dict[str, Any] = Field(default_factory=dict, description="Granular mathematical proof & metrics justification")


# Stage 2 Schema
class FaceBiometricsResult(BaseModel):
    passed: bool = Field(..., description="Whether VLM biometric verification passed")
    vlm_confidence: float = Field(0.0, description="Cloud VLM biometric match confidence score (0.0 - 1.0)")
    is_live: bool = Field(True, description="Liveness & anti-spoof assessment")
    is_match: bool = Field(False, description="Whether face crops belong to the same person")
    estimated_age_delta_years: Optional[int] = Field(None, description="Estimated age gap between selfie & ID")
    privacy_face_cropped: bool = Field(True, description="True if ID was tightly cropped to facial region")
    pii_leakage_prevented: bool = Field(True, description="True if all card text/numbers/PII were excluded")
    facial_feature_notes: Optional[str] = Field(None, description="Facial landmark & geometry analysis")
    reasoning: str = Field("", description="Detailed biometric comparison & forensic reasoning")
    verdict: str = Field("UNKNOWN", description="VLM verdict (MATCH_CONFIRMED, MISMATCH, INCONCLUSIVE)")
    dl_face_detected: bool = Field(False, description="Whether face was detected on DL image")
    selfie_face_detected: bool = Field(False, description="Whether face was detected on selfie")
    dl_face_crop_preview: Optional[str] = Field(None, description="Base64 preview of isolated DL face crop")
    selfie_face_crop_preview: Optional[str] = Field(None, description="Base64 preview of normalized selfie crop")
    confidence_proof: Dict[str, Any] = Field(default_factory=dict, description="Forensic biometric geometry & liveness proof")


# Stage 3 Schema
class VehicleVerificationResult(BaseModel):
    passed: bool = Field(..., description="Whether vehicle ALPR & color verification passed")
    confidence: float = Field(0.0, description="Stage 3 overall vehicle verification confidence score")
    extracted_plate: Optional[str] = Field(None, description="Plate number extracted by ALPR engine")
    plate_matched: bool = Field(False, description="Whether extracted plate matches submitted registration")
    plate_similarity: float = Field(0.0, description="Plate string similarity after OCR confusion normalization")
    color_matched: bool = Field(False, description="Whether vehicle body color matches registered color")
    detected_color: str = Field("Unknown", description="Dominant vehicle color detected")
    color_confidence: float = Field(0.0, description="Color detection confidence")
    raw_ocr_lines: List[str] = Field(default_factory=list, description="Raw detected text lines from ALPR")
    plate_box_detected: bool = Field(False, description="Whether license plate contour was detected")
    details: str = Field("", description="Diagnostic summary of Stage 3")
    plate_crop_preview: Optional[str] = Field(None, description="Base64 preview of isolated vehicle license plate crop")
    confidence_proof: Dict[str, Any] = Field(default_factory=dict, description="ALPR confusion matrix & color histogram proof")


class StagesBreakdown(BaseModel):
    license_ocr: LicenseOcrResult
    face_biometrics: FaceBiometricsResult
    vehicle_verification: VehicleVerificationResult


class StorageArtifacts(BaseModel):
    session_directory: Optional[str] = Field(None, description="Absolute path to session root folder")
    originals_folder: Optional[str] = Field(None, description="Path to raw uploaded originals subfolder")
    original_folder: Optional[str] = Field(None, description="Path to originals subfolder (alias)")
    compressed_folder: Optional[str] = Field(None, description="Path to WebP compressed images subfolder")
    cropped_folder: Optional[str] = Field(None, description="Path to cropped images subfolder")
    original_files: Dict[str, Any] = Field(default_factory=dict, description="Saved raw original image paths & metadata")
    compressed_files: Dict[str, Any] = Field(default_factory=dict, description="Saved compressed image paths & metadata")
    cropped_files: Dict[str, Any] = Field(default_factory=dict, description="Saved cropped image paths & metadata")
    metadata_file: Optional[str] = Field(None, description="Path to saved session_metadata.json")


class VerificationResponse(BaseModel):
    request_id: str = Field(..., description="Tracing request ID")
    driver_id: str = Field(..., description="Driver identifier")
    decision: DecisionEnum = Field(..., description="Overall verification decision (APPROVED, MANUAL_REVIEW, REJECTED)")
    composite_confidence: float = Field(..., description="Weighted composite confidence score (0.0 - 1.0)")
    execution_time_ms: float = Field(..., description="Total pipeline execution latency in milliseconds")
    stages: StagesBreakdown = Field(..., description="Detailed stage-by-stage results")
    rejection_reasons: List[str] = Field(default_factory=list, description="List of reasons triggering rejection")
    manual_review_reasons: List[str] = Field(default_factory=list, description="List of reasons requiring human review")
    saved_artifacts: Optional[StorageArtifacts] = Field(None, description="Paths to persisted original and cropped images")
    composite_proof: Dict[str, Any] = Field(default_factory=dict, description="Full mathematical breakdown justifying final score")


# Privacy Preview Endpoint
class PrivacyCropPreviewRequest(BaseModel):
    image_base64: str = Field(..., description="Driving license image in base64 format")
    margin_ratio: float = Field(0.15, description="Padding margin ratio around detected face (default 0.15)")


class PrivacyCropPreviewResponse(BaseModel):
    face_detected: bool
    bounding_box: Optional[Dict[str, int]] = None
    face_crop_base64: Optional[str] = None
    pii_sanitized: bool
    message: str
    saved_artifacts: Optional[Dict[str, Any]] = Field(None, description="Paths to persisted preview images")


# Health Check Schema
class SubsystemChecks(BaseModel):
    ocr_engine: bool = Field(True, description="RapidOCR Engine readiness")
    face_detector: bool = Field(True, description="YuNet Face Detector readiness")
    storage_writable: bool = Field(True, description="Upload storage filesystem writable check")
    memory_usage_mb: Optional[float] = Field(None, description="Current process RAM usage in MB")


class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Overall health status (healthy or degraded)")
    app: str = Field(default="Komüte Driver Verification Service", description="Application Name")
    version: str = Field(..., description="Application semantic version")
    environment: str = Field("production", description="Active runtime environment")
    uptime_seconds: float = Field(0.0, description="Uptime duration in seconds since boot")
    uptime_human: str = Field("0s", description="Human-readable uptime string")
    timestamp: str = Field(..., description="Current UTC ISO 8601 timestamp")
    vlm_provider: str = Field(..., description="Configured Vision-Language Model provider")
    vlm_model: str = Field(..., description="Configured VLM model identifier")
    ocr_available: bool = Field(True, description="Whether OCR inference engine is operational")
    face_detector_available: bool = Field(True, description="Whether Face detector is operational")
    checks: Optional[SubsystemChecks] = Field(None, description="Detailed breakdown of subsystem health")

