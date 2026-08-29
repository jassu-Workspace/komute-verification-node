import asyncio
import logging
import time
from typing import List, Tuple, Dict, Any
from app.config import settings
from app.schemas import (
    DecisionEnum,
    FaceBiometricsResult,
    LicenseOcrResult,
    StagesBreakdown,
    StorageArtifacts,
    VehicleVerificationResult,
    VerificationRequest,
    VerificationResponse,
)
from core.dl_ocr import dl_ocr_engine
from core.face_privacy_cropper import face_privacy_cropper
from core.image_utils import compress_and_normalize_base64, decode_base64_to_image
from storage.storage import storage_manager
from core.vehicle_alpr import vehicle_alpr_engine
from core.vlm_face_verifier import vlm_face_verifier

logger = logging.getLogger(__name__)


class VerificationPipeline:
    """
    Master 3-Stage Orchestrator and Composite Decision Engine.
    Executes License OCR, Zero-PII Face Biometrics, and Vehicle ALPR with memory-safe execution.
    """

    async def execute_verification(self, request: VerificationRequest) -> VerificationResponse:
        """
        Run end-to-end multi-stage identity & vehicle verification.
        """
        start_time = time.perf_counter()
        logger.info(f"Starting verification pipeline for request_id: {request.request_id}, driver_id: {request.driver_id}")

        # 1. Image Decoding & Preprocessing
        # NOTE (ponytail: keep license image uncompressed during verification to prevent OCR quality degradation)
        # License image is preserved at 100% native resolution for Stage 1 OCR and Face Cropper.
        try:
            raw_dl_img = decode_base64_to_image(request.images.license_image_base64)
            raw_selfie_img = decode_base64_to_image(request.images.selfie_base64)
            raw_vehicle_img = decode_base64_to_image(request.images.vehicle_photo_base64)

            from core.image_utils import flatten_id_card
            raw_dl_img = flatten_id_card(raw_dl_img)

            # Uncompressed original DL image used for all verification stages
            dl_img = raw_dl_img

            selfie_img, _ = compress_and_normalize_base64(
                request.images.selfie_base64,
                out_format="webp",
                max_dim=1200,
                quality=80,
            )
            vehicle_img, _ = compress_and_normalize_base64(
                request.images.vehicle_photo_base64,
                out_format="webp",
                max_dim=1200,
                quality=90,
            )
        except Exception as e:
            logger.error(f"Image decoding error: {e}")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return VerificationResponse(
                request_id=request.request_id,
                driver_id=request.driver_id,
                decision=DecisionEnum.REJECTED,
                composite_confidence=0.0,
                execution_time_ms=round(elapsed_ms, 2),
                stages=StagesBreakdown(
                    license_ocr=LicenseOcrResult(
                        passed=False,
                        details=f"Image decoding failed: {e}",
                    ),
                    face_biometrics=FaceBiometricsResult(
                        passed=False,
                        reasoning=f"Image decoding failed: {e}",
                    ),
                    vehicle_verification=VehicleVerificationResult(
                        passed=False,
                        details=f"Image decoding failed: {e}",
                    ),
                ),
                rejection_reasons=[f"Invalid or corrupted image payload: {e}"],
                manual_review_reasons=[],
            )

        # 2. Stage 2A: Privacy Face Isolation (Zero-PII Cropping with RetinaFace MobileNet0.25 Verification Layer)
        # Extract driver face ONLY from DL card using full uncompressed native resolution
        dl_face_crop, dl_bbox, dl_sanitized = face_privacy_cropper.extract_isolated_face(
            img_bgr=dl_img,
            margin_ratio=settings.face_crop_padding_ratio,
        )

        # Stage 2B: Cloud VLM Biometric Verification (runs asynchronously over network)
        stage2_task = asyncio.create_task(
            vlm_face_verifier.verify_biometrics(
                selfie_crop_bgr=selfie_img,
                dl_face_crop_bgr=dl_face_crop,
            )
        )

        # Execute Stage 1 (DL OCR on uncompressed full-res image) and Stage 3 (Vehicle ALPR)
        stage1_res = await asyncio.to_thread(
            dl_ocr_engine.verify_license,
            img_bgr=dl_img,
            personal_info=request.personal_info,
            license_details=request.license_details,
        )

        stage3_output = await asyncio.to_thread(
            vehicle_alpr_engine.verify_vehicle,
            vehicle_img=vehicle_img,
            vehicle_details=request.vehicle_details,
            return_crop=True,
        )

        # Await Stage 2 Biometrics
        stage2_res = await stage2_task

        # Unpack Stage 3 output (vehicle verification result + cropped license plate)
        if isinstance(stage3_output, tuple):
            stage3_res, plate_crop = stage3_output
        else:
            stage3_res, plate_crop = stage3_output, None

        # 3. Composite Decision Matrix
        decision, composite_score, rejection_reasons, manual_reasons, composite_proof = self._evaluate_composite_decision(
            s1=stage1_res,
            s2=stage2_res,
            s3=stage3_res,
        )

        # 4. Post-Verification Conditional Compression
        # Compress license image ONLY if verification succeeded (APPROVED)
        compressed_images_dict = {
            "selfie": selfie_img,
            "vehicle": vehicle_img,
        }

        if decision == DecisionEnum.APPROVED:
            try:
                compressed_dl_img, _ = compress_and_normalize_base64(
                    request.images.license_image_base64,
                    out_format="webp",
                    max_dim=1400,
                    quality=70,
                )
                compressed_images_dict["license"] = compressed_dl_img
                logger.info(f"Verification APPROVED: compressed license image to WebP for archival.")
            except Exception as e:
                logger.warning(f"Post-approval license compression warning: {e}")
        else:
            logger.info("Verification REJECTED: license image compression skipped as requested.")

        # 5. Persist original, compressed, and cropped images into structured uploads folders
        saved_storage_data = None
        try:
            saved_storage_data = storage_manager.save_verification_session_images(
                driver_id=request.driver_id,
                request_id=request.request_id,
                original_images={
                    "selfie": raw_selfie_img,
                    "license": raw_dl_img,
                    "vehicle": raw_vehicle_img,
                },
                compressed_images=compressed_images_dict,
                cropped_images={
                    "dl_face": dl_face_crop,
                    "vehicle_plate": plate_crop,
                },
                metadata={
                    "driver_name": request.personal_info.full_name,
                    "license_number": request.license_details.license_number,
                    "vehicle_plate": request.vehicle_details.plate,
                    "verification_decision": decision.value,
                    "composite_confidence": composite_score,
                    "rejection_reasons": rejection_reasons,
                    "stages": {
                        "license_ocr": stage1_res.model_dump(),
                        "face_biometrics": stage2_res.model_dump(),
                        "vehicle_alpr": stage3_res.model_dump(),
                    },
                    "composite_proof": composite_proof,
                    "dl_face_detected": stage2_res.dl_face_detected,
                    "selfie_face_detected": stage2_res.selfie_face_detected,
                },
            )
        except Exception as e:
            logger.error(f"Error saving session images to uploads: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            f"Verification complete for {request.driver_id}: Decision={decision}, "
            f"Score={composite_score:.3f}, Latency={elapsed_ms:.1f}ms"
        )

        saved_artifacts = StorageArtifacts(**saved_storage_data) if saved_storage_data else None

        return VerificationResponse(
            request_id=request.request_id,
            driver_id=request.driver_id,
            decision=decision,
            composite_confidence=round(composite_score, 3),
            execution_time_ms=round(elapsed_ms, 2),
            stages=StagesBreakdown(
                license_ocr=stage1_res,
                face_biometrics=stage2_res,
                vehicle_verification=stage3_res,
            ),
            rejection_reasons=rejection_reasons,
            manual_review_reasons=manual_reasons,
            saved_artifacts=saved_artifacts,
            composite_proof=composite_proof,
        )

    def _evaluate_composite_decision(
        self,
        s1: LicenseOcrResult,
        s2: FaceBiometricsResult,
        s3: VehicleVerificationResult,
    ) -> Tuple[DecisionEnum, float, List[str], List[str], Dict[str, Any]]:
        """
        Evaluate multi-stage confidence and audit criteria against hard and soft thresholds.
        """
        rejection_reasons: List[str] = []
        manual_reasons: List[str] = []

        # 1. Hard Rejection Triggers
        if s1.is_expired:
            rejection_reasons.append("Driving license has EXPIRED")

        if not s1.driver_age_valid:
            rejection_reasons.append("Driver is under 18 years of age")

        if not s2.dl_face_detected:
            rejection_reasons.append("No driver portrait could be isolated from Driving License card")

        if not s2.is_match and s2.vlm_confidence < 0.50:
            rejection_reasons.append(
                f"Facial biometrics mismatch: VLM confidence {s2.vlm_confidence:.2f} below minimum threshold"
            )

        if not s2.is_live:
            rejection_reasons.append("Selfie liveness check failed: spoof / photo replay detected")

        if s3.plate_similarity < 0.40 and not s3.plate_matched:
            rejection_reasons.append(
                f"Vehicle plate mismatch: extracted '{s3.extracted_plate}' does not match registration"
            )

        # 2. Stage Confidence Calculations
        s1_score = s1.confidence
        s2_score = s2.vlm_confidence if s2.is_live else 0.0
        s3_score = (s3.plate_similarity * 0.75) + (0.25 if s3.color_matched else 0.0)

        # Composite score weighting: DL (0.30) + Face (0.45) + Vehicle (0.25)
        s1_weight = 0.30
        s2_weight = 0.45
        s3_weight = 0.25

        s1_points = s1_score * s1_weight
        s2_points = s2_score * s2_weight
        s3_points = s3_score * s3_weight

        composite_score = min(1.0, max(0.0, s1_points + s2_points + s3_points))

        # 3. Stage Failure Checks
        if not s1.passed:
            if s1.is_expired:
                rejection_reasons.append("Driving license has EXPIRED")
            elif not s1.driver_age_valid:
                rejection_reasons.append("Driver is under 18 years of age")
            elif not s1.number_matched:
                rejection_reasons.append(f"DL number mismatch: extracted '{s1.extracted_dl_number}', expected registration match")
            elif not s1.name_matched:
                rejection_reasons.append(f"DL full name mismatch: extracted '{s1.extracted_name}'")
            else:
                rejection_reasons.append(f"Stage 1 Driving License OCR validation failed: {s1.details}")

        if not s2.passed:
            if not s2.dl_face_detected:
                rejection_reasons.append("No driver portrait could be isolated from Driving License card")
            elif not s2.selfie_face_detected:
                rejection_reasons.append("No human face detected in uploaded selfie image")
            elif not s2.is_live:
                rejection_reasons.append("Selfie liveness check failed: spoof / photo replay detected")
            elif not s2.is_match:
                rejection_reasons.append(f"Facial biometrics mismatch: VLM confidence {s2.vlm_confidence:.2f}")
            else:
                rejection_reasons.append(f"Stage 2 Face Biometrics verification failed: {s2.reasoning or s2.verdict}")

        if not s3.passed:
            if not s3.plate_matched and s3.plate_similarity < 0.85:
                rejection_reasons.append(
                    f"Vehicle plate mismatch: extracted '{s3.extracted_plate}' (similarity: {s3.plate_similarity:.2f})"
                )
            if not s3.color_matched:
                rejection_reasons.append(
                    f"Vehicle color mismatch: detected '{s3.detected_color}' differs from registration"
                )

        # 4. Strict Binary Classification (APPROVED vs REJECTED)
        all_stages_passed = s1.passed and s2.passed and s3.passed
        if all_stages_passed and composite_score >= settings.composite_approval_threshold and not rejection_reasons:
            decision = DecisionEnum.APPROVED
        else:
            decision = DecisionEnum.REJECTED
            if not rejection_reasons:
                rejection_reasons.append(f"Composite confidence score {composite_score:.2f} is below approval threshold ({settings.composite_approval_threshold:.2f})")

        # Granular Mathematical Breakdown Justifying Composite Score
        composite_proof = {
            "formula": "(0.30 * Stage_1_OCR) + (0.45 * Stage_2_Biometrics) + (0.25 * Stage_3_Vehicle)",
            "weights": {
                "stage_1_dl_ocr": {
                    "weight": s1_weight,
                    "stage_score": round(float(s1_score), 3),
                    "weighted_contribution": round(float(s1_points), 3),
                    "passed": s1.passed,
                },
                "stage_2_face_biometrics": {
                    "weight": s2_weight,
                    "stage_score": round(float(s2_score), 3),
                    "weighted_contribution": round(float(s2_points), 3),
                    "passed": s2.passed,
                },
                "stage_3_vehicle_alpr": {
                    "weight": s3_weight,
                    "stage_score": round(float(s3_score), 3),
                    "weighted_contribution": round(float(s3_points), 3),
                    "passed": s3.passed,
                },
            },
            "composite_sum": round(float(composite_score), 3),
            "approval_threshold": settings.composite_approval_threshold,
            "threshold_passed": bool(composite_score >= settings.composite_approval_threshold),
            "all_stages_passed": bool(all_stages_passed),
            "rejection_triggers_active": len(rejection_reasons) > 0,
            "final_decision": decision.value,
            "justification_summary": f"Calculated composite score {composite_score*100:.1f}% >= threshold {settings.composite_approval_threshold*100:.0f}% with 0 hard rejection triggers" if decision == DecisionEnum.APPROVED else f"Rejected: {'; '.join(rejection_reasons)}",
        }

        return decision, composite_score, rejection_reasons, [], composite_proof


# Global instance
pipeline_engine = VerificationPipeline()
