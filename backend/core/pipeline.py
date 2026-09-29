import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional
from app.config import settings
from app.schemas import (
    DecisionEnum,
    FaceBiometricsResult,
    LicenseOcrResult,
    ServiceStatus,
    StagesBreakdown,
    StorageArtifacts,
    VehicleVerificationResult,
    VerificationRequest,
    VerificationResponse,
)
from core.dl_ocr import dl_ocr_engine
from core.face_privacy_cropper import face_privacy_cropper
from core.image_utils import (
    compress_and_normalize_base64,
    decode_base64_to_image,
    flatten_id_card,
)
from storage.storage import storage_manager
from core.vehicle_alpr import vehicle_alpr_engine
from core.vlm_face_verifier import vlm_face_verifier

logger = logging.getLogger(__name__)


class VerificationPipeline:
    """
    Master 3-Stage Orchestrator and Composite Decision Engine.
    Executes License OCR, Zero-PII Face Biometrics, and Vehicle ALPR with memory-safe execution.
    """

    async def execute_verification(
        self,
        request: VerificationRequest,
        background_tasks: Optional[Any] = None,
    ) -> VerificationResponse:
        """
        Run end-to-end multi-stage identity & vehicle verification with memory-safe execution.
        """
        start_time = time.perf_counter()
        logger.info(f"Starting verification pipeline for request_id: {request.request_id}, driver_id: {request.driver_id}")

        # 1. Image Decoding & Preprocessing
        # NOTE: License image is flattened and preserved for Stage 1 OCR and Face Cropper.
        try:
            raw_dl_img = decode_base64_to_image(request.images.license_image_base64)
            raw_selfie_img = decode_base64_to_image(request.images.selfie_base64)
            raw_vehicle_img = decode_base64_to_image(request.images.vehicle_photo_base64)

            raw_dl_img = flatten_id_card(raw_dl_img)

            # Pre-flattened DL image used for downstream verification
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

        # 2. Stage 2A: Privacy Face Isolation (Zero-PII Cropping with YuNet Landmark Verification Layer)
        # Extract driver face ONLY from DL card using native resolution
        dl_face_crop, dl_bbox, dl_sanitized = await asyncio.to_thread(
            face_privacy_cropper.extract_isolated_face,
            img_bgr=dl_img,
            margin_ratio=settings.face_crop_padding_ratio,
        )

        # Zero-PII Selfie Isolation: Crop selfie face to exclude private room/background scenes before cloud transmission
        selfie_face_crop, selfie_bbox, _ = await asyncio.to_thread(
            face_privacy_cropper.extract_isolated_face,
            img_bgr=selfie_img,
            margin_ratio=settings.face_crop_padding_ratio,
        )
        if selfie_face_crop is None:
            selfie_face_crop = selfie_img

        # Concurrently execute Stage 1 (DL OCR), Stage 2B (Cloud VLM Biometrics), and Stage 3 (Vehicle ALPR)
        stage1_task = asyncio.to_thread(
            dl_ocr_engine.verify_license,
            img_bgr=dl_img,
            personal_info=request.personal_info,
            license_details=request.license_details,
            is_preflattened=True,
        )

        stage2_task = vlm_face_verifier.verify_biometrics(
            selfie_crop_bgr=selfie_face_crop,
            dl_face_crop_bgr=dl_face_crop,
        )

        stage3_task = asyncio.to_thread(
            vehicle_alpr_engine.verify_vehicle,
            vehicle_img=vehicle_img,
            vehicle_details=request.vehicle_details,
            return_crop=True,
        )

        timeout_sec = getattr(settings, "pipeline_timeout_seconds", 45.0)
        named_tasks = {
            "license_ocr": asyncio.ensure_future(stage1_task),
            "face_biometrics": asyncio.ensure_future(stage2_task),
            "vehicle_alpr": asyncio.ensure_future(stage3_task),
        }
        try:
            done, pending = await asyncio.wait(
                named_tasks.values(), timeout=timeout_sec
            )
            if pending:
                # Attribute the timeout to the specific stage(s) still running so a
                # slow VLM is never misreported as an OCR/ALPR problem.
                stalled = sorted(
                    name for name, task in named_tasks.items() if task in pending
                )
                raise asyncio.TimeoutError(f"stalled stages: {', '.join(stalled)}")

            stage1_res = named_tasks["license_ocr"].result()
            stage2_res = named_tasks["face_biometrics"].result()
            stage3_output = named_tasks["vehicle_alpr"].result()
        except (asyncio.TimeoutError, TimeoutError) as e:
            stalled_note = str(e) if "stalled stages" in str(e) else "unknown"
            logger.error(
                "Verification pipeline stages timed out after %ss (%s).", timeout_sec, stalled_note
            )
            for task in named_tasks.values():
                if not task.done():
                    task.cancel()
            vlm_timed_out = "face_biometrics" in stalled_note
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
                        details=(
                            f"Verification timed out after {timeout_sec}s "
                            f"({stalled_note})."
                            if "license_ocr" in stalled_note
                            else "Stage 1 completed within the pipeline budget."
                        ),
                    ),
                    face_biometrics=FaceBiometricsResult(
                        passed=False,
                        reasoning=(
                            f"Verification timed out after {timeout_sec}s during biometric "
                            "matching. The cloud VLM exceeded its latency budget."
                            if vlm_timed_out
                            else "Stage 2 completed within the pipeline budget."
                        ),
                        vlm_attempt_chain=[
                            {
                                "provider": settings.vlm_provider,
                                "model": settings.active_model,
                                "ok": False,
                                "reason": "timeout",
                                "latency_ms": 0,
                            }
                        ] if vlm_timed_out else [],
                    ),
                    vehicle_verification=VehicleVerificationResult(
                        passed=False,
                        details=(
                            f"Verification timed out after {timeout_sec}s "
                            f"({stalled_note})."
                            if "vehicle_alpr" in stalled_note
                            else "Stage 3 completed within the pipeline budget."
                        ),
                    ),
                ),
                rejection_reasons=[
                    f"Verification processing timed out ({timeout_sec}s limit exceeded, "
                    f"stalled: {stalled_note}). Please retry with clearer or smaller images."
                ],
                manual_review_reasons=["Processing timeout: Exceeded SLA latency ceiling."],
                service_status="degraded",
                degraded_subsystems=[
                    ServiceStatus(
                        name="verification_pipeline",
                        status="degraded",
                        reason="timeout",
                        detail=f"Stalled stages after {timeout_sec}s: {stalled_note}",
                        last_error_at=datetime.now(timezone.utc),
                    )
                ],
            )

        # Unpack Stage 3 output (vehicle verification result + cropped license plate)
        if isinstance(stage3_output, tuple):
            stage3_res, plate_crop = stage3_output
        else:
            stage3_res, plate_crop = stage3_output, None


        # 3. Composite Decision Matrix
        (
            decision,
            composite_score,
            rejection_reasons,
            manual_reasons,
            composite_proof,
            degraded,
        ) = self._evaluate_composite_decision(
            s1=stage1_res,
            s2=stage2_res,
            s3=stage3_res,
        )

        service_status = "degraded" if degraded else "ok"

        # 4. Post-Decision Image Compression
        compressed_images_dict = {
            "selfie": selfie_img,
            "vehicle": vehicle_img,
        }

        if decision == DecisionEnum.APPROVED:
            try:
                compressed_dl_img, _ = compress_and_normalize_base64(
                    request.images.license_image_base64,
                    out_format="webp",
                    max_dim=1200,
                    quality=80,
                )
                compressed_images_dict["license"] = compressed_dl_img
                logger.info(f"Verification APPROVED: compressed license image to WebP for archival.")
            except Exception as e:
                logger.warning(f"Post-approval license compression warning: {e}")
        else:
            logger.info("Verification REJECTED: license image compression skipped as requested.")

        # 5. Persist original, compressed, and cropped images into structured uploads folders
        storage_kwargs = {
            "driver_id": request.driver_id,
            "request_id": request.request_id,
            "original_images": {
                "selfie": raw_selfie_img,
                "license": raw_dl_img,
                "vehicle": raw_vehicle_img,
            },
            "compressed_images": compressed_images_dict,
            "cropped_images": {
                "dl_face": dl_face_crop,
                "vehicle_plate": plate_crop,
            },
            "metadata": {
                "driver_name": request.personal_info.full_name,
                "license_number": request.license_details.license_number,
                "vehicle_plate": request.vehicle_details.plate,
                "verification_decision": decision.value,
                "composite_confidence": composite_score,
                "rejection_reasons": rejection_reasons,
                "service_status": service_status,
                "degraded_subsystems": [d.model_dump(mode="json") for d in degraded],
                "stages": {
                    "license_ocr": stage1_res.model_dump(),
                    "face_biometrics": stage2_res.model_dump(),
                    "vehicle_verification": stage3_res.model_dump(),
                },
                "composite_proof": composite_proof,
            },
        }

        saved_storage_data = None
        try:
            if background_tasks is not None:
                background_tasks.add_task(storage_manager.save_verification_session_images, **storage_kwargs)
            else:
                saved_storage_data = await asyncio.to_thread(
                    storage_manager.save_verification_session_images,
                    **storage_kwargs,
                )
        except Exception as e:
            logger.error(f"Error saving session images to uploads: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            f"Verification complete for {request.driver_id}: Decision={decision}, "
            f"Score={composite_score:.3f}, Latency={elapsed_ms:.1f}ms, "
            f"ServiceStatus={service_status}"
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
            service_status=service_status,
            degraded_subsystems=degraded,
        )

    def _evaluate_composite_decision(
        self,
        s1: LicenseOcrResult,
        s2: FaceBiometricsResult,
        s3: VehicleVerificationResult,
    ) -> Tuple[DecisionEnum, float, List[str], List[str], Dict[str, Any], List[ServiceStatus]]:
        """
        Evaluate multi-stage confidence and audit criteria against hard and soft thresholds.

        Returns the decision, score, rejection reasons, manual-review reasons, the
        score proof, and any degraded subsystem statuses. A degraded subsystem is an
        infrastructure outage and is reported separately from an identity rejection
        so an HTTP 200 response can never hide a broken dependency.
        """
        rejection_reasons: List[str] = []
        manual_reasons: List[str] = []
        degraded: List[ServiceStatus] = []
        hard_triggers: List[str] = []

        # 1. Evaluate Hard Triggers
        if s1.is_expired:
            hard_triggers.append("Driving license has EXPIRED")

        if not s1.driver_age_valid:
            hard_triggers.append("Driver is under 18 years of age")

        if not s2.dl_face_detected:
            hard_triggers.append("No driver portrait could be isolated from Driving License card")

        if s2.verdict == "SERVICE_UNAVAILABLE":
            hard_triggers.append("Cloud facial biometric service is temporarily unavailable. Fail-safe rejection enforced.")
            degraded.append(
                ServiceStatus(
                    name="cloud_vlm_biometrics",
                    status="degraded",
                    reason=(
                        (s2.confidence_proof.get("metrics", {})
                         .get("provider_provenance", {})
                         .get("reason_code"))
                        or "provider_error"
                    ),
                    detail=s2.reasoning[:300] or None,
                    provider=s2.vlm_provider_used,
                    model=s2.vlm_model_used,
                    last_error_at=datetime.now(timezone.utc),
                )
            )
        else:
            if not s2.is_match and s2.vlm_confidence < settings.vlm_mismatch_floor:
                hard_triggers.append(
                    f"Facial biometrics mismatch: VLM confidence {s2.vlm_confidence:.2f} below minimum threshold"
                )

            if not s2.is_live:
                hard_triggers.append("Selfie liveness check failed: spoof / photo replay detected")

        if s3.plate_similarity < settings.plate_hard_floor and not s3.plate_matched:
            hard_triggers.append(
                f"Vehicle plate mismatch: extracted '{s3.extracted_plate}' does not match registration"
            )

        # 2. Field-Level Scoring & Confidence Calculation
        field_items = [
            ("Full Name", float(s1.name_similarity), float(settings.field_weight_name)),
            ("DL Number", float(s1.number_similarity), float(settings.field_weight_dl_number)),
            ("Date of Birth", float(s1.dob_similarity), float(settings.field_weight_dob)),
            ("Expiry Date", float(s1.expiry_similarity), float(settings.field_weight_expiry)),
            ("Face Biometrics", float(s2.vlm_confidence if s2.is_live else 0.0), float(settings.field_weight_face)),
            ("Vehicle Plate", float(s3.plate_similarity), float(settings.field_weight_plate)),
            (
                "Vehicle Color",
                float(s3.color_confidence if s3.color_matched else (1.0 if s3.color_matched else 0.0)),
                float(settings.field_weight_color),
            ),
        ]

        if settings.scoring_mode.lower() == "stage_weighted":
            s1_score = s1.confidence
            s2_score = s2.vlm_confidence if s2.is_live else 0.0
            s3_score = (s3.plate_similarity * settings.stage3_plate_weight) + (
                settings.stage3_color_weight if s3.color_matched else 0.0
            )

            s1_weight = settings.stage1_weight
            s2_weight = settings.stage2_weight
            s3_weight = settings.stage3_weight

            s1_points = s1_score * s1_weight
            s2_points = s2_score * s2_weight
            s3_points = s3_score * s3_weight

            composite_score = min(1.0, max(0.0, s1_points + s2_points + s3_points))
            formula_desc = f"({s1_weight:.2f} * Stage_1_OCR) + ({s2_weight:.2f} * Stage_2_Biometrics) + ({s3_weight:.2f} * Stage_3_Vehicle)"
        else:
            # "field_average": weighted arithmetic mean of all field matching percentages
            total_weight = sum(w for _, _, w in field_items)
            if total_weight > 0:
                weighted_sum = sum(score * weight for _, score, weight in field_items)
                composite_score = min(1.0, max(0.0, weighted_sum / total_weight))
            else:
                composite_score = 0.0
            formula_desc = f"Average across {len(field_items)} fields: ∑(field_score * weight) / ∑(weights)"

        # 3. Decision Matrix (Average >= Threshold -> APPROVED)
        threshold = settings.composite_approval_threshold
        threshold_passed = composite_score >= threshold
        all_stages_passed = s1.passed and s2.passed and s3.passed

        if threshold_passed:
            if settings.enforce_hard_rejections and hard_triggers:
                decision = DecisionEnum.REJECTED
                rejection_reasons.extend(hard_triggers)
                composite_score = min(composite_score, max(0.0, threshold - 0.01))
            elif settings.strict_stage_pass_required and not all_stages_passed:
                decision = DecisionEnum.REJECTED
                composite_score = min(composite_score, max(0.0, threshold - 0.01))
                if not s1.passed:
                    rejection_reasons.append(f"Stage 1 DL OCR failed: {s1.details}")
                if not s2.passed:
                    rejection_reasons.append(f"Stage 2 Face Biometrics failed: {s2.reasoning or s2.verdict}")
                if not s3.passed:
                    rejection_reasons.append(f"Stage 3 Vehicle ALPR failed: {s3.details}")
            else:
                decision = DecisionEnum.APPROVED
                # Record any non-blocking hard triggers as advisory manual review reasons
                manual_reasons.extend(hard_triggers)
        else:
            decision = DecisionEnum.REJECTED
            rejection_reasons.append(
                f"Composite average matching score {composite_score * 100:.1f}% is below approval threshold ({threshold * 100:.0f}%)"
            )
            rejection_reasons.extend(hard_triggers)
            if s1.name_similarity < settings.fuzzy_name_threshold:
                rejection_reasons.append(f"Full name similarity ({s1.name_similarity * 100:.1f}%) below threshold")
            if s1.number_similarity < settings.fuzzy_dl_threshold:
                rejection_reasons.append(f"DL number similarity ({s1.number_similarity * 100:.1f}%) below threshold")
            if s1.dob_similarity < (settings.fuzzy_dob_threshold / 100.0):
                rejection_reasons.append(f"DOB similarity ({s1.dob_similarity * 100:.1f}%) below threshold")
            if s2.vlm_confidence < settings.vlm_min_match_confidence:
                rejection_reasons.append(f"Facial biometric confidence ({s2.vlm_confidence * 100:.1f}%) below threshold")
            if s3.plate_similarity < settings.plate_match_threshold:
                rejection_reasons.append(f"Vehicle plate similarity ({s3.plate_similarity * 100:.1f}%) below threshold")

        rejection_reasons = list(dict.fromkeys(rejection_reasons))
        manual_reasons = list(dict.fromkeys(manual_reasons))

        field_scores_proof = [
            {
                "field": name,
                "score": round(float(score), 3),
                "score_percent": f"{score * 100:.1f}%",
                "weight": weight,
                "weighted_points": round(float(score * weight), 3),
            }
            for name, score, weight in field_items
        ]

        composite_proof = {
            "scoring_mode": settings.scoring_mode,
            "formula": formula_desc,
            "approval_threshold": threshold,
            "threshold_percentage": f"{threshold * 100:.0f}%",
            "composite_score": round(float(composite_score), 3),
            "composite_percentage": f"{composite_score * 100:.1f}%",
            "threshold_passed": bool(threshold_passed),
            "all_stages_passed": bool(all_stages_passed),
            "enforce_hard_rejections": settings.enforce_hard_rejections,
            "strict_stage_pass_required": settings.strict_stage_pass_required,
            "field_scores": field_scores_proof,
            "stage_scores": {
                "stage_1_dl_ocr": {
                    "stage_score": round(float(s1.confidence), 3),
                    "passed": s1.passed,
                },
                "stage_2_face_biometrics": {
                    "stage_score": round(float(s2.vlm_confidence), 3),
                    "passed": s2.passed,
                },
                "stage_3_vehicle_alpr": {
                    "stage_score": round(float(s3.confidence), 3),
                    "passed": s3.passed,
                },
            },
            "final_decision": decision.value,
            "justification_summary": (
                f"Calculated composite score {composite_score * 100:.1f}% >= threshold {threshold * 100:.0f}% -> APPROVED"
                if decision == DecisionEnum.APPROVED
                else f"Rejected: {'; '.join(rejection_reasons)}"
            ),
        }

        return decision, composite_score, rejection_reasons, manual_reasons, composite_proof, degraded


# Global instance
pipeline_engine = VerificationPipeline()
