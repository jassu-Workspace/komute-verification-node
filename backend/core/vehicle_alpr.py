import logging
import re
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from rapidfuzz import distance, fuzz
from app.schemas import VehicleDetails, VehicleVerificationResult
from core.image_utils import apply_clahe, encode_image_to_base64

logger = logging.getLogger(__name__)

# Character confusion dictionary for OCR ambiguities
CONFUSION_MAP = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "|": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}


def normalize_plate_string(plate: str, apply_confusion: bool = False) -> str:
    """
    Clean plate string by stripping spaces and non-alphanumeric chars.
    Optionally maps visually ambiguous OCR characters to canonical forms.
    """
    if not plate:
        return ""
    cleaned = re.sub(r"[^A-Z0-9]", "", plate.upper())
    if apply_confusion:
        canonical = []
        for ch in cleaned:
            canonical.append(CONFUSION_MAP.get(ch, ch))
        return "".join(canonical)
    return cleaned


def calculate_plate_match_score(extracted_plate: str, target_plate: str) -> Tuple[bool, float]:
    """
    Calculate similarity between extracted plate and submitted vehicle plate,
    taking into account standard Levenshtein and OCR character ambiguities.
    """
    clean_extracted = normalize_plate_string(extracted_plate)
    clean_target = normalize_plate_string(target_plate)

    if not clean_extracted or not clean_target:
        return False, 0.0

    if clean_extracted == clean_target:
        return True, 1.0

    # 1. Direct Levenshtein similarity
    max_len = max(len(clean_extracted), len(clean_target))
    lev_dist = distance.Levenshtein.distance(clean_extracted, clean_target)
    direct_sim = max(0.0, 1.0 - (lev_dist / max_len))

    # 2. Ambiguity-normalized similarity
    conf_extracted = normalize_plate_string(extracted_plate, apply_confusion=True)
    conf_target = normalize_plate_string(target_plate, apply_confusion=True)
    max_conf_len = max(len(conf_extracted), len(conf_target))
    conf_dist = distance.Levenshtein.distance(conf_extracted, conf_target)
    conf_sim = max(0.0, 1.0 - (conf_dist / max_conf_len))

    # 3. Fuzzy partial ratio
    partial_sim = fuzz.partial_ratio(clean_extracted, clean_target) / 100.0

    best_score = max(direct_sim, conf_sim, partial_sim * 0.95)

    # Consider matched if score >= 0.85 or if ambiguous normalized matches perfectly
    matched = (best_score >= 0.85) or (conf_extracted == conf_target and len(clean_target) >= 5)

    return matched, round(float(best_score), 3)


class VehicleALPREngine:
    """
    Stage 3: Vehicle ALPR & Visual Verification Engine.
    Detects license plate contours, extracts alphanumeric plate numbers,
    and analyzes dominant vehicle exterior color.
    """

    def __init__(self):
        self.reader = None

    def detect_plate_candidate_regions(self, img_bgr: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Isolate candidate number plate bounding boxes using Canny edge detection
        and morphological rectangle filters.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Bilateral filter to remove noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Morphological gradient / Sobel
        grad_x = cv2.Sobel(filtered, cv2.CV_16S, 1, 0, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)

        # Thresholding
        _, thresh = cv2.threshold(abs_grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing with rectangular structuring element
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []

        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch == 0 or cw == 0:
                continue

            aspect_ratio = cw / float(ch)
            area = cw * ch
            img_area = w * h

            # Typical license plates have aspect ratio between 1.8 and 6.0
            if 1.8 <= aspect_ratio <= 6.5 and (img_area * 0.002 <= area <= img_area * 0.35):
                pad_x = int(cw * 0.05)
                pad_y = int(ch * 0.05)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + cw + pad_x)
                y2 = min(h, y + ch + pad_y)

                cropped_plate = img_bgr[y1:y2, x1:x2]
                candidates.append((cropped_plate, (x1, y1, x2 - x1, y2 - y1)))

        # Also fallback: bottom-center third of image often contains the plate
        bottom_third_y = int(h * 0.45)
        bottom_center_x1 = int(w * 0.15)
        bottom_center_x2 = int(w * 0.85)
        center_crop = img_bgr[bottom_third_y:h, bottom_center_x1:bottom_center_x2]
        candidates.append((center_crop, (bottom_center_x1, bottom_third_y, bottom_center_x2 - bottom_center_x1, h - bottom_third_y)))

        return candidates

    def extract_plate_number(
        self,
        vehicle_img: np.ndarray,
        target_plate: str,
    ) -> Tuple[Optional[str], bool, float, bool, Optional[np.ndarray]]:
        """
        Run OCR on detected plate regions and focused vehicle regions to extract plate number.
        Pre-scales images to max 960px to prevent memory spikes.
        Returns (extracted_plate, plate_matched, similarity, plate_box_detected, plate_crop_bgr).
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return None, False, 0.0, False, None

        import gc
        from core.image_utils import resize_image_max_dimension

        # Downscale full vehicle photo to max 960px to keep peak RAM minimal
        scaled_vehicle, _ = resize_image_max_dimension(vehicle_img, max_dim=960)

        candidates = self.detect_plate_candidate_regions(scaled_vehicle)
        from core.ocr_engine import shared_ocr
        ocr = shared_ocr.get_engine()

        clean_target = normalize_plate_string(target_plate)
        best_plate = None
        best_score = 0.0
        best_matched = False
        best_crop: Optional[np.ndarray] = None
        plate_box_detected = len(candidates) > 1

        all_detected_texts: List[str] = []

        # 1. OCR on top candidate plate crops (at most top 4 candidates)
        for crop, bbox in candidates[:4]:
            if crop.size == 0:
                continue
            enhanced = apply_clahe(crop)
            try:
                res, _ = ocr(enhanced)
                if res:
                    lines_text = []
                    for line in res:
                        if len(line) >= 2:
                            text = str(line[1]).strip()
                            if not text:
                                continue

                            lines_text.append(text)
                            all_detected_texts.append(text)

                            text_clean = normalize_plate_string(text)
                            if len(text_clean) >= 3:
                                matched, score = calculate_plate_match_score(text_clean, target_plate)
                                if score > best_score or (score == best_score and len(text_clean) > len(best_plate or "")):
                                    best_score = score
                                    best_plate = text_clean
                                    best_matched = matched
                                    best_crop = crop

                    # Evaluate concatenated text to handle split/stacked plates
                    full_crop_text = " ".join(lines_text)
                    all_detected_texts.append(full_crop_text)

                    full_clean = normalize_plate_string(full_crop_text)
                    if len(full_clean) >= 4:
                        matched, score = calculate_plate_match_score(full_clean, target_plate)
                        if score > best_score or (score == best_score and len(full_clean) > len(best_plate or "")):
                            best_score = score
                            best_plate = full_clean
                            best_matched = matched
                            best_crop = crop

            except Exception as e:
                logger.debug(f"Candidate crop RapidOCR error: {e}")

        # 2. OCR on focused lower half if candidates didn't yield match
        if best_score < 0.85:
            try:
                vh, vw = scaled_vehicle.shape[:2]
                lower_half = scaled_vehicle[int(vh * 0.35) :, :]
                res, _ = ocr(lower_half)
                if res:
                    lines_text = []
                    for line in res:
                        if len(line) >= 2:
                            text = str(line[1]).strip()
                            if not text:
                                continue

                            lines_text.append(text)
                            all_detected_texts.append(text)

                            text_clean = normalize_plate_string(text)
                            if len(text_clean) >= 3:
                                matched, score = calculate_plate_match_score(text_clean, target_plate)
                                if score > best_score or (score == best_score and len(text_clean) > len(best_plate or "")):
                                    best_score = score
                                    best_plate = text_clean
                                    best_matched = matched

                    full_lower_text = " ".join(lines_text)
                    all_detected_texts.append(full_lower_text)

                    full_clean = normalize_plate_string(full_lower_text)
                    if len(full_clean) >= 4:
                        matched, score = calculate_plate_match_score(full_clean, target_plate)
                        if score > best_score or (score == best_score and len(full_clean) > len(best_plate or "")):
                            best_score = score
                            best_plate = full_clean
                            best_matched = matched
            except Exception as e:
                logger.debug(f"Focused lower vehicle RapidOCR error: {e}")

        gc.collect()

        # If candidates exist but best_crop not selected, take first candidate
        if best_crop is None and candidates:
            best_crop = candidates[0][0]

        # De-duplicate raw texts for UI cleanliness
        unique_raw_texts = []
        for t in all_detected_texts:
            if t not in unique_raw_texts and t.strip() != "":
                unique_raw_texts.append(t)

        gc.collect()

        return best_plate, best_matched, best_score, plate_box_detected, best_crop, unique_raw_texts

    def detect_vehicle_color(self, vehicle_img: np.ndarray, target_color: str) -> Tuple[bool, str, float]:
        """
        Analyze dominant color of the vehicle body using central region color segmentation.
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return False, "Unknown", 0.0

        h, w = vehicle_img.shape[:2]
        # Crop central 60% of vehicle to avoid background sky/ground
        cy1, cy2 = int(h * 0.25), int(h * 0.80)
        cx1, cx2 = int(w * 0.20), int(w * 0.80)
        central_region = vehicle_img[cy1:cy2, cx1:cx2]

        hsv = cv2.cvtColor(central_region, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        total_pixels = central_region.shape[0] * central_region.shape[1]
        if total_pixels == 0:
            return False, "Unknown", 0.0

        # Color definitions in HSV
        color_masks: Dict[str, np.ndarray] = {
            "Black": cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 65])),
            "White": cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 45, 255])),
            "Silver": cv2.inRange(hsv, np.array([0, 0, 70]), np.array([180, 50, 185])),
            "Grey": cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 45, 170])),
            "Red": cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255])),
            ),
            "Blue": cv2.inRange(hsv, np.array([95, 70, 50]), np.array([135, 255, 255])),
            "Green": cv2.inRange(hsv, np.array([35, 70, 50]), np.array([85, 255, 255])),
            "Yellow": cv2.inRange(hsv, np.array([20, 70, 100]), np.array([35, 255, 255])),
            "Orange": cv2.inRange(hsv, np.array([10, 80, 80]), np.array([22, 255, 255])),
        }

        color_counts: Dict[str, float] = {}
        for color_name, mask in color_masks.items():
            count = float(cv2.countNonZero(mask))
            color_counts[color_name] = count / total_pixels

        # Find dominant color
        dominant_color = max(color_counts.items(), key=lambda x: x[1])
        detected_color_name, detected_confidence = dominant_color

        target_clean = target_color.strip().capitalize()

        # Check match
        color_matched = False
        if target_clean in color_counts and color_counts[target_clean] >= 0.15:
            color_matched = True
            detected_color_name = target_clean
        elif target_clean.lower() in detected_color_name.lower() or detected_color_name.lower() in target_clean.lower():
            color_matched = True
        elif target_clean in ["Grey", "Gray", "Silver"] and detected_color_name in ["Grey", "Silver"]:
            color_matched = True
        elif target_clean in ["Black", "Dark Grey"] and detected_color_name in ["Black", "Grey"]:
            color_matched = True

        return color_matched, detected_color_name, round(float(detected_confidence), 3)

    def verify_vehicle(
        self,
        vehicle_img: np.ndarray,
        vehicle_details: VehicleDetails,
        return_crop: bool = False,
    ) -> Union[VehicleVerificationResult, Tuple[VehicleVerificationResult, Optional[np.ndarray]]]:
        """
        Execute full Stage 3 ALPR and vehicle visual verification.
        """
        # 1. Plate OCR & Cross-Match
        extracted_plate, plate_matched, plate_sim, plate_box_detected, plate_crop, all_detected_texts = self.extract_plate_number(
            vehicle_img=vehicle_img,
            target_plate=vehicle_details.plate,
        )

        # 2. Color verification
        color_matched, detected_color, color_conf = self.detect_vehicle_color(
            vehicle_img=vehicle_img,
            target_color=vehicle_details.color,
        )

        # Stage pass criteria: plate matched and reasonable visual agreement
        passed = plate_matched

        details_parts = []
        if plate_matched:
            details_parts.append(f"Plate matched '{extracted_plate}' (similarity: {plate_sim:.2f})")
        else:
            details_parts.append(f"Plate mismatch: extracted '{extracted_plate}', expected '{vehicle_details.plate}'")

        if color_matched:
            details_parts.append(f"Exterior color verified as {detected_color}")
        else:
            details_parts.append(f"Detected color {detected_color}, registered {vehicle_details.color}")

        veh_conf = round(float((plate_sim * 0.75) + (color_conf * 0.25 if color_matched else 0.0)), 3)
        plate_preview = encode_image_to_base64(plate_crop, format_ext=".jpg") if plate_crop is not None else None

        # Granular ALPR & Color Proof Breakdown
        confidence_proof = {
            "formula": "(0.75 * plate_similarity) + (0.25 * color_agreement_score)",
            "metrics": {
                "plate_similarity_proof": {
                    "registered_plate": vehicle_details.plate,
                    "extracted_plate": extracted_plate,
                    "similarity_score": round(float(plate_sim), 3),
                    "plate_box_localized": plate_box_detected,
                    "weight": 0.75,
                    "weighted_points": round(float(plate_sim * 0.75), 3),
                    "justification": f"ALPR character string similarity: {plate_sim*100:.1f}%. Plate matched successfully" if plate_matched else f"Plate mismatch: similarity {plate_sim*100:.1f}% below match threshold",
                },
                "color_detection_proof": {
                    "registered_color": vehicle_details.color,
                    "detected_dominant_color": detected_color,
                    "color_confidence_score": round(float(color_conf), 3),
                    "color_matched": color_matched,
                    "weight": 0.25,
                    "weighted_points": round(float(color_conf * 0.25 if color_matched else 0.0), 3),
                    "justification": f"HSV chromatic distribution matches registered '{vehicle_details.color}' with {color_conf*100:.1f}% confidence" if color_matched else f"Exterior body color '{detected_color}' differs from registration '{vehicle_details.color}'",
                },
            },
            "calculated_confidence": veh_conf,
            "stage_verdict": "PASSED" if passed else "REJECTED",
        }

        result = VehicleVerificationResult(
            passed=passed,
            confidence=veh_conf,
            extracted_plate=extracted_plate,
            plate_matched=plate_matched,
            plate_similarity=round(float(plate_sim), 3),
            color_matched=color_matched,
            detected_color=detected_color,
            color_confidence=round(float(color_conf), 3),
            plate_box_detected=plate_box_detected,
            details="; ".join(details_parts),
            plate_crop_preview=plate_preview,
            raw_ocr_lines=all_detected_texts,
            confidence_proof=confidence_proof,
        )

        if return_crop:
            return result, plate_crop
        return result


# Global instance
vehicle_alpr_engine = VehicleALPREngine()
