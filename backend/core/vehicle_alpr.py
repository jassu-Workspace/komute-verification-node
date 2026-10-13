import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from rapidfuzz import distance, fuzz
from app.schemas import VehicleDetails, VehicleVerificationResult
from app.config import settings
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
    "Y": "V",
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
    Enforces whole-plate matching: short substrings (e.g. 4-character fragments of a 9-character plate)
    are strictly penalized by coverage and cannot falsely trigger a match.
    """
    clean_extracted = normalize_plate_string(extracted_plate)
    clean_target = normalize_plate_string(target_plate)

    if not clean_extracted or not clean_target:
        return False, 0.0

    if clean_extracted == clean_target:
        return True, 1.0

    # 1. Direct Levenshtein similarity
    max_len = max(len(clean_extracted), len(clean_target))
    min_len = min(len(clean_extracted), len(clean_target))
    lev_dist = distance.Levenshtein.distance(clean_extracted, clean_target)
    direct_sim = max(0.0, 1.0 - (lev_dist / max_len))

    # 2. Ambiguity-normalized similarity
    conf_extracted = normalize_plate_string(extracted_plate, apply_confusion=True)
    conf_target = normalize_plate_string(target_plate, apply_confusion=True)
    max_conf_len = max(len(conf_extracted), len(conf_target))
    conf_dist = distance.Levenshtein.distance(conf_extracted, conf_target)
    conf_sim = max(0.0, 1.0 - (conf_dist / max_conf_len))

    # 3. Whole-plate coverage and fuzzy partial ratio
    len_coverage = min_len / max_len if max_len > 0 else 0.0

    if clean_target in clean_extracted:
        # Full target plate is embedded inside longer OCR text (e.g. "ONTARIODL7CQ1939")
        partial_sim = 1.0
    elif clean_extracted in clean_target and len_coverage < 0.80:
        # Incomplete fragment only (e.g. 'L7CQ' vs 'DL7CQ1939') - strictly penalize, no partial bonus
        partial_sim = direct_sim
    elif len_coverage >= 0.80:
        # High coverage (at most 1 character missing/extra), allow scaled partial ratio
        partial_sim = (fuzz.partial_ratio(clean_extracted, clean_target) / 100.0) * len_coverage
    else:
        partial_sim = direct_sim

    best_score = max(direct_sim, conf_sim, partial_sim)

    # Require BOTH:
    # 1. Similarity score >= threshold
    # 2. Whole plate matched: either clean_target is in clean_extracted, or length coverage >= 0.80
    matched = (best_score >= settings.plate_match_threshold) and (
        (clean_target in clean_extracted) or (len_coverage >= 0.80)
    )
    if not matched and conf_extracted == conf_target and len(clean_target) >= 5:
        matched = True
        best_score = 1.0

    return matched, round(float(best_score), 3)


# Known provincial slogans and headers on Canadian / North American plates
PROVINCIAL_SLOGANS = {
    "ALBERTA", "WILD ROSE COUNTRY", "WILD ROSE",
    "ONTARIO", "YOURS TO DISCOVER", "A PLACE TO GROW",
    "BRITISH COLUMBIA", "BEAUTIFUL BRITISH COLUMBIA", "BEST PLACE ON EARTH",
    "MANITOBA", "FRIENDLY MANITOBA", "100 000 LAKES", "100000 LAKES",
    "QUEBEC", "JE ME SOUVIENS",
    "SASKATCHEWAN", "LAND OF LIVING SKIES",
    "NOVA SCOTIA", "CANADA'S OCEAN PLAYGROUND", "CANADAS OCEAN PLAYGROUND",
    "NEW BRUNSWICK", "NOUVEAU BRUNSWICK",
    "PRINCE EDWARD ISLAND", "PEI", "BIRTHPLACE OF CONFEDERATION",
    "NEWFOUNDLAND", "LABRADOR",
    "YUKON", "THE KLONDIKE",
    "NORTHWEST TERRITORIES", "SPECTACULAR",
    "NUNAVUT", "CANADA", "COMMERCIAL", "DEALER", "TRAILER",
}


def is_slogan_noise(text: str) -> bool:
    """Check if detected text is a pure provincial header or slogan without digits."""
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not clean or any(ch.isdigit() for ch in clean):
        return False
    for s in PROVINCIAL_SLOGANS:
        s_clean = re.sub(r"[^A-Z0-9]", "", s)
        if clean == s_clean or s_clean in clean:
            return True
    return False


def group_horizontal_tokens(ocr_lines: List[Any], source_img: Optional[np.ndarray] = None) -> List[Tuple[str, Optional[np.ndarray]]]:
    """
    Given OCR lines [(box, text, conf), ...], groups tokens sharing approximately
    the same horizontal baseline and reconstructs multi-word / hyphen-separated plate numbers.
    Also produces cropped plate patches for the combined tokens.
    """
    if not ocr_lines:
        return []

    items = []
    for item in ocr_lines:
        if len(item) < 2:
            continue
        box = item[0]
        text = str(item[1]).strip()
        conf = float(item[2]) if len(item) >= 3 else 1.0
        if not text:
            continue
        pts = box
        y_coords = [p[1] for p in pts]
        x_coords = [p[0] for p in pts]
        y_min, y_max = min(y_coords), max(y_coords)
        x_min, x_max = min(x_coords), max(x_coords)
        y_center = (y_min + y_max) / 2.0
        h = max(1.0, y_max - y_min)
        items.append({
            "text": text,
            "conf": conf,
            "y_center": y_center,
            "y_min": y_min,
            "y_max": y_max,
            "x_min": x_min,
            "x_max": x_max,
            "h": h,
            "box": box,
        })

    sorted_items = sorted(enumerate(items), key=lambda x: (x[1]["y_center"], x[1]["x_min"]))
    lines: List[List[Dict[str, Any]]] = []
    used = set()

    for i, item in sorted_items:
        if i in used:
            continue
        cur_line = [item]
        used.add(i)
        for j, other in sorted_items:
            if j in used:
                continue
            h_ref = (item["h"] + other["h"]) / 2.0
            y_diff = abs(item["y_center"] - other["y_center"])
            y_overlap = max(0.0, min(item["y_max"], other["y_max"]) - max(item["y_min"], other["y_min"]))
            if y_diff <= 0.60 * h_ref or (y_overlap / h_ref) >= 0.40:
                cur_line.append(other)
                used.add(j)

        cur_line.sort(key=lambda x: x["x_min"])
        lines.append(cur_line)

    results: List[Tuple[str, Optional[np.ndarray]]] = []
    ih, iw = (source_img.shape[:2]) if source_img is not None else (0, 0)

    for line in lines:
        words = [x["text"] for x in line]
        boxes = [x["box"] for x in line]

        # 1. Individual word tokens
        for idx, w in enumerate(words):
            crop = None
            if source_img is not None:
                b = boxes[idx]
                pts = np.array(b, dtype=np.int32)
                bx, by, bw, bh = cv2.boundingRect(pts)
                pad = 4
                crop = source_img[max(0, by - pad):min(ih, by + bh + pad), max(0, bx - pad):min(iw, bx + bw + pad)]
            results.append((w, crop))

        # 2. Adjacent horizontal n-grams (e.g. ['CKZ', '3449'] -> 'CKZ-3449', 'CKZ 3449', 'CKZ3449')
        n = len(words)
        if n >= 2:
            for k in range(2, n + 1):
                for start_idx in range(n - k + 1):
                    slice_words = words[start_idx:start_idx + k]
                    slice_boxes = boxes[start_idx:start_idx + k]
                    crop = None
                    if source_img is not None:
                        all_pts = np.vstack([np.array(b, dtype=np.int32) for b in slice_boxes])
                        bx, by, bw, bh = cv2.boundingRect(all_pts)
                        pad = 6
                        crop = source_img[max(0, by - pad):min(ih, by + bh + pad), max(0, bx - pad):min(iw, bx + bw + pad)]
                    results.append(("".join(slice_words), crop))
                    results.append(("-".join(slice_words), crop))
                    results.append((" ".join(slice_words), crop))

        # 3. Non-slogan tokens combination (skips embedded provincial headers/slogans)
        non_slogan_items = [x for x in line if not is_slogan_noise(x["text"])]
        if 2 <= len(non_slogan_items) < n:
            ns_words = [x["text"] for x in non_slogan_items]
            ns_boxes = [x["box"] for x in non_slogan_items]
            crop = None
            if source_img is not None:
                all_pts = np.vstack([np.array(b, dtype=np.int32) for b in ns_boxes])
                bx, by, bw, bh = cv2.boundingRect(all_pts)
                pad = 6
                crop = source_img[max(0, by - pad):min(ih, by + bh + pad), max(0, bx - pad):min(iw, bx + bw + pad)]
            results.append(("".join(ns_words), crop))
            results.append(("-".join(ns_words), crop))
            results.append((" ".join(ns_words), crop))

    return results


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
        Isolate candidate number plate bounding boxes using edge detection
        and multi-scale morphological filters. Supports both full vehicle photos
        and close-up license plate uploads.
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

        # Multi-scale morphological closing kernels (standard 17x3, 27x5, 35x5, and 45x7 to bridge wide hyphen/emblem gaps)
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (27, 5)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (45, 7)),
        ]

        raw_boxes: List[Tuple[int, int, int, int]] = []
        img_area = w * h

        for rect_kernel in kernels:
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                if ch == 0 or cw == 0:
                    continue

                aspect_ratio = cw / float(ch)
                area = cw * ch

                # License plates typically have aspect ratio between 1.2 and 7.0
                # Expanded area ceiling to 0.95 to accommodate close-up / cropped plate photos
                if 1.2 <= aspect_ratio <= 7.0 and (img_area * 0.002 <= area <= img_area * 0.95):
                    raw_boxes.append((x, y, cw, ch))

        # Sort candidate boxes by area descending and score proximity to standard plate aspect ratio (~2.0 - 4.5)
        def box_rank(b):
            _, _, cw, ch = b
            ar = cw / float(ch)
            ar_diff = abs(ar - 2.5)
            return (cw * ch) / (1.0 + ar_diff)

        raw_boxes.sort(key=box_rank, reverse=True)

        # Non-maximum suppression / overlap deduplication
        deduped_boxes: List[Tuple[int, int, int, int]] = []
        for b in raw_boxes:
            x, y, cw, ch = b
            overlap = False
            for dx, dy, dcw, dch in deduped_boxes:
                # Check intersection over union
                ix = max(x, dx)
                iy = max(y, dy)
                iw_box = min(x + cw, dx + dcw) - ix
                ih_box = min(y + ch, dy + dch) - iy
                if iw_box > 0 and ih_box > 0:
                    inter_area = iw_box * ih_box
                    union_area = (cw * ch) + (dcw * dch) - inter_area
                    if (inter_area / union_area) > 0.45:
                        overlap = True
                        break
            if not overlap:
                deduped_boxes.append(b)

        candidates: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []
        for x, y, cw, ch in deduped_boxes[:6]:
            pad_x = int(cw * 0.06)
            pad_y = int(ch * 0.06)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)

            cropped_plate = img_bgr[y1:y2, x1:x2]
            candidates.append((cropped_plate, (x1, y1, x2 - x1, y2 - y1)))

        # Fallback 1: Center 80% crop (captures centered plates on close-up or tight vehicle photos)
        c_y1, c_y2 = int(h * 0.10), int(h * 0.90)
        c_x1, c_x2 = int(w * 0.05), int(w * 0.95)
        candidates.append((img_bgr[c_y1:c_y2, c_x1:c_x2], (c_x1, c_y1, c_x2 - c_x1, c_y2 - c_y1)))

        # Fallback 2: Lower-center crop (for full vehicle photos where plate is on the front/rear bumper)
        bottom_y1 = int(h * 0.35)
        bottom_x1 = int(w * 0.10)
        bottom_x2 = int(w * 0.90)
        candidates.append((img_bgr[bottom_y1:h, bottom_x1:bottom_x2], (bottom_x1, bottom_y1, bottom_x2 - bottom_x1, h - bottom_y1)))

        return candidates

    def extract_plate_number(
        self,
        vehicle_img: np.ndarray,
        target_plate: str,
    ) -> Tuple[Optional[str], bool, float, bool, Optional[np.ndarray], List[str]]:
        """
        Run OCR on full vehicle image and detected plate candidate regions to extract plate number.
        Employs multi-pass extraction:
          1. Direct full-image OCR + spatial horizontal token reconstruction
          2. CLAHE contrast-enhanced full-image OCR
          3. Multi-scale candidate contour crops
          4. Balanced bumper & central crops
        Pre-scales images to max 960px to prevent memory spikes.
        Returns (extracted_plate, plate_matched, similarity, plate_box_detected, plate_crop_bgr, unique_raw_texts).
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return None, False, 0.0, False, None, []

        import gc
        from core.image_utils import resize_image_max_dimension

        # Downscale full vehicle photo to max 960px to keep peak RAM minimal
        scaled_vehicle, _ = resize_image_max_dimension(vehicle_img, max_dim=960)

        from core.ocr_engine import shared_ocr
        ocr = shared_ocr.get_engine()

        clean_target = normalize_plate_string(target_plate)
        best_plate = None
        best_score = 0.0
        best_matched = False
        best_crop: Optional[np.ndarray] = None
        plate_box_detected = False

        all_detected_texts: List[str] = []

        def evaluate_candidate(cand_text: str, crop_img: Optional[np.ndarray]):
            nonlocal best_score, best_plate, best_matched, best_crop
            if not cand_text:
                return

            text_clean = normalize_plate_string(cand_text)
            if len(text_clean) < 3:
                return

            # Suppress pure provincial slogans (e.g. 'WILDROSECOUNTRY') from winning over real plates
            if is_slogan_noise(cand_text):
                return

            matched, score = calculate_plate_match_score(text_clean, target_plate)

            # Determine best representation of the plate
            if text_clean == clean_target:
                # Clean match matches target plate; preserve target's formatted string (e.g. 'CKZ-3449')
                effective_plate = target_plate.strip().upper()
            elif clean_target in text_clean:
                effective_plate = target_plate.strip().upper()
            elif cand_text.strip().replace(" ", "-").upper() == target_plate.strip().upper():
                effective_plate = target_plate.strip().upper()
            else:
                effective_plate = cand_text.strip().upper() if len(cand_text.strip()) <= 12 else text_clean

            is_better = False
            if score > best_score:
                is_better = True
            elif score == best_score and best_score > 0.0:
                clean_eff = normalize_plate_string(effective_plate)
                clean_cur = normalize_plate_string(best_plate or "")
                # 1. Exact match to clean_target always beats non-exact match
                if clean_eff == clean_target and clean_cur != clean_target:
                    is_better = True
                elif clean_cur == clean_target and clean_eff != clean_target:
                    is_better = False
                else:
                    # 2. Closer length to target plate wins (eliminates noise words like 'AND')
                    new_diff = abs(len(clean_eff) - len(clean_target))
                    cur_diff = abs(len(clean_cur) - len(clean_target))
                    if new_diff < cur_diff:
                        is_better = True

            if is_better:
                best_score = score
                best_plate = effective_plate
                best_matched = matched
                if crop_img is not None and crop_img.size > 0:
                    best_crop = crop_img

        def process_ocr_results(res: List[Any], src_img: Optional[np.ndarray]):
            if not res:
                return
            lines_text = []
            for line in res:
                if len(line) >= 2:
                    t = str(line[1]).strip()
                    if t:
                        lines_text.append(t)
                        all_detected_texts.append(t)

            # Evaluate horizontally grouped tokens (single words + adjacent n-grams)
            grouped_candidates = group_horizontal_tokens(res, src_img)
            for cand_txt, cand_crop in grouped_candidates:
                all_detected_texts.append(cand_txt)
                evaluate_candidate(cand_txt, cand_crop)

            # Also evaluate concatenated text of non-slogan lines
            non_slogan_lines = [l for l in lines_text if not is_slogan_noise(l)]
            if non_slogan_lines:
                concat_text = " ".join(non_slogan_lines)
                all_detected_texts.append(concat_text)
                evaluate_candidate(concat_text, src_img)

        # =========================================================================
        # PASS 1: Direct Full-Image OCR Pass
        # If user uploaded a close-up plate photo or cropped vehicle, the plate is visible immediately.
        # =========================================================================
        try:
            full_res, _ = ocr(scaled_vehicle)
            if full_res:
                process_ocr_results(full_res, scaled_vehicle)
                if best_score >= 0.70:
                    plate_box_detected = True
        except Exception as e:
            logger.debug(f"Pass 1 full-image RapidOCR error: {e}")

        # =========================================================================
        # PASS 2: Contrast-Enhanced Full-Image Pass (CLAHE)
        # =========================================================================
        if best_score < settings.plate_match_threshold:
            try:
                enhanced_full = apply_clahe(scaled_vehicle)
                full_clahe_res, _ = ocr(enhanced_full)
                if full_clahe_res:
                    process_ocr_results(full_clahe_res, scaled_vehicle)
                    if best_score >= 0.70:
                        plate_box_detected = True
            except Exception as e:
                logger.debug(f"Pass 2 full-image CLAHE RapidOCR error: {e}")

        # =========================================================================
        # PASS 3: Candidate Contour Bounding Boxes (for full-vehicle photos)
        # =========================================================================
        if best_score < settings.plate_match_threshold:
            candidates = self.detect_plate_candidate_regions(scaled_vehicle)
            if len(candidates) > 2:
                plate_box_detected = True

            for crop, bbox in candidates[:6]:
                if crop.size == 0:
                    continue

                # 3a. Direct candidate crop
                try:
                    c_res, _ = ocr(crop)
                    if c_res:
                        process_ocr_results(c_res, crop)
                        if best_matched:
                            break
                except Exception as e:
                    logger.debug(f"Candidate crop RapidOCR error: {e}")

                # 3b. CLAHE enhanced crop
                if best_score < settings.plate_match_threshold:
                    enhanced = apply_clahe(crop)
                    try:
                        c_res, _ = ocr(enhanced)
                        if c_res:
                            process_ocr_results(c_res, crop)
                            if best_matched:
                                break
                    except Exception as e:
                        logger.debug(f"Candidate crop CLAHE RapidOCR error: {e}")

                # 3c. 2x upscaled crop for small or distant plates
                ch, cw = crop.shape[:2]
                if best_score < settings.plate_match_threshold and (ch < 150 or cw < 300):
                    try:
                        upscaled = cv2.resize(crop, (cw * 2, ch * 2), interpolation=cv2.INTER_CUBIC)
                        upscaled_enh = apply_clahe(upscaled)
                        c_res, _ = ocr(upscaled_enh)
                        if c_res:
                            process_ocr_results(c_res, crop)
                            if best_matched:
                                break
                    except Exception as e:
                        logger.debug(f"Candidate crop upscaled RapidOCR error: {e}")

        # If candidates were found and best_crop not set, default to first candidate crop
        if best_crop is None:
            cands = self.detect_plate_candidate_regions(scaled_vehicle)
            if cands:
                best_crop = cands[0][0]

        # De-duplicate raw texts for UI cleanliness
        unique_raw_texts: List[str] = []
        for t in all_detected_texts:
            t_str = str(t).strip()
            if t_str and t_str not in unique_raw_texts:
                unique_raw_texts.append(t_str)

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
        if target_clean in color_counts and color_counts[target_clean] >= settings.color_agreement_min:
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
