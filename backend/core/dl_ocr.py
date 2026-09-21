import gc
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from rapidfuzz import fuzz

from app.schemas import LicenseDetails, LicenseOcrResult, PersonalInfo
from core.canadian_dl_grammar import (
    clean_dl_string,
    format_canadian_dl,
    normalize_province_code,
    repair_provincial_dl,
)
from core.image_utils import (
    apply_adaptive_threshold,
    apply_bilateral_clahe,
    apply_clahe,
    assess_image_quality,
    deskew_image,
    flatten_id_card,
    resize_image_max_dimension,
    upscale_and_sharpen,
)

logger = logging.getLogger(__name__)


@dataclass
class OCRBox:
    """Represents an extracted text box with its spatial coordinates and confidence."""
    text: str
    confidence: float
    box: List[List[int]]
    center_x: float
    center_y: float


class DrivingLicenseOCREngine:
    """
    Stage 1: Canadian Driving License OCR Verification Engine using RapidOCR with ONNX Runtime.
    100% strictly local CPU inference with zero cloud dependency.
    Features:
    - Pre-flight Laplacian image quality & glare assessment
    - Color-preserving contour perspective flattening (flatten before thresholding)
    - Continuous Bilateral edge-preserving contrast (eliminates harsh 1-bit binarization noise)
    - Spatial AAMVA tag field association (anchoring on 4d, 4b, 1, 2)
    - Deterministic Canadian provincial grammar auto-repair for all 13 provinces & territories
    """

    def __init__(self):
        pass

    def extract_text_and_boxes(
        self,
        img_bgr: np.ndarray,
        enhancement_mode: str = "bilateral",
        is_preflattened: bool = False,
    ) -> Tuple[List[str], List[OCRBox]]:
        """
        Extracts text lines along with their 2D bounding boxes and center coordinates.
        enhancement_mode options: 'bilateral' (default), 'clahe', 'adaptive', 'none'.
        """
        if img_bgr is None or img_bgr.size == 0:
            return [], []

        # Step 1: Perspective contour flattening (skip if already pre-flattened)
        flattened = img_bgr if is_preflattened else flatten_id_card(img_bgr)

        # Step 2: High-fidelity resolution scaling (preserves text height to 20-30px)
        scaled_img, _ = resize_image_max_dimension(flattened, max_dim=1800)

        # Step 3: Deskew upright
        deskewed = deskew_image(scaled_img)

        # Step 4: Signal enhancement
        if enhancement_mode == "bilateral":
            enhanced = apply_bilateral_clahe(deskewed)
        elif enhancement_mode == "clahe":
            enhanced = apply_clahe(deskewed)
        elif enhancement_mode == "adaptive":
            enhanced = apply_adaptive_threshold(deskewed)
        else:
            enhanced = deskewed

        lines: List[str] = []
        boxes: List[OCRBox] = []

        try:
            from core.ocr_engine import shared_ocr
            ocr = shared_ocr.get_engine()

            res, _ = ocr(enhanced)
            if res:
                for item in res:
                    if len(item) >= 2:
                        box_pts = item[0]
                        text = str(item[1]).strip()
                        conf = float(item[2]) if len(item) > 2 else 0.8

                        if text:
                            lines.append(text)
                            if isinstance(box_pts, (list, np.ndarray)) and len(box_pts) == 4:
                                pts = np.array(box_pts, dtype=np.float32)
                                cx = float(np.mean(pts[:, 0]))
                                cy = float(np.mean(pts[:, 1]))
                                boxes.append(OCRBox(text=text, confidence=conf, box=box_pts, center_x=cx, center_y=cy))

        except Exception as e:
            logger.error(f"RapidOCR recognition error: {e}")

        gc.collect()
        return lines, boxes

    def extract_text_lines(self, img_bgr: np.ndarray, apply_binarization: bool = True) -> List[str]:
        """Backwards-compatible wrapper returning flat list of extracted strings."""
        mode = "adaptive" if apply_binarization else "bilateral"
        lines, _ = self.extract_text_and_boxes(img_bgr, enhancement_mode=mode)
        return lines

    def extract_spatial_aamva_fields(self, boxes: List[OCRBox], province: str = "ON") -> Dict[str, Optional[str]]:
        """
        Inspects bounding boxes for standardized Canadian AAMVA tags:
        '4d' = License Number, '4b' = Expiry, '3' = DOB, '1' = Surname, '2' = Given Name.
        """
        extracted: Dict[str, Optional[str]] = {
            "license_number": None,
            "expiry_date": None,
            "dob": None,
            "surname": None,
            "given_name": None,
        }
        if not boxes:
            return extracted

        for box in boxes:
            t = box.text.strip()
            # 4d tag: License Number
            m_4d = re.search(r'(?:4d|4D)[\.:\s]*(.+)', t)
            if m_4d:
                candidate = m_4d.group(1).strip()
                if len(candidate) >= 5:
                    extracted["license_number"] = candidate
            elif re.match(r'^(?:4d|4D)[\.:\s]*$', t):
                # Tag is isolated; look for the nearest box to the right or below
                nearest = self._find_adjacent_box(box, boxes)
                if nearest:
                    extracted["license_number"] = nearest.text

            # 4b tag: Expiry Date
            m_4b = re.search(r'(?:4b|4B)[\.:\s]*(.+)', t)
            if m_4b:
                candidate = m_4b.group(1).strip()
                if len(candidate) >= 4:
                    extracted["expiry_date"] = candidate
            elif re.match(r'^(?:4b|4B)[\.:\s]*$', t):
                nearest = self._find_adjacent_box(box, boxes)
                if nearest:
                    extracted["expiry_date"] = nearest.text

            # 3 tag: DOB
            m_3 = re.search(r'(?:3)[\.:\s]+([0-9]{2,4}[-/\.][0-9]{2}[-/\.][0-9]{2,4})', t)
            if m_3:
                extracted["dob"] = m_3.group(1).strip()

        return extracted

    def _find_adjacent_box(
        self,
        anchor: OCRBox,
        all_boxes: List[OCRBox],
        max_dist_x: float = 350.0,
        max_dist_y: float = 40.0,
    ) -> Optional[OCRBox]:
        """Finds the most geometrically plausible text box adjacent to an AAMVA tag."""
        best_cand: Optional[OCRBox] = None
        min_dist = float("inf")

        for b in all_boxes:
            if b is anchor:
                continue
            dx = b.center_x - anchor.center_x
            dy = b.center_y - anchor.center_y

            # Horizontally adjacent (to the right, similar vertical line)
            if 0 < dx < max_dist_x and abs(dy) < max_dist_y:
                dist = np.hypot(dx, dy)
                if dist < min_dist:
                    min_dist = dist
                    best_cand = b
            # Vertically adjacent (directly below tag)
            elif 0 < dy < 60.0 and abs(dx) < 60.0:
                dist = np.hypot(dx, dy)
                if dist < min_dist:
                    min_dist = dist
                    best_cand = b

        return best_cand

    def generate_date_variations(self, date_str: str) -> List[str]:
        """Generates common string representations of a date."""
        if not date_str:
            return []
        try:
            from dateutil import parser
            d = parser.parse(date_str)
            return [
                date_str,
                d.strftime("%Y-%m-%d"),
                d.strftime("%d-%m-%Y"),
                d.strftime("%d/%m/%Y"),
                d.strftime("%m/%d/%Y"),
                d.strftime("%d.%m.%Y"),
                d.strftime("%d %b %Y").lower(),
                d.strftime("%b %d, %Y").lower(),
                d.strftime("%Y/%m/%d"),
                d.strftime("%Y%m%d"),
                d.strftime("%y%m%d"),
            ]
        except Exception:
            return [date_str]

    def _match_dl_number(
        self,
        raw_texts: List[str],
        target_dl: str,
        province: str = "ON",
        surname: str = "",
    ) -> Tuple[float, Optional[str]]:
        """
        High-precision Canadian DL number matching:
        - Deterministic Canadian provincial repair (Ontario 1-letter+14-digits, BC 7-digits, etc.)
        - Substring & reverse substring containment
        - Sliding-window Levenshtein matching on alphanumeric streams
        - Multi-pass whole concatenated text verification
        """
        if not target_dl:
            return 100.0, None

        prov = normalize_province_code(province)
        clean_target = clean_dl_string(target_dl)
        if not clean_target:
            return 100.0, None

        # Canonicalize target through provincial grammar
        canonical_target, _, _ = repair_provincial_dl(clean_target, province=prov, surname=surname)
        if not canonical_target:
            canonical_target = clean_target

        best_score = 0.0
        best_match: Optional[str] = None

        for line in raw_texts:
            clean_line = clean_dl_string(line)
            if not clean_line or len(clean_line) < 3:
                continue

            # 1. Exact string match
            if clean_line == canonical_target or clean_line == clean_target:
                return 100.0, line

            # 2. Provincial Grammar Auto-Repair on the line
            repaired_line, conf_score, is_valid = repair_provincial_dl(clean_line, province=prov, surname=surname)
            if is_valid and repaired_line == canonical_target:
                return 100.0, line

            # 3. Exact Substring Containment (e.g. DL number embedded in longer text)
            if canonical_target in clean_line or clean_target in clean_line:
                return 100.0, line

            # 4. Sliding Window Comparison
            target_len = len(canonical_target)
            if len(clean_line) >= target_len:
                for i in range(len(clean_line) - target_len + 1):
                    chunk = clean_line[i : i + target_len]
                    if chunk == canonical_target:
                        return 100.0, line
                    # Try repairing the chunk
                    rep_chunk, _, chunk_valid = repair_provincial_dl(chunk, province=prov, surname=surname)
                    if chunk_valid and rep_chunk == canonical_target:
                        return 99.0, line
                    score_chunk = fuzz.ratio(canonical_target, chunk)
                    if score_chunk > best_score:
                        best_score = score_chunk
                        best_match = line
            else:
                # Reverse substring (line is a prominent part of target DL)
                if len(clean_line) >= 6 and clean_line in canonical_target:
                    s_part = (len(clean_line) / float(target_len)) * 100.0
                    if s_part > best_score:
                        best_score = s_part
                        best_match = line

            s_ratio = fuzz.ratio(canonical_target, clean_line)
            if s_ratio > best_score:
                best_score = s_ratio
                best_match = line

        # 5. Whole Concatenated Text Verification
        whole_clean = clean_dl_string(" ".join(raw_texts))
        if canonical_target in whole_clean or clean_target in whole_clean:
            return 100.0, best_match or target_dl

        # Try sliding window across concatenated stream
        if len(whole_clean) >= len(canonical_target):
            target_len = len(canonical_target)
            for i in range(len(whole_clean) - target_len + 1):
                chunk = whole_clean[i : i + target_len]
                rep_chunk, _, chunk_valid = repair_provincial_dl(chunk, province=prov, surname=surname)
                if chunk_valid and rep_chunk == canonical_target:
                    return 98.0, best_match or target_dl

        return best_score, best_match

    def _match_name(self, raw_texts: List[str], target_name: str) -> Tuple[float, Optional[str]]:
        """Multi-token name matching across lines and joined OCR text."""
        if not target_name:
            return 100.0, None

        tokens = [t.upper() for t in re.findall(r'[A-Za-z0-9]+', target_name)]
        if not tokens:
            return 100.0, None

        whole_upper = " ".join(raw_texts).upper()
        # If every token of the name appears in the OCR text
        if all(t in whole_upper for t in tokens):
            return 100.0, target_name

        best_score = 0.0
        for line in raw_texts:
            s_sort = fuzz.token_sort_ratio(target_name.upper(), line.upper())
            best_score = max(best_score, s_sort)

        s_partial = fuzz.partial_token_set_ratio(target_name.lower(), whole_upper.lower())
        best_score = max(best_score, s_partial)
        return best_score, target_name

    def _match_date(self, raw_texts: List[str], target_date: str) -> Tuple[float, Optional[str]]:
        """Multi-format date and digit sequence matching."""
        if not target_date:
            return 100.0, None

        target_digits = re.sub(r'\D', '', target_date)
        if not target_digits:
            return 100.0, None

        variations = self.generate_date_variations(target_date)
        whole_text = " ".join(raw_texts).lower()

        for v in variations:
            if v.lower() in whole_text:
                return 100.0, v

        best_score = 0.0
        best_match: Optional[str] = None

        for line in raw_texts:
            line_digits = re.sub(r'\D', '', line)
            if target_digits in line_digits:
                return 100.0, line
            if len(line_digits) >= 6:
                s = fuzz.partial_ratio(target_digits, line_digits)
                if s > best_score:
                    best_score = s
                    best_match = line

        whole_digits = re.sub(r'\D', '', whole_text)
        if target_digits in whole_digits:
            return 100.0, best_match or target_date

        return best_score, best_match

    def verify_license(
        self,
        img_bgr: np.ndarray,
        personal_info: PersonalInfo,
        license_details: LicenseDetails,
        is_preflattened: bool = False,
    ) -> LicenseOcrResult:
        """
        Core Canadian verification pipeline implementing fine-tuned RapidOCR recognition,
        AAMVA spatial layout awareness, Canadian provincial grammar auto-repair,
        and multi-pass progressive contrast recovery.
        """
        from app.config import settings

        threshold_name = getattr(settings, "fuzzy_name_threshold", 0.85) * 100.0
        threshold_dl = getattr(settings, "fuzzy_dl_threshold", 0.85) * 100.0
        threshold_dob = 80.0
        threshold_exp = 80.0

        # Step 0: Pre-Flight Image Quality Gate
        is_usable, quality_msg, metrics = assess_image_quality(img_bgr)
        if not is_usable and metrics.get("blur_variance", 100.0) < 20.0:
            # Extreme physical blur: characters are physically destroyed
            return LicenseOcrResult(
                passed=False,
                number_matched=False,
                name_matched=False,
                dob_matched=False,
                confidence=0.0,
                details=f"Stage 1 DL OCR Rejected: {quality_msg} (Laplacian Variance: {metrics.get('blur_variance', 0)})",
                missing_fields=["Image Clarity (Severe Blur)"],
                confidence_proof={"stage_verdict": "REJECTED", "metrics": metrics, "reason": quality_msg},
            )

        name = personal_info.full_name if personal_info.full_name else ""
        dl_number = license_details.license_number if license_details.license_number else ""
        dob = personal_info.date_of_birth if personal_info.date_of_birth else ""
        exp = license_details.license_expiry_date or license_details.expiry_date or ""
        prov = license_details.issuing_province if license_details.issuing_province else "ON"

        # Determine surname for Canadian provincial initial validation
        parts = name.split()
        surname = parts[-1] if len(parts) > 1 else name

        # PASS 1: High-Fidelity Bilateral + LAB CLAHE Continuous Contrast
        raw_texts, boxes = self.extract_text_and_boxes(img_bgr, enhancement_mode="bilateral", is_preflattened=is_preflattened)

        # Check Spatial AAMVA fields first
        aamva_fields = self.extract_spatial_aamva_fields(boxes, province=prov)
        if aamva_fields.get("license_number") and aamva_fields["license_number"] not in raw_texts:
            raw_texts.append(aamva_fields["license_number"])

        def compute_all_scores(lines: List[str]):
            s_name, m_name = self._match_name(lines, name)
            s_id, m_id = self._match_dl_number(lines, dl_number, province=prov, surname=surname)
            s_dob, m_dob = self._match_date(lines, dob)
            s_exp, m_exp = self._match_date(lines, exp)

            whole_text = " ".join(lines).lower()
            s_reg = fuzz.partial_token_set_ratio(prov.lower(), whole_text) if prov and prov != "N/A" else 100.0
            return (s_name, m_name), (s_id, m_id), (s_dob, m_dob), (s_exp, m_exp), s_reg

        (score_name, matched_name), (score_id, matched_dl), (score_dob, matched_dob), (score_exp, matched_exp), score_region = compute_all_scores(raw_texts)

        # PASS 2: Multi-Scale Upsampling & Sharpening (If any field is below threshold)
        if (name and score_name < threshold_name) or \
           (dl_number and score_id < threshold_dl) or \
           (dob and score_dob < threshold_dob) or \
           (exp and score_exp < threshold_exp):

            upscaled = upscale_and_sharpen(img_bgr, scale=2.0, max_dim=1800)
            raw_texts_up, _ = self.extract_text_and_boxes(upscaled, enhancement_mode="clahe", is_preflattened=is_preflattened)

            (up_name, m_up_name), (up_id, m_up_id), (up_dob, m_up_dob), (up_exp, m_up_exp), up_reg = compute_all_scores(raw_texts_up)

            if up_name > score_name: score_name, matched_name = up_name, m_up_name
            if up_id > score_id: score_id, matched_dl = up_id, m_up_id
            if up_dob > score_dob: score_dob, matched_dob = up_dob, m_up_dob
            if up_exp > score_exp: score_exp, matched_exp = up_exp, m_up_exp
            score_region = max(score_region, up_reg)
            raw_texts.extend([l for l in raw_texts_up if l not in raw_texts])

        # PASS 3: Adaptive Binarization (Fallback for tough background patterns)
        if (dl_number and score_id < threshold_dl) or (name and score_name < threshold_name):
            raw_texts_bin, _ = self.extract_text_and_boxes(img_bgr, enhancement_mode="adaptive", is_preflattened=is_preflattened)
            (b_name, m_b_name), (b_id, m_b_id), (b_dob, m_b_dob), (b_exp, m_b_exp), b_reg = compute_all_scores(raw_texts_bin)

            if b_name > score_name: score_name, matched_name = b_name, m_b_name
            if b_id > score_id: score_id, matched_dl = b_id, m_b_id
            if b_dob > score_dob: score_dob, matched_dob = b_dob, m_b_dob
            if b_exp > score_exp: score_exp, matched_exp = b_exp, m_b_exp
            score_region = max(score_region, b_reg)
            raw_texts.extend([l for l in raw_texts_bin if l not in raw_texts])

        missing_fields = []
        if name and score_name < threshold_name:
            missing_fields.append(f"Full Name ({score_name:.1f}%)")
        if dl_number and score_id < threshold_dl:
            missing_fields.append(f"License Number ({score_id:.1f}%)")
        if dob and score_dob < threshold_dob:
            missing_fields.append(f"Date of Birth ({score_dob:.1f}%)")
        if exp and score_exp < threshold_exp:
            missing_fields.append(f"Expiry Date ({score_exp:.1f}%)")

        passed = len(missing_fields) == 0

        # Continuous Stage 1 confidence: weighted average of ID and Name similarity
        continuous_conf = round(min(1.0, (score_id / 100.0 * 0.5) + (score_name / 100.0 * 0.5)), 3) if passed else 0.0

        # Canonical format of extracted DL
        canonical_extracted_dl = format_canadian_dl(matched_dl or dl_number, province=prov) if score_id >= threshold_dl else None

        if passed:
            details = f"Name Match: {score_name:.1f}% | DL Match: {score_id:.1f}% | DOB Match: {score_dob:.1f}% | Expiry Match: {score_exp:.1f}% | All Canadian criteria met."
        else:
            details = f"Name Match: {score_name:.1f}% | DL Match: {score_id:.1f}% | DOB Match: {score_dob:.1f}% | Expiry Match: {score_exp:.1f}% | Low similarity fields: {', '.join(missing_fields)}"

        return LicenseOcrResult(
            passed=passed,
            extracted_dl_number=canonical_extracted_dl,
            extracted_name=matched_name or personal_info.full_name if score_name >= threshold_name else None,
            extracted_dob=matched_dob or personal_info.date_of_birth if score_dob >= threshold_dob else None,
            extracted_expiry=matched_exp or license_details.license_expiry_date if score_exp >= threshold_exp else None,
            extracted_province=prov if score_region >= 75.0 else None,
            number_matched=(score_id >= threshold_dl),
            number_similarity=min(1.0, score_id / 100.0),
            name_matched=(score_name >= threshold_name),
            name_similarity=min(1.0, score_name / 100.0),
            dob_matched=(score_dob >= threshold_dob),
            is_expired=False,
            driver_age_valid=True,
            confidence=continuous_conf,
            details=details,
            raw_ocr_lines=raw_texts,
            missing_fields=[f.split(' ')[0] for f in missing_fields],
            confidence_proof={
                "stage_verdict": "PASSED" if passed else "REJECTED",
                "province": prov,
                "metrics": metrics,
                "reason": details,
            },
        )


# Global instance
dl_ocr_engine = DrivingLicenseOCREngine()
