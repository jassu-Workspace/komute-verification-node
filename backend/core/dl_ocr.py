import logging
from typing import Any, Dict, List, Tuple, Optional
import cv2
import numpy as np
from rapidfuzz import fuzz

from app.schemas import LicenseDetails, LicenseOcrResult, PersonalInfo
from core.image_utils import apply_adaptive_threshold, deskew_image, resize_image_max_dimension

logger = logging.getLogger(__name__)


class DrivingLicenseOCREngine:
    """
    Stage 1: Driving License OCR Verification Engine using RapidOCR with ONNX Runtime.
    Performs auto-cropping, flattening, deskewing, and multi-pass fuzzy matching
    against expected driver registration details.
    """

    def __init__(self):
        pass

    def extract_text_lines(self, img_bgr: np.ndarray, apply_binarization: bool = True) -> List[str]:
        """
        Processes image through auto-crop, binarization, flattening, deskewing,
        and extracts text lines using RapidOCR (ONNX Runtime).
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        import gc
        scaled_img, _ = resize_image_max_dimension(img_bgr, max_dim=1024)

        # Binarization / Contrast pre-processing
        if apply_binarization:
            enhanced = apply_adaptive_threshold(scaled_img)
        else:
            enhanced = scaled_img

        from core.image_utils import flatten_id_card
        flattened = flatten_id_card(enhanced)
        deskewed = deskew_image(flattened)

        lines: List[str] = []
        try:
            from core.ocr_engine import shared_ocr
            ocr = shared_ocr.get_engine()

            res, _ = ocr(deskewed)
            if res:
                for line in res:
                    if len(line) >= 2:
                        text = str(line[1]).strip()
                        if text:
                            lines.append(text)

        except Exception as e:
            logger.error(f"RapidOCR recognition error: {e}")

        gc.collect()
        return lines

    def generate_date_variations(self, date_str: str) -> List[str]:
        """Generates common string representations of a date."""
        if not date_str:
            return []
        try:
            from dateutil import parser
            d = parser.parse(date_str)
            return [
                date_str,  # Original
                d.strftime("%Y-%m-%d"),
                d.strftime("%d-%m-%Y"),
                d.strftime("%d/%m/%Y"),
                d.strftime("%m/%d/%Y"),
                d.strftime("%d.%m.%Y"),
                d.strftime("%d %b %Y").lower(),
                d.strftime("%b %d, %Y").lower(),
                d.strftime("%Y/%m/%d"),
                d.strftime("%Y%m%d"),
            ]
        except Exception:
            return [date_str]

    def _normalize_digits(self, text: str, apply_confusion: bool = False) -> str:
        """Punctuation cleanup and optional character confusion normalization for DL strings."""
        import re
        text = text.replace('/', '-').replace('\\', '-').replace('.', '-').replace(' ', '')
        text = re.sub(r'[^a-zA-Z0-9\-]', '', text).upper()
        if apply_confusion:
            confusion = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'Z': '2', 'G': '6'}
            for k, v in confusion.items():
                text = text.replace(k, v)
        return text

    def _match_dl_number(self, raw_texts: List[str], target_dl: str) -> Tuple[float, Optional[str]]:
        """
        High-precision DL number matching:
        - Checks line-by-line exact substring & reverse substring containment
        - Sliding-window Levenshtein matching on alphanumeric streams
        - Confusion-matrix normalization (0/O, 1/I, 5/S, 2/Z, 8/B)
        - Whole concatenated text fallback
        """
        if not target_dl:
            return 100.0, None

        import re
        clean_target = re.sub(r'[^A-Za-z0-9]', '', target_dl).upper()
        if not clean_target:
            return 100.0, None

        confusion_map = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'Z': '2', 'G': '6'}

        def apply_confusion(s: str) -> str:
            for k, v in confusion_map.items():
                s = s.replace(k, v)
            return s

        target_conf = apply_confusion(clean_target)
        best_score = 0.0
        best_match: Optional[str] = None

        for line in raw_texts:
            clean_line = re.sub(r'[^A-Za-z0-9]', '', line).upper()
            if not clean_line or len(clean_line) < 3:
                continue

            # 1. Exact Substring Containment (e.g. D61014070660905 in D61014070660905E)
            if clean_target in clean_line:
                return 100.0, line

            # 2. Confusion-aware Substring Containment
            line_conf = apply_confusion(clean_line)
            if target_conf in line_conf:
                score = 98.0
                if score > best_score:
                    best_score = score
                    best_match = line

            # 3. Sliding Window Comparison if line is longer than target
            if len(clean_line) >= len(clean_target):
                w_len = len(clean_target)
                for i in range(len(clean_line) - w_len + 1):
                    chunk = clean_line[i : i + w_len]
                    s1 = fuzz.ratio(clean_target, chunk)
                    if s1 > best_score:
                        best_score = s1
                        best_match = line

                    s2 = fuzz.ratio(target_conf, apply_confusion(chunk))
                    if s2 > best_score:
                        best_score = s2
                        best_match = line
            else:
                # Reverse substring (line is a significant chunk of DL number)
                if len(clean_line) >= 7 and clean_line in clean_target:
                    s3 = (len(clean_line) / len(clean_target)) * 100.0
                    if s3 > best_score:
                        best_score = s3
                        best_match = line

                s4 = fuzz.partial_ratio(clean_target, clean_line)
                if s4 > best_score:
                    best_score = s4
                    best_match = line

            s5 = fuzz.ratio(clean_target, clean_line)
            if s5 > best_score:
                best_score = s5
                best_match = line

        # 4. Whole Concatenated Clean Text Check
        whole_clean = re.sub(r'[^A-Za-z0-9]', '', " ".join(raw_texts)).upper()
        if clean_target in whole_clean:
            return 100.0, best_match or target_dl

        if target_conf in apply_confusion(whole_clean):
            best_score = max(best_score, 95.0)

        return best_score, best_match

    def _match_name(self, raw_texts: List[str], target_name: str) -> Tuple[float, Optional[str]]:
        """Multi-token name matching across lines and joined OCR text."""
        if not target_name:
            return 100.0, None

        import re
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

        import re
        target_digits = re.sub(r'\D', '', target_date)
        if not target_digits:
            return 100.0, None

        # Try date variations
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
        self, img_bgr: np.ndarray, personal_info: PersonalInfo, license_details: LicenseDetails
    ) -> LicenseOcrResult:
        """
        Core verification pipeline implementing fine-tuned RapidOCR recognition and multi-pass fuzzy validation.
        """
        from app.config import settings

        threshold_name = getattr(settings, "fuzzy_name_threshold", 0.85) * 100.0
        threshold_dl = getattr(settings, "fuzzy_dl_threshold", 0.85) * 100.0
        threshold_dob = 80.0
        threshold_exp = 80.0

        # PASS 1: Standard Extraction
        raw_texts = self.extract_text_lines(img_bgr)

        name = personal_info.full_name if personal_info.full_name else ""
        dl_number = license_details.license_number if license_details.license_number else ""
        dob = personal_info.date_of_birth if personal_info.date_of_birth else ""
        exp = license_details.license_expiry_date or license_details.expiry_date or ""
        prov = license_details.issuing_province if license_details.issuing_province else ""

        def compute_all_scores(lines: List[str]):
            s_name, m_name = self._match_name(lines, name)
            s_id, m_id = self._match_dl_number(lines, dl_number)
            s_dob, m_dob = self._match_date(lines, dob)
            s_exp, m_exp = self._match_date(lines, exp)
            
            whole_text = " ".join(lines).lower()
            s_reg = fuzz.partial_token_set_ratio(prov.lower(), whole_text) if prov and prov != "N/A" else 100.0
            return (s_name, m_name), (s_id, m_id), (s_dob, m_dob), (s_exp, m_exp), s_reg

        (score_name, matched_name), (score_id, matched_dl), (score_dob, matched_dob), (score_exp, matched_exp), score_region = compute_all_scores(raw_texts)

        # PASS 2: Multi-Scale Upsampling (If any field is below threshold)
        if (name and score_name < threshold_name) or \
           (dl_number and score_id < threshold_dl) or \
           (dob and score_dob < threshold_dob) or \
           (exp and score_exp < threshold_exp):

            from core.image_utils import upscale_and_sharpen
            upscaled = upscale_and_sharpen(img_bgr, scale=2.5)
            raw_texts_up = self.extract_text_lines(upscaled, apply_binarization=False)

            (up_name, m_up_name), (up_id, m_up_id), (up_dob, m_up_dob), (up_exp, m_up_exp), up_reg = compute_all_scores(raw_texts_up)

            if up_name > score_name: score_name, matched_name = up_name, m_up_name
            if up_id > score_id: score_id, matched_dl = up_id, m_up_id
            if up_dob > score_dob: score_dob, matched_dob = up_dob, m_up_dob
            if up_exp > score_exp: score_exp, matched_exp = up_exp, m_up_exp
            score_region = max(score_region, up_reg)
            raw_texts.extend([l for l in raw_texts_up if l not in raw_texts])

        # PASS 3: CLAHE Contrast Recovery
        if (dob and score_dob < threshold_dob) or (exp and score_exp < threshold_exp) or (dl_number and score_id < threshold_dl):
            from core.image_utils import apply_clahe
            clahe_img = apply_clahe(img_bgr)
            raw_texts_clahe = self.extract_text_lines(clahe_img, apply_binarization=False)

            (c_name, m_c_name), (c_id, m_c_id), (c_dob, m_c_dob), (c_exp, m_c_exp), c_reg = compute_all_scores(raw_texts_clahe)

            if c_name > score_name: score_name, matched_name = c_name, m_c_name
            if c_id > score_id: score_id, matched_dl = c_id, m_c_id
            if c_dob > score_dob: score_dob, matched_dob = c_dob, m_c_dob
            if c_exp > score_exp: score_exp, matched_exp = c_exp, m_c_exp
            score_region = max(score_region, c_reg)
            raw_texts.extend([l for l in raw_texts_clahe if l not in raw_texts])

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

        if passed:
            details = f"Name Match: {score_name:.1f}% | DL Match: {score_id:.1f}% | DOB Match: {score_dob:.1f}% | Expiry Match: {score_exp:.1f}% | All fields met the similarity threshold."
        else:
            details = f"Name Match: {score_name:.1f}% | DL Match: {score_id:.1f}% | DOB Match: {score_dob:.1f}% | Expiry Match: {score_exp:.1f}% | Low similarity fields: {', '.join(missing_fields)}"

        import gc
        gc.collect()

        return LicenseOcrResult(
            passed=passed,
            extracted_dl_number=matched_dl or license_details.license_number if score_id >= threshold_dl else None,
            extracted_name=matched_name or personal_info.full_name if score_name >= threshold_name else None,
            extracted_dob=matched_dob or personal_info.date_of_birth if score_dob >= threshold_dob else None,
            extracted_expiry=matched_exp or license_details.license_expiry_date if score_exp >= threshold_exp else None,
            extracted_province=license_details.issuing_province if score_region >= 75.0 else None,
            number_matched=(score_id >= threshold_dl),
            number_similarity=min(1.0, score_id / 100.0),
            name_matched=(score_name >= threshold_name),
            name_similarity=min(1.0, score_name / 100.0),
            dob_matched=(score_dob >= threshold_dob),
            is_expired=False,
            driver_age_valid=True,
            confidence=1.0 if passed else 0.0,
            details=details,
            raw_ocr_lines=raw_texts,
            missing_fields=[f.split(' ')[0] for f in missing_fields],
            confidence_proof={"stage_verdict": "PASSED" if passed else "REJECTED", "reason": details},
        )


# Global instance
dl_ocr_engine = DrivingLicenseOCREngine()

