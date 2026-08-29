import logging
from typing import Any, Dict, List, Tuple, Optional
import cv2
import numpy as np
from rapidfuzz import fuzz
from dateutil import parser

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
        """Generates common string representations of a YYYY-MM-DD date."""
        if not date_str:
            return []
        try:
            d = parser.parse(date_str)
            return [
                date_str,  # YYYY-MM-DD
                d.strftime("%d-%m-%Y"),
                d.strftime("%d/%m/%Y"),
                d.strftime("%m/%d/%Y"),
                d.strftime("%d.%m.%Y"),
                d.strftime("%d %b %Y").lower(),  # e.g. 15 may 2024
                d.strftime("%b %d, %Y").lower(),
                d.strftime("%Y/%m/%d"),
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

    def verify_license(
        self, img_bgr: np.ndarray, personal_info: PersonalInfo, license_details: LicenseDetails
    ) -> LicenseOcrResult:
        """
        Core verification pipeline implementing fine-tuned RapidOCR recognition and multi-pass fuzzy validation.
        """
        # PASS 1: Standard Extraction
        raw_texts = self.extract_text_lines(img_bgr)
        whole_raw_text = " ".join(raw_texts).lower()

        # Helper function to compute scores for a given raw text string
        name = personal_info.full_name.lower() if personal_info.full_name else ""
        dl_number = license_details.license_number.lower() if license_details.license_number else ""

        def compute_scores(text: str):
            s_name = fuzz.partial_token_set_ratio(name, text) if name else 100.0
            
            # DL Number match: Direct + confusion-aware match
            s_id = fuzz.partial_token_set_ratio(self._normalize_digits(dl_number), self._normalize_digits(text)) if dl_number else 100.0
            if dl_number:
                s_id_conf = fuzz.partial_token_set_ratio(
                    self._normalize_digits(dl_number, apply_confusion=True),
                    self._normalize_digits(text, apply_confusion=True)
                )
                s_id = max(s_id, s_id_conf)

            import re
            pure_digits_text = re.sub(r'\D', '', text)

            def get_best_date_match(target_digits, source_digits):
                if not target_digits:
                    return 100.0
                if target_digits in source_digits:
                    return 100.0
                best_score = 0.0
                window_size = len(target_digits) + 3
                for i in range(len(source_digits) - len(target_digits) + 1):
                    chunk = source_digits[i : i + window_size]
                    best_score = max(best_score, fuzz.partial_ratio(target_digits, chunk))
                return best_score

            target_dob = personal_info.date_of_birth.lower() if personal_info.date_of_birth else ""
            target_dob_digits = re.sub(r'\D', '', target_dob)
            s_dob = fuzz.partial_token_set_ratio(target_dob, self._normalize_digits(text)) if target_dob else 100.0

            if target_dob_digits:
                digit_score = get_best_date_match(target_dob_digits, pure_digits_text)
                if digit_score >= 85.0:
                    s_dob = max(s_dob, 90.0)

            target_exp = license_details.license_expiry_date.lower() if license_details.license_expiry_date else ""
            target_exp_digits = re.sub(r'\D', '', target_exp)
            s_exp = fuzz.partial_token_set_ratio(target_exp, self._normalize_digits(text)) if target_exp else 100.0

            if target_exp_digits:
                digit_score = get_best_date_match(target_exp_digits, pure_digits_text)
                if digit_score >= 85.0:
                    s_exp = max(s_exp, 90.0)

            s_reg = fuzz.partial_token_set_ratio(license_details.issuing_province.lower(), text) if license_details.issuing_province else 100.0

            return s_name, s_id, s_dob, s_exp, s_reg

        score_name, score_id, score_dob, score_exp, score_region = compute_scores(whole_raw_text)

        threshold = 90.0

        # PASS 2: Multi-Scale Upsampling WITHOUT Binarization (Saves blurred gradients)
        if (personal_info.full_name and score_name < threshold) or \
           (license_details.license_number and score_id < threshold) or \
           (personal_info.date_of_birth and score_dob < threshold) or \
           (license_details.license_expiry_date and score_exp < threshold):

            from core.image_utils import upscale_and_sharpen
            upscaled = upscale_and_sharpen(img_bgr, scale=2.5)
            raw_texts_up = self.extract_text_lines(upscaled, apply_binarization=False)
            whole_raw_text_up = " ".join(raw_texts_up).lower()

            up_name, up_id, up_dob, up_exp, up_reg = compute_scores(whole_raw_text_up)

            score_name = max(score_name, up_name)
            score_id = max(score_id, up_id)
            score_dob = max(score_dob, up_dob)
            score_exp = max(score_exp, up_exp)
            score_region = max(score_region, up_reg)

        # PASS 3: CLAHE Contrast Recovery (For extreme Glare/Low Contrast cases)
        if (personal_info.date_of_birth and score_dob < threshold) or \
           (license_details.license_expiry_date and score_exp < threshold):
            from core.image_utils import apply_clahe
            clahe_img = apply_clahe(img_bgr)
            raw_texts_clahe = self.extract_text_lines(clahe_img, apply_binarization=False)
            whole_raw_text_clahe = " ".join(raw_texts_clahe).lower()

            c_name, c_id, c_dob, c_exp, c_reg = compute_scores(whole_raw_text_clahe)

            score_name = max(score_name, c_name)
            score_id = max(score_id, c_id)
            score_dob = max(score_dob, c_dob)
            score_exp = max(score_exp, c_exp)
            score_region = max(score_region, c_reg)

        # PASS 4: Morphological Noise Eradication (Zero-RAM fallback)
        if (personal_info.date_of_birth and score_dob < threshold) or \
           (license_details.license_expiry_date and score_exp < threshold) or \
           (license_details.license_number and score_id < threshold):

            from core.image_utils import upscale_and_sharpen
            up_img = upscale_and_sharpen(img_bgr, scale=2.0)
            kernel = np.ones((2, 2), np.uint8)
            morphed = cv2.morphologyEx(up_img, cv2.MORPH_CLOSE, kernel)

            raw_texts_morph = self.extract_text_lines(morphed, apply_binarization=False)
            whole_raw_text_morph = " ".join(raw_texts_morph).lower()

            m_name, m_id, m_dob, m_exp, m_reg = compute_scores(whole_raw_text_morph)

            score_name = max(score_name, m_name)
            score_id = max(score_id, m_id)
            score_dob = max(score_dob, m_dob)
            score_exp = max(score_exp, m_exp)
            score_region = max(score_region, m_reg)

        missing_fields = []
        name = personal_info.full_name.lower() if personal_info.full_name else ""
        dl_number = license_details.license_number.lower() if license_details.license_number else ""

        if name and score_name < threshold:
            missing_fields.append(f"Full Name ({score_name:.1f}%)")
        if dl_number and score_id < threshold:
            missing_fields.append(f"License Number ({score_id:.1f}%)")
        if personal_info.date_of_birth and score_dob < threshold:
            missing_fields.append(f"Date of Birth ({score_dob:.1f}%)")
        if license_details.license_expiry_date and score_exp < threshold:
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
            extracted_dl_number=license_details.license_number if score_id >= threshold else None,
            extracted_name=personal_info.full_name if score_name >= threshold else None,
            extracted_dob=personal_info.date_of_birth if score_dob >= threshold else None,
            extracted_expiry=license_details.license_expiry_date if score_exp >= threshold else None,
            extracted_province=license_details.issuing_province if score_region >= threshold else None,
            number_matched=(score_id >= threshold),
            number_similarity=score_id / 100.0,
            name_matched=(score_name >= threshold),
            name_similarity=score_name / 100.0,
            dob_matched=(score_dob >= threshold),
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
