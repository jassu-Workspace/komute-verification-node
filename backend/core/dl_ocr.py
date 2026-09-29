import gc
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from rapidfuzz import fuzz

from app.schemas import LicenseDetails, LicenseOcrResult, PersonalInfo
from app.config import settings
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
        scaled_img, _ = resize_image_max_dimension(flattened, max_dim=settings.ocr_working_max_dim)

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
        '4d' = License Number, '4b' = Expiry, '3' = DOB, '4a' = Issue Date,
        '15' = Gender/Sex, '9' = Class, '1' = Surname, '2' = Given Name.
        """
        extracted: Dict[str, Optional[str]] = {
            "license_number": None,
            "expiry_date": None,
            "dob": None,
            "issue_date": None,
            "address": None,
            "gender": None,
            "license_class": None,
            "surname": None,
            "given_name": None,
        }
        if not boxes:
            return extracted

        for box in boxes:
            t = box.text.strip()
            # 4d tag: License Number (e.g. 4d, NUMBER, NUMERO)
            m_4d = re.search(r'(?:4[dD]|[dD]?NUMBER|NUMERO)[\.:\s]*(.+)?', t, re.IGNORECASE)
            if m_4d and not extracted["license_number"]:
                candidate = (m_4d.group(1) or "").strip()
                if len(candidate) >= 5 and re.search(r'\d', candidate):
                    extracted["license_number"] = candidate
                else:
                    nearest = self._find_adjacent_box(box, boxes)
                    if nearest and len(nearest.text.strip()) >= 5 and re.search(r'\d', nearest.text):
                        extracted["license_number"] = nearest.text.strip()

            # 4b tag: Expiry Date (AAMVA 4b or Ontario/Canadian 4b.EXP / 40EXP / 4EXP)
            m_4b = re.search(r'(?:4[bB06]?[\.:\s]*EXP|EXP[\s/]+EXP|4b|4B)[\.:\s]*(.+)?', t, re.IGNORECASE)
            if m_4b and not extracted["expiry_date"]:
                candidate = (m_4b.group(1) or "").strip()
                date_m = re.search(r'(19\d{2}|20\d{2})[-/\.1|lI]?(0[1-9]|1[0-2])[-/\.1|lI]?(0[1-9]|[12]\d|3[01])', candidate) if candidate else None
                if date_m:
                    extracted["expiry_date"] = f"{date_m.group(1)}/{date_m.group(2)}/{date_m.group(3)}"
                else:
                    nearest = self._find_adjacent_box(box, boxes)
                    if nearest:
                        date_m = re.search(r'(19\d{2}|20\d{2})[-/\.1|lI]?(0[1-9]|1[0-2])[-/\.1|lI]?(0[1-9]|[12]\d|3[01])', nearest.text)
                        if date_m:
                            extracted["expiry_date"] = f"{date_m.group(1)}/{date_m.group(2)}/{date_m.group(3)}"
                        elif len(nearest.text.strip()) >= 4 and re.search(r'\d{4}', nearest.text):
                            extracted["expiry_date"] = nearest.text.strip()

            # 4a tag: Issue Date (ISS / DEL)
            m_4a = re.search(r'(?:4a|4A|ISS|DEL)[\.:\s]*(.+)?', t, re.IGNORECASE)
            if m_4a and not extracted["issue_date"]:
                cand = (m_4a.group(1) or "").strip()
                date_m = re.search(r'(19\d{2}|20\d{2})[-/\.1|lI]?(0[1-9]|1[0-2])[-/\.1|lI]?(0[1-9]|[12]\d|3[01])', cand) if cand else None
                if date_m:
                    extracted["issue_date"] = f"{date_m.group(1)}/{date_m.group(2)}/{date_m.group(3)}"
                else:
                    nearest = self._find_adjacent_box(box, boxes)
                    if nearest:
                        date_m = re.search(r'(19\d{2}|20\d{2})[-/\.1|lI]?(0[1-9]|1[0-2])[-/\.1|lI]?(0[1-9]|[12]\d|3[01])', nearest.text)
                        if date_m:
                            extracted["issue_date"] = f"{date_m.group(1)}/{date_m.group(2)}/{date_m.group(3)}"

            # 3 tag: DOB (AAMVA 3 or DOB or DDN or CATEG)
            m_3 = re.search(r'(?:3[\.:\s]|DOB|DDN|CATEG)[\.:\s]*(.+)?', t, re.IGNORECASE)
            if m_3 and not extracted["dob"]:
                cand = (m_3.group(1) or "").strip()
                date_m = re.search(r'(19\d{2}|20\d{2})[-/\.1|lI]?(0[1-9]|1[0-2])[-/\.1|lI]?(0[1-9]|[12]\d|3[01])', cand) if cand else None
                if date_m:
                    extracted["dob"] = f"{date_m.group(1)}/{date_m.group(2)}/{date_m.group(3)}"
                else:
                    nearest = self._find_adjacent_box(box, boxes)
                    if nearest:
                        date_m = re.search(r'(19\d{2}|20\d{2})[-/\.1|lI]?(0[1-9]|1[0-2])[-/\.1|lI]?(0[1-9]|[12]\d|3[01])', nearest.text)
                        if date_m:
                            extracted["dob"] = f"{date_m.group(1)}/{date_m.group(2)}/{date_m.group(3)}"

            # 15 tag: Gender / Sex
            m_sex = re.search(r'(?:15[\.:\s\-]*|SEX|SEXE|GEX)[\.:\s\-\_]*([MFX])', t, re.IGNORECASE)
            if m_sex and not extracted["gender"]:
                extracted["gender"] = m_sex.group(1).upper()

            # 9 tag: License Class
            m_cls = re.search(r'(?:9[\.:\s]*C[A-Za-z]+|S?CASS[A-Za-z]*|CLASS|CLASSE)[\.:\s\-\_]*([A-Za-z0-9]{1,3})?', t, re.IGNORECASE)
            if m_cls and not extracted["license_class"]:
                val = (m_cls.group(1) or "").upper()
                if val and val not in ("CLASS", "CATEG", "CLASSE", "CASSL", "SCASSL"):
                    extracted["license_class"] = val
                else:
                    nearest = self._find_adjacent_box(box, boxes)
                    if nearest:
                        n_val = nearest.text.strip().upper()
                        if n_val and len(n_val) <= 4 and n_val not in ("CLASS", "CATEG", "CLASSE", "CASSL", "SCASSL"):
                            extracted["license_class"] = n_val

        return extracted

    def _find_adjacent_box(
        self,
        anchor: OCRBox,
        all_boxes: List[OCRBox],
        max_dist_x: float = 350.0,
        max_dist_y: float = 40.0,
    ) -> Optional[OCRBox]:
        """Finds the most geometrically plausible text box adjacent to an AAMVA tag, prioritizing same-line reading order."""
        same_line = []
        below_line = []

        for b in all_boxes:
            if b is anchor:
                continue
            dx = b.center_x - anchor.center_x
            dy = b.center_y - anchor.center_y

            # Horizontally adjacent (to the right on the same line)
            if 0 < dx < max_dist_x and abs(dy) <= 8.0:
                same_line.append((dx, b))
            # Vertically adjacent (directly below tag)
            elif 0 < dy < max_dist_y and abs(dx) < 60.0:
                below_line.append((dy, b))

        if same_line:
            same_line.sort(key=lambda x: x[0])
            return same_line[0][1]
        if below_line:
            below_line.sort(key=lambda x: x[0])
            return below_line[0][1]

        return None

    def generate_date_variations(self, date_str: str) -> List[str]:
        """Generates common string representations of a date with dayfirst and monthfirst support."""
        if not date_str:
            return []
        variations = {date_str.strip()}
        try:
            from dateutil import parser
            clean_str = date_str.strip()
            if re.match(r"^(19\d{2}|20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$", clean_str):
                clean_str = f"{clean_str[:4]}/{clean_str[4:6]}/{clean_str[6:]}"

            parsed_dates = []
            for df in (False, True):
                try:
                    d = parser.parse(clean_str, dayfirst=df)
                    parsed_dates.append(d)
                except Exception:
                    pass

            for d in parsed_dates:
                variations.update([
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
                ])
        except Exception:
            pass
        return list(variations)

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
        """Multi-format date and digit sequence matching with dayfirst/monthfirst resolution."""
        if not target_date or not str(target_date).strip():
            return 100.0, None

        target_digits = re.sub(r'\D', '', str(target_date))
        if not target_digits:
            return 100.0, None

        # Augment search texts with normalized YYYY/MM/DD dates where delimiters were misread as 1, l, I, |
        search_texts = list(raw_texts)
        for l in raw_texts:
            for m in re.finditer(r'(19\d{2}|20\d{2})[-/\.1|lI](0[1-9]|1[0-2])[-/\.1|lI](0[1-9]|[12]\d|3[01])', l):
                norm = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
                if norm not in search_texts:
                    search_texts.append(norm)

        # 1. Parse target date into components (both dayfirst=False and dayfirst=True)
        from dateutil import parser
        target_components = set()
        clean_target = str(target_date).strip()
        if re.match(r"^(19\d{2}|20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$", clean_target):
            clean_target = f"{clean_target[:4]}/{clean_target[4:6]}/{clean_target[6:]}"

        for df in (False, True):
            try:
                dt = parser.parse(clean_target, dayfirst=df)
                target_components.add((dt.year, dt.month, dt.day))
            except Exception:
                pass

        # 2. Check each candidate date in search_texts by year, month, day components
        for line in search_texts:
            for m in re.finditer(r'(19\d{2}|20\d{2})[-/\.](0[1-9]|1[0-2])[-/\.](0[1-9]|[12]\d|3[01])', line):
                cand_comp = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if cand_comp in target_components:
                    return 100.0, f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
            for m in re.finditer(r'(0[1-9]|[12]\d|3[01])[-/\.](0[1-9]|1[0-2])[-/\.](19\d{2}|20\d{2})', line):
                cand_comp_d = (int(m.group(3)), int(m.group(2)), int(m.group(1)))
                cand_comp_m = (int(m.group(3)), int(m.group(1)), int(m.group(2)))
                if cand_comp_d in target_components or cand_comp_m in target_components:
                    return 100.0, line

        variations = self.generate_date_variations(target_date)
        whole_text = " ".join(search_texts).lower()

        for v in variations:
            if v.lower() in whole_text:
                return 100.0, v

        best_score = 0.0
        best_match: Optional[str] = None

        for line in search_texts:
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
        import time

        from app.config import settings

        stage_t0 = time.perf_counter()

        threshold_name = settings.fuzzy_name_threshold * 100.0
        threshold_dl = settings.fuzzy_dl_threshold * 100.0
        threshold_dob = settings.fuzzy_dob_threshold
        threshold_exp = settings.fuzzy_expiry_threshold

        # Step 0: Pre-Flight Image Quality Gate
        is_usable, quality_msg, metrics = assess_image_quality(img_bgr)
        if not is_usable and metrics.get("blur_variance", 100.0) < settings.severe_blur_variance:
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

        # PASS 2: Multi-Scale Upsampling & Sharpening (If any field is below threshold
        # AND inside the recovery time budget — otherwise Pass-1 scores stand and
        # the stage returns instead of blowing the pipeline SLA.)
        def _within_recovery_budget() -> bool:
            return (time.perf_counter() - stage_t0) < settings.stage1_recovery_budget_seconds

        if _within_recovery_budget() and (
            (name and score_name < threshold_name)
            or (dl_number and score_id < threshold_dl)
            or (dob and score_dob < threshold_dob)
            or (exp and score_exp < threshold_exp)
        ):

            upscaled = upscale_and_sharpen(img_bgr, scale=2.0, max_dim=settings.ocr_working_max_dim)
            raw_texts_up, _ = self.extract_text_and_boxes(upscaled, enhancement_mode="clahe", is_preflattened=is_preflattened)

            (up_name, m_up_name), (up_id, m_up_id), (up_dob, m_up_dob), (up_exp, m_up_exp), up_reg = compute_all_scores(raw_texts_up)

            if up_name > score_name: score_name, matched_name = up_name, m_up_name
            if up_id > score_id: score_id, matched_dl = up_id, m_up_id
            if up_dob > score_dob: score_dob, matched_dob = up_dob, m_up_dob
            if up_exp > score_exp: score_exp, matched_exp = up_exp, m_up_exp
            score_region = max(score_region, up_reg)
            raw_texts.extend([l for l in raw_texts_up if l not in raw_texts])

        # PASS 3: Adaptive Binarization (Fallback for tough background patterns,
        # same recovery budget as Pass 2.)
        if _within_recovery_budget() and (
            (dl_number and score_id < threshold_dl) or (name and score_name < threshold_name)
        ):
            raw_texts_bin, _ = self.extract_text_and_boxes(img_bgr, enhancement_mode="adaptive", is_preflattened=is_preflattened)
            (b_name, m_b_name), (b_id, m_b_id), (b_dob, m_b_dob), (b_exp, m_b_exp), b_reg = compute_all_scores(raw_texts_bin)

            if b_name > score_name: score_name, matched_name = b_name, m_b_name
            if b_id > score_id: score_id, matched_dl = b_id, m_b_id
            if b_dob > score_dob: score_dob, matched_dob = b_dob, m_b_dob
            if b_exp > score_exp: score_exp, matched_exp = b_exp, m_b_exp
            score_region = max(score_region, b_reg)
            raw_texts.extend([l for l in raw_texts_bin if l not in raw_texts])

        # Collect all detected dates from raw_texts
        from datetime import date
        current_year = date.today().year
        all_detected_dates = []
        for l in raw_texts:
            for m in re.finditer(r'(19\d{2}|20\d{2})[-/\.1|lI]?(0[1-9]|1[0-2])[-/\.1|lI]?(0[1-9]|[12]\d|3[01])', l):
                norm_d = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
                if norm_d not in all_detected_dates:
                    all_detected_dates.append(norm_d)

        # Expiry date extraction: AAMVA tag, matched_exp, EXP line, or future date
        card_expiry = aamva_fields.get("expiry_date")
        if not card_expiry or not re.search(r'\d{4}', card_expiry):
            if score_exp >= threshold_exp and matched_exp:
                card_expiry = matched_exp
        if not card_expiry or not re.search(r'\d{4}', card_expiry):
            for line in raw_texts:
                if re.search(r'(?:4[bB06]?[\.:\s]*EXP|EXP)', line, re.IGNORECASE):
                    dm = re.search(r'(20\d{2})[-/\.1|lI]?(0[1-9]|1[0-2])[-/\.1|lI]?(0[1-9]|[12]\d|3[01])', line)
                    if dm:
                        card_expiry = f"{dm.group(1)}/{dm.group(2)}/{dm.group(3)}"
                        break
        if not card_expiry or not re.search(r'\d{4}', card_expiry):
            for d in all_detected_dates:
                if int(d.split("/")[0]) >= current_year - 1:
                    card_expiry = d
                    break

        # Issue date extraction: AAMVA tag or ISS/DEL line
        card_issue = aamva_fields.get("issue_date")
        if not card_issue or not re.search(r'\d{4}', card_issue):
            for i, line in enumerate(raw_texts):
                if re.search(r'(?:4a|ISS|DEL)', line, re.IGNORECASE):
                    for target in (line, raw_texts[i+1] if i+1 < len(raw_texts) else ""):
                        dm = re.search(r'(19\d{2}|20\d{2})[-/\.1|lI]?(0[1-9]|1[0-2])[-/\.1|lI]?(0[1-9]|[12]\d|3[01])', target)
                        if dm:
                            cand_issue = f"{dm.group(1)}/{dm.group(2)}/{dm.group(3)}"
                            if cand_issue != card_expiry:
                                card_issue = cand_issue
                                break
                    if card_issue:
                        break
        if not card_issue:
            for d in all_detected_dates:
                if d != card_expiry and d != (matched_dob or ""):
                    y = int(d.split("/")[0])
                    if 1990 <= y <= current_year:
                        card_issue = d
                        break

        # DOB extraction:
        # 1. Prefer matched_dob if DOB fuzzy score >= threshold_dob
        # 2. Then check aamva_fields.get("dob")
        # 3. Then search all_detected_dates for valid birth date (must be older than issue date and current_year - 16)
        card_dob = None
        if score_dob >= threshold_dob and matched_dob:
            card_dob = matched_dob
        elif aamva_fields.get("dob") and re.search(r'\d{4}', aamva_fields["dob"]):
            card_dob = aamva_fields["dob"]

        if not card_dob:
            issue_year = int(card_issue.split("/")[0]) if (card_issue and re.match(r'^\d{4}', card_issue)) else current_year
            max_birth_year = min(current_year - 16, issue_year - 16)
            for d in all_detected_dates:
                y = int(d.split("/")[0])
                if 1920 <= y <= max_birth_year and d != card_expiry and d != card_issue:
                    card_dob = d
                    break

        # Address extraction: Street + City, Province, Postal Code
        card_address = aamva_fields.get("address")
        if not card_address:
            addr_parts = []
            for line in raw_texts:
                if re.search(r'\d+\s*[A-Za-z]+', line) and not re.search(r'(?:EXP|HGT|HAUT|ISS|DEL|CATEG|NUM|D61|ONTARIO|DRIVER|PERMIS)', line, re.IGNORECASE):
                    addr_parts.append(line)
                elif re.search(r'[A-Za-z0-9\s]+,\s*[A-Z]{2}', line):
                    addr_parts.append(line)
            if addr_parts:
                card_address = ", ".join(addr_parts[:2])

        # Gender & Class
        card_gender = aamva_fields.get("gender")
        if not card_gender:
            for line in raw_texts:
                m_g = re.search(r'(?:15[\.:\s]|SEX|GEX|SEXE).*?([MFX])$', line, re.IGNORECASE)
                if m_g:
                    card_gender = m_g.group(1).upper()
                    break

        card_class = aamva_fields.get("license_class")
        if not card_class:
            for line in raw_texts:
                m_c = re.search(r'(?:9[\.:\s]*C[A-Za-z]+|CLASS|CLASSE)[\.:\s\-\_]*([A-Za-z0-9]{1,3})', line, re.IGNORECASE)
                if m_c and m_c.group(1).upper() not in ("CLASS", "CATEG", "CLASSE"):
                    card_class = m_c.group(1).upper()
                    break

        # Expiry date validation
        is_expired = False
        if card_expiry:
            try:
                exp_parts = [int(p) for p in re.findall(r'\d+', card_expiry)]
                if len(exp_parts) == 3:
                    is_expired = date(exp_parts[0], exp_parts[1], exp_parts[2]) < date.today()
            except Exception:
                pass

        # Driver age validation
        driver_age_valid = True
        dob_ref = card_dob or matched_dob or dob
        if dob_ref:
            try:
                dob_parts = [int(p) for p in re.findall(r'\d+', dob_ref)]
                if len(dob_parts) == 3:
                    if dob_parts[0] <= 31 and dob_parts[2] > 1900:
                        dob_obj = date(dob_parts[2], dob_parts[1], dob_parts[0])
                    else:
                        dob_obj = date(dob_parts[0], dob_parts[1], dob_parts[2])
                    driver_age_valid = ((date.today() - dob_obj).days // 365) >= 18
            except Exception:
                pass

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

        # Normalized field-level matching scores (0.0 to 1.0)
        name_sim = round(min(1.0, score_name / 100.0), 3) if name else 1.0
        dl_sim = round(min(1.0, score_id / 100.0), 3) if dl_number else 1.0
        dob_sim = round(min(1.0, score_dob / 100.0), 3) if dob else 1.0
        exp_sim = round(min(1.0, score_exp / 100.0), 3) if exp else 1.0

        # Continuous Stage 1 confidence: average across active card fields
        active_sims = [dl_sim, name_sim]
        if dob:
            active_sims.append(dob_sim)
        if exp:
            active_sims.append(exp_sim)
        continuous_conf = round(sum(active_sims) / float(len(active_sims)), 3)

        # Canonical format of extracted DL
        canonical_extracted_dl = format_canadian_dl(matched_dl or dl_number, province=prov) if score_id >= threshold_dl else None

        if passed:
            details = f"Name Match: {score_name:.1f}% | DL Match: {score_id:.1f}% | DOB Match: {score_dob:.1f}% | Expiry Match: {score_exp:.1f}% | All Canadian criteria met."
        else:
            details = f"Name Match: {score_name:.1f}% | DL Match: {score_id:.1f}% | DOB Match: {score_dob:.1f}% | Expiry Match: {score_exp:.1f}% | Low similarity fields: {', '.join(missing_fields)}"

        return LicenseOcrResult(
            passed=passed,
            extracted_dl_number=canonical_extracted_dl,
            extracted_name=matched_name or (personal_info.full_name if score_name >= threshold_name else None),
            extracted_dob=card_dob or matched_dob or (personal_info.date_of_birth if score_dob >= threshold_dob else None),
            extracted_expiry=card_expiry or matched_exp or (license_details.license_expiry_date if score_exp >= threshold_exp else None),
            extracted_province=prov if score_region >= settings.region_gate_score else None,
            extracted_address=card_address,
            extracted_gender=card_gender,
            extracted_class=card_class,
            extracted_issue_date=card_issue,
            number_matched=(score_id >= threshold_dl),
            number_similarity=dl_sim,
            name_matched=(score_name >= threshold_name),
            name_similarity=name_sim,
            dob_matched=(score_dob >= threshold_dob) if dob else True,
            dob_similarity=dob_sim,
            expiry_similarity=exp_sim,
            is_expired=is_expired,
            driver_age_valid=driver_age_valid,
            confidence=continuous_conf,
            details=details,
            raw_ocr_lines=raw_texts,
            missing_fields=[re.sub(r'\s*\(\d+.*?\)', '', f).strip() for f in missing_fields],
            confidence_proof={
                "stage_verdict": "PASSED" if passed else "REJECTED",
                "province": prov,
                "metrics": metrics,
                "reason": details,
            },
        )


# Global instance
dl_ocr_engine = DrivingLicenseOCREngine()
