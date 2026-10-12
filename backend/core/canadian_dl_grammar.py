import calendar
from datetime import date, datetime
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import unicodedata

logger = logging.getLogger(__name__)

# Mapping of common optical OCR character confusions
# Digits that get misrecognized as letters and vice-versa
CONFUSION_TO_DIGIT: Dict[str, str] = {
    "O": "0",
    "o": "0",
    "Q": "0",
    "D": "0",  # In strictly numeric fields, D is frequently 0
    "I": "1",
    "i": "1",
    "l": "1",
    "L": "1",
    "|": "1",
    "!": "1",
    "Z": "2",
    "z": "2",
    "S": "5",
    "s": "5",
    "G": "6",
    "b": "6",
    "T": "7",
    "B": "8",
    "g": "9",
    "q": "9",
}

CONFUSION_TO_LETTER: Dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}

# Standard AAMVA field identifiers
AAMVA_TAGS: Dict[str, str] = {
    "1": "surname",
    "2": "given_names",
    "3": "date_of_birth",
    "4a": "issue_date",
    "4b": "expiry_date",
    "4d": "license_number",
    "4c": "issuing_authority",
    "8": "address",
    "9": "license_class",
    "12": "restrictions",
    "15": "sex",
    "16": "height",
}

# Standard Canadian Province/Territory Codes & Full Names
CANADIAN_PROVINCE_CODES: Dict[str, str] = {
    "ON": "Ontario",
    "BC": "British Columbia",
    "AB": "Alberta",
    "QC": "Quebec",
    "MB": "Manitoba",
    "SK": "Saskatchewan",
    "NS": "Nova Scotia",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "PE": "Prince Edward Island",
    "PEI": "Prince Edward Island",
    "YT": "Yukon",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
}

# AAMVA Issuer Identification Number (IIN) mapping for all 13 Canadian jurisdictions
AAMVA_IIN_MAP: Dict[str, str] = {
    "636012": "ON",
    "604428": "QC",
    "636028": "BC",
    "604432": "AB",
    "636044": "SK",
    "636048": "MB",
    "636017": "NB",
    "636013": "NS",
    "604426": "PE",
    "636016": "NL",
    "604429": "YT",
    "604434": "NT",
    "604433": "NU",
}

# Master Provincial DL Regexes (with all dashes, spaces, and formatting separators removed)
PROVINCIAL_DL_REGEX: Dict[str, str] = {
    "ON": r"^[A-Z]\d{14}$",
    "QC": r"^[A-Z]\d{12}$",
    "BC": r"^\d{7,8}$",
    "AB": r"^(?:\d{6,9}|[A-Z]{1,2}\d{4,5})$",
    "SK": r"^\d{8}$",
    "MB": r"^(?:[A-Z][A-Z*]{4}[A-Z][A-Z*][A-Z0-9]{5,6}|[A-Z]\d{12,13})$",
    "NB": r"^\d{1,7}$",
    "NS": r"^[A-Z][A-Z* ]{4}\d{9}$",
    "PE": r"^\d{1,6}$",
    "PEI": r"^\d{1,6}$",
    "NL": r"^[A-Z]\d{9}$",
    "YT": r"^\d{1,6}$",
    "NT": r"^\d{5,9}$",
    "NU": r"^\d{5,6}$",
}


def normalize_province_code(prov: Optional[str]) -> str:
    """Normalizes province names or codes (e.g. 'Ontario' -> 'ON'). Defaults to 'ON'."""
    if not prov:
        return "ON"
    cleaned = prov.strip().upper()
    if cleaned in CANADIAN_PROVINCE_CODES:
        return "PE" if cleaned == "PEI" else cleaned
    for code, name in CANADIAN_PROVINCE_CODES.items():
        if cleaned == name.upper() or cleaned in name.upper():
            return "PE" if code == "PEI" else code
    return "ON"


def lookup_iin_province(iin: str) -> Optional[str]:
    """Resolves an AAMVA Issuer Identification Number (IIN) to a province code."""
    if not iin:
        return None
    raw = iin.strip()
    if len(raw) == 5 and not raw.startswith("6"):
        raw = "6" + raw
    return AAMVA_IIN_MAP.get(raw)


def clean_dl_string(raw: str) -> str:
    """Strips all spaces, dashes, dots, and non-alphanumeric punctuation."""
    if not raw:
        return ""
    return re.sub(r"[^A-Za-z0-9*]", "", raw).upper()


def get_initial(name: str) -> str:
    """Extracts normalized ASCII single-character initial of a name."""
    if not name:
        return ""
    norm = unicodedata.normalize("NFKD", name.strip().upper())
    clean = re.sub(r"[^A-Z]", "", norm)
    return clean[:1] if clean else ""


def parse_date_safely(date_val: Optional[Union[str, date]]) -> Optional[date]:
    """Parses various date representations into a standard datetime.date."""
    if not date_val:
        return None
    if isinstance(date_val, date) and not isinstance(date_val, datetime):
        return date_val
    if isinstance(date_val, datetime):
        return date_val.date()
    try:
        from dateutil import parser
        s = str(date_val).strip()
        # Handle compact YYYYMMDD
        if re.match(r"^\d{8}$", s):
            return date(int(s[:4]), int(s[4:6]), int(s[6:]))
        dt = parser.parse(s, dayfirst=False)
        return dt.date()
    except Exception:
        try:
            from dateutil import parser
            dt = parser.parse(str(date_val).strip(), dayfirst=True)
            return dt.date()
        except Exception:
            return None


def check_embedded_province_rules(
    dl_number: str,
    province: str,
    surname: str = "",
    given_names: str = "",
    dob: Optional[Union[str, date]] = None,
    expiry: Optional[Union[str, date]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates deterministic mathematical/checksum rules embedded inside Canadian DL numbers
    as specified in the Canadian Driver's Licence OCR Reference document.

    Returns:
        (passed, list_of_failure_reasons)
        An empty list_of_failure_reasons indicates all applicable checks passed.
    """
    prov = normalize_province_code(province)
    cleaned = clean_dl_string(dl_number)
    if not cleaned:
        return False, ["empty_number"]

    # Match against the province's master regex
    expected_pattern = PROVINCIAL_DL_REGEX.get(prov)
    if expected_pattern and not re.match(expected_pattern, cleaned):
        return False, ["format_regex_mismatch"]

    dob_date = parse_date_safely(dob)
    expiry_date = parse_date_safely(expiry)
    failed: List[str] = []

    # 1. Ontario (ON): A1234-56789-01234 (15 chars)
    # Digits 9-10 = birth year (YY), 11-12 = birth month (MM, soft), 13-14 = birth day (DD)
    if prov == "ON":
        m = re.match(r"^([A-Z])(\d{14})$", cleaned)
        if not m:
            return False, ["format_ontario_length"]
        initial_letter = m.group(1)
        digits = m.group(2)

        # Surname initial check (with reversed given names fallback)
        sur_init = get_initial(surname)
        giv_init = get_initial(given_names)
        if sur_init and initial_letter != sur_init:
            if giv_init and initial_letter == giv_init:
                failed.append("names_reversed")
            else:
                failed.append("surname_initial_mismatch")

        # Birth date checks
        if dob_date:
            expected_yy = f"{dob_date.year % 100:02d}"
            expected_mm = f"{dob_date.month:02d}"
            expected_dd = f"{dob_date.day:02d}"

            # Digits 9-10 are at index 8:10 in the 14-digit sequence
            if digits[8:10] != expected_yy:
                failed.append("birth_year_mismatch")
            # Digits 11-12 are at index 10:12 in the 14-digit sequence (soft flag)
            if digits[10:12] != expected_mm:
                failed.append("birth_month_soft_mismatch")
            # Digits 13-14 are at index 12:14 in the 14-digit sequence
            if digits[12:14] != expected_dd:
                failed.append("birth_day_mismatch")

    # 2. Quebec (QC): A1234-123456-12 (13 chars)
    # 1 letter + 4 digits + DDMMYY (digits 5-10) + 2 digits
    elif prov == "QC":
        m = re.match(r"^([A-Z])(\d{4})(\d{6})(\d{2})$", cleaned)
        if not m:
            return False, ["format_quebec_length"]
        initial_letter = m.group(1)
        dob_digits = m.group(3)

        sur_init = get_initial(surname)
        if sur_init and initial_letter != sur_init:
            failed.append("surname_initial_mismatch")

        if dob_date:
            expected_ddmmyy = f"{dob_date.day:02d}{dob_date.month:02d}{dob_date.year % 100:02d}"
            if dob_digits != expected_ddmmyy:
                failed.append("birth_date_ddmmyy_mismatch")

    # 3. Nova Scotia (NS): AAAAADDMMYY### (14 chars)
    # First 5 chars = surname (space or * padded), chars 6-11 = DDMMYY, chars 12-14 = 3 digits
    elif prov == "NS":
        m = re.match(r"^([A-Z][A-Z* ]{4})(\d{6})(\d{3})$", cleaned)
        if not m:
            return False, ["format_novascotia_length"]
        surname_part = m.group(1).rstrip("* ")
        dob_digits = m.group(2)

        if surname:
            clean_sur = re.sub(r"[^A-Z]", "", unicodedata.normalize("NFKD", surname.strip().upper()))
            expected_sur_prefix = clean_sur[:5]
            if not expected_sur_prefix.startswith(surname_part) and not surname_part.startswith(expected_sur_prefix):
                failed.append("surname_prefix_mismatch")

        if dob_date:
            expected_ddmmyy = f"{dob_date.day:02d}{dob_date.month:02d}{dob_date.year % 100:02d}"
            if dob_digits != expected_ddmmyy:
                failed.append("birth_date_ddmmyy_mismatch")

    # 4. Newfoundland and Labrador (NL): AYYMMDD### (10 chars)
    # Char 0 = surname initial, chars 1-6 = YYMMDD, chars 7-9 = 3 digits
    elif prov == "NL":
        m = re.match(r"^([A-Z])(\d{6})(\d{3})$", cleaned)
        if not m:
            return False, ["format_newfoundland_length"]
        initial_letter = m.group(1)
        dob_digits = m.group(2)

        sur_init = get_initial(surname)
        if sur_init and initial_letter != sur_init:
            failed.append("surname_initial_mismatch")

        if dob_date:
            expected_yymmdd = f"{dob_date.year % 100:02d}{dob_date.month:02d}{dob_date.day:02d}"
            if dob_digits != expected_yymmdd:
                failed.append("birth_date_yymmdd_mismatch")

    # 5. Saskatchewan (SK): 8 digits
    # Expiry must fall on the last day of the birth month
    elif prov == "SK":
        if not re.match(r"^\d{8}$", cleaned):
            return False, ["format_saskatchewan_length"]
        if dob_date and expiry_date:
            # Expiry month should equal birth month
            if expiry_date.month != dob_date.month:
                failed.append("sk_expiry_month_mismatch")
            # Expiry day must be the last calendar day of that month
            last_day = calendar.monthrange(expiry_date.year, expiry_date.month)[1]
            if expiry_date.day != last_day:
                failed.append("sk_expiry_not_monthend")

    # 6. Manitoba (MB): Name-derived letters then digits or legacy 1-letter + digits
    elif prov == "MB":
        is_name_derived = bool(re.match(r"^[A-Z][A-Z*]{4}[A-Z][A-Z*][A-Z0-9]{5,6}$", cleaned))
        is_legacy = bool(re.match(r"^[A-Z]\d{12,13}$", cleaned))
        if not (is_name_derived or is_legacy):
            return False, ["format_manitoba_structure"]
        if surname:
            sur_init = get_initial(surname)
            if sur_init and cleaned[0] != sur_init:
                failed.append("surname_initial_soft_mismatch")

    # Filter out soft warnings for binary pass/fail decision
    hard_failures = [f for f in failed if "soft" not in f and "reversed" not in f]
    passed = len(hard_failures) == 0
    return passed, failed


def repair_provincial_dl(
    raw_dl: str,
    province: Optional[str] = "ON",
    surname: Optional[str] = "",
) -> Tuple[str, float, bool]:
    """
    Validates and deterministically auto-repairs an OCR-extracted driving license number
    against the official Canadian provincial specification across all 13 jurisdictions.

    Returns:
        (repaired_dl, confidence_score, is_valid)
    """
    prov = normalize_province_code(province)
    cleaned = clean_dl_string(raw_dl)
    if not cleaned:
        return "", 0.0, False

    surname_initial = get_initial(surname or "")

    if prov == "ON":
        return _repair_ontario_dl(cleaned, surname_initial)
    elif prov == "QC":
        return _repair_quebec_dl(cleaned, surname_initial)
    elif prov == "BC":
        return _repair_bc_dl(cleaned)
    elif prov == "AB":
        return _repair_alberta_dl(cleaned)
    elif prov == "SK":
        return _repair_numeric_dl(cleaned, expected_len=8)
    elif prov == "MB":
        return _repair_manitoba_dl(cleaned, surname_initial)
    elif prov == "NS":
        return _repair_novascotia_dl(cleaned, surname or "")
    elif prov == "NL":
        return _repair_newfoundland_dl(cleaned, surname_initial)
    elif prov == "NB":
        return _repair_numeric_range_dl(cleaned, min_len=1, max_len=7)
    elif prov in ("PE", "PEI"):
        return _repair_numeric_range_dl(cleaned, min_len=1, max_len=6)
    elif prov == "YT":
        return _repair_numeric_range_dl(cleaned, min_len=1, max_len=6)
    elif prov == "NT":
        return _repair_numeric_range_dl(cleaned, min_len=5, max_len=9)
    elif prov == "NU":
        return _repair_numeric_range_dl(cleaned, min_len=5, max_len=6)

    # Fallback
    return cleaned, 0.8, True


def _repair_ontario_dl(cleaned: str, surname_initial: str = "") -> Tuple[str, float, bool]:
    """Ontario format: 1 Letter + 14 Digits (Total 15 chars)."""
    chars = list(cleaned)
    repairs_made = 0

    if len(chars) > 15:
        chars = chars[:15]

    # Slot 0: Letter
    if len(chars) >= 1:
        if chars[0].isdigit():
            if surname_initial and surname_initial.isalpha():
                chars[0] = surname_initial
                repairs_made += 1
            elif chars[0] in CONFUSION_TO_LETTER:
                chars[0] = CONFUSION_TO_LETTER[chars[0]]
                repairs_made += 1
        elif surname_initial and surname_initial.isalpha() and chars[0] != surname_initial:
            if chars[0] in ("O", "Q", "B", "C") and surname_initial in ("D", "O", "B"):
                chars[0] = surname_initial
                repairs_made += 1

    # Slots 1 to 14: Digits
    for i in range(1, len(chars)):
        ch = chars[i]
        if not ch.isdigit():
            if ch in CONFUSION_TO_DIGIT:
                chars[i] = CONFUSION_TO_DIGIT[ch]
                repairs_made += 1

    repaired = "".join(chars)
    is_valid = len(repaired) == 15 and repaired[0].isalpha() and repaired[1:].isdigit()
    confidence = max(0.5, 1.0 - (repairs_made * 0.05)) if is_valid else 0.5
    return repaired, confidence, is_valid


def _repair_quebec_dl(cleaned: str, surname_initial: str = "") -> Tuple[str, float, bool]:
    """Quebec format: 1 Letter + 12 Digits (Total 13 chars)."""
    chars = list(cleaned)
    repairs_made = 0

    if len(chars) > 13:
        chars = chars[:13]

    if len(chars) >= 1:
        if chars[0].isdigit():
            if surname_initial and surname_initial.isalpha():
                chars[0] = surname_initial
                repairs_made += 1
            elif chars[0] in CONFUSION_TO_LETTER:
                chars[0] = CONFUSION_TO_LETTER[chars[0]]
                repairs_made += 1

    for i in range(1, len(chars)):
        if not chars[i].isdigit():
            if chars[i] in CONFUSION_TO_DIGIT:
                chars[i] = CONFUSION_TO_DIGIT[chars[i]]
                repairs_made += 1

    repaired = "".join(chars)
    is_valid = len(repaired) == 13 and repaired[0].isalpha() and repaired[1:].isdigit()
    confidence = max(0.5, 1.0 - (repairs_made * 0.05)) if is_valid else 0.5
    return repaired, confidence, is_valid


def _repair_bc_dl(cleaned: str) -> Tuple[str, float, bool]:
    """British Columbia format: 7 or 8 digits."""
    return _repair_numeric_range_dl(cleaned, min_len=7, max_len=8)


def _repair_alberta_dl(cleaned: str) -> Tuple[str, float, bool]:
    """Alberta format: 6 to 9 digits, or 1-2 letters + 4-5 digits."""
    chars = list(cleaned)
    repairs_made = 0

    # If starts with digits, treat as numeric 6-9 digits
    if len(chars) > 0 and chars[0].isdigit():
        for i in range(len(chars)):
            if not chars[i].isdigit():
                if chars[i] in CONFUSION_TO_DIGIT:
                    chars[i] = CONFUSION_TO_DIGIT[chars[i]]
                    repairs_made += 1
        repaired = "".join(chars)[:9]
        is_valid = 6 <= len(repaired) <= 9 and repaired.isdigit()
        confidence = max(0.5, 1.0 - (repairs_made * 0.05)) if is_valid else 0.5
        return repaired, confidence, is_valid

    # Otherwise older pattern A12345 or AB1234
    repaired = "".join(chars)
    is_valid = bool(re.match(r"^[A-Z]{1,2}\d{4,5}$", repaired))
    return repaired, 0.9 if is_valid else 0.5, is_valid


def _repair_manitoba_dl(cleaned: str, surname_initial: str = "") -> Tuple[str, float, bool]:
    """Manitoba format: Name-derived letters then digits (12 to 13 chars) or legacy 1-letter + 12-13 digits."""
    chars = list(cleaned)
    repairs_made = 0
    if len(chars) > 14:
        chars = chars[:14]

    # Slot 0: Must be a letter
    if len(chars) >= 1:
        if chars[0].isdigit():
            if surname_initial:
                chars[0] = surname_initial
                repairs_made += 1
            elif chars[0] in CONFUSION_TO_LETTER:
                chars[0] = CONFUSION_TO_LETTER[chars[0]]
                repairs_made += 1

    repaired = "".join(chars)
    is_name_derived = bool(re.match(r"^[A-Z][A-Z*]{4}[A-Z][A-Z*][A-Z0-9]{5,6}$", repaired))
    is_legacy = bool(re.match(r"^[A-Z]\d{12,13}$", repaired))
    is_valid = is_name_derived or is_legacy
    confidence = max(0.5, 1.0 - (repairs_made * 0.05)) if is_valid else 0.5
    return repaired, confidence, is_valid


def _repair_novascotia_dl(cleaned: str, surname: str = "") -> Tuple[str, float, bool]:
    """Nova Scotia format: 5 surname chars + 9 digits (Total 14 chars)."""
    chars = list(cleaned)
    repairs_made = 0
    if len(chars) > 14:
        chars = chars[:14]

    # Slots 0 to 4: Letters (or * / space)
    clean_sur = re.sub(r"[^A-Z]", "", unicodedata.normalize("NFKD", surname.strip().upper())) if surname else ""
    for i in range(min(5, len(chars))):
        if chars[i].isdigit():
            if i < len(clean_sur):
                chars[i] = clean_sur[i]
                repairs_made += 1
            elif chars[i] in CONFUSION_TO_LETTER:
                chars[i] = CONFUSION_TO_LETTER[chars[i]]
                repairs_made += 1

    # Slots 5 to 13: Digits
    for i in range(5, len(chars)):
        if not chars[i].isdigit():
            if chars[i] in CONFUSION_TO_DIGIT:
                chars[i] = CONFUSION_TO_DIGIT[chars[i]]
                repairs_made += 1

    repaired = "".join(chars)
    is_valid = len(repaired) == 14 and bool(re.match(r"^[A-Z][A-Z* ]{4}\d{9}$", repaired))
    confidence = max(0.5, 1.0 - (repairs_made * 0.05)) if is_valid else 0.5
    return repaired, confidence, is_valid


def _repair_newfoundland_dl(cleaned: str, surname_initial: str = "") -> Tuple[str, float, bool]:
    """Newfoundland format: 1 Letter + 9 Digits (Total 10 chars)."""
    chars = list(cleaned)
    repairs_made = 0
    if len(chars) > 10:
        chars = chars[:10]

    # Slot 0: Letter
    if len(chars) >= 1:
        if chars[0].isdigit():
            if surname_initial:
                chars[0] = surname_initial
                repairs_made += 1
            elif chars[0] in CONFUSION_TO_LETTER:
                chars[0] = CONFUSION_TO_LETTER[chars[0]]
                repairs_made += 1

    # Slots 1 to 9: Digits
    for i in range(1, len(chars)):
        if not chars[i].isdigit():
            if chars[i] in CONFUSION_TO_DIGIT:
                chars[i] = CONFUSION_TO_DIGIT[chars[i]]
                repairs_made += 1

    repaired = "".join(chars)
    is_valid = len(repaired) == 10 and repaired[0].isalpha() and repaired[1:].isdigit()
    confidence = max(0.5, 1.0 - (repairs_made * 0.05)) if is_valid else 0.5
    return repaired, confidence, is_valid


def _repair_numeric_dl(cleaned: str, expected_len: int) -> Tuple[str, float, bool]:
    """Repair strictly numeric licenses of an exact length."""
    chars = list(cleaned)
    repairs_made = 0

    if len(chars) > expected_len:
        chars = chars[:expected_len]

    for i in range(len(chars)):
        if not chars[i].isdigit():
            if chars[i] in CONFUSION_TO_DIGIT:
                chars[i] = CONFUSION_TO_DIGIT[chars[i]]
                repairs_made += 1

    repaired = "".join(chars)
    is_valid = len(repaired) == expected_len and repaired.isdigit()
    confidence = max(0.5, 1.0 - (repairs_made * 0.05)) if is_valid else 0.5
    return repaired, confidence, is_valid


def _repair_numeric_range_dl(cleaned: str, min_len: int, max_len: int) -> Tuple[str, float, bool]:
    """Repair numeric licenses with variable length between min_len and max_len."""
    chars = list(cleaned)
    repairs_made = 0

    if len(chars) > max_len:
        chars = chars[:max_len]

    for i in range(len(chars)):
        if not chars[i].isdigit():
            if chars[i] in CONFUSION_TO_DIGIT:
                chars[i] = CONFUSION_TO_DIGIT[chars[i]]
                repairs_made += 1

    repaired = "".join(chars)
    is_valid = min_len <= len(repaired) <= max_len and repaired.isdigit()
    confidence = max(0.5, 1.0 - (repairs_made * 0.05)) if is_valid else 0.5
    return repaired, confidence, is_valid


def format_canadian_dl(dl: str, province: str) -> str:
    """
    Formats a cleaned Canadian DL into its canonical display format with standard dashes.
    Example:
        Ontario: D12345678901234 -> D1234-56789-01234
        Quebec: L123415029001 -> L1234-150290-01
        Alberta: 123456789 -> 123456-789
        Manitoba: PUBLIJQ123AB -> PUBLI-JQ123-AB
        Nova Scotia: PUBLI150290123 -> PUBLI-150290-123
        Newfoundland: P900215123 -> P-900215-123
    """
    prov = normalize_province_code(province)
    cleaned = clean_dl_string(dl)
    if prov == "ON" and len(cleaned) == 15:
        return f"{cleaned[0:5]}-{cleaned[5:10]}-{cleaned[10:15]}"
    elif prov == "QC" and len(cleaned) == 13:
        return f"{cleaned[0:5]}-{cleaned[5:11]}-{cleaned[11:13]}"
    elif prov == "AB" and len(cleaned) == 9 and cleaned.isdigit():
        return f"{cleaned[0:6]}-{cleaned[6:9]}"
    elif prov == "MB" and len(cleaned) >= 12:
        return f"{cleaned[0:5]}-{cleaned[5:10]}-{cleaned[10:]}"
    elif prov == "NS" and len(cleaned) == 14:
        return f"{cleaned[0:5]}-{cleaned[5:11]}-{cleaned[11:14]}"
    elif prov == "NL" and len(cleaned) == 10:
        return f"{cleaned[0]}-{cleaned[1:7]}-{cleaned[7:10]}"
    return cleaned
