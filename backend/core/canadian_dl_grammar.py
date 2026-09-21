import re
from typing import Dict, List, Optional, Tuple

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

# Standard AAMVA (American Association of Motor Vehicle Administrators) field identifiers
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

# Standard Canadian Province/Territory Codes
CANADIAN_PROVINCE_CODES = {
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


def normalize_province_code(prov: Optional[str]) -> str:
    """Normalizes province names or codes (e.g. 'Ontario' -> 'ON')."""
    if not prov:
        return "ON"
    cleaned = prov.strip().upper()
    if cleaned in CANADIAN_PROVINCE_CODES:
        return cleaned
    # Check if full name provided
    for code, name in CANADIAN_PROVINCE_CODES.items():
        if cleaned == name.upper() or cleaned in name.upper():
            return code
    return "ON"


def clean_dl_string(raw: str) -> str:
    """Strips all spaces, dashes, dots, and non-alphanumeric punctuation."""
    if not raw:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", raw).upper()


def repair_provincial_dl(
    raw_dl: str,
    province: Optional[str] = "ON",
    surname: Optional[str] = "",
) -> Tuple[str, float, bool]:
    """
    Validates and deterministically auto-repairs an OCR-extracted driving license number
    against the official Canadian provincial specification.

    Returns:
        (repaired_dl, confidence_score, is_valid)
    """
    prov = normalize_province_code(province)
    cleaned = clean_dl_string(raw_dl)
    if not cleaned:
        return "", 0.0, False

    surname_initial = surname.strip()[0].upper() if surname and surname.strip() else ""

    if prov == "ON":
        # Ontario: 1 Letter (matches surname) + 14 digits (Total 15 chars)
        # Format on card: A0000-00000-00000
        return _repair_ontario_dl(cleaned, surname_initial)

    elif prov == "BC":
        # British Columbia: Exactly 7 digits
        return _repair_bc_dl(cleaned)

    elif prov == "AB":
        # Alberta: 7 to 9 digits (sometimes optional 3-digit suffix e.g. 123456-789)
        return _repair_alberta_dl(cleaned)

    elif prov == "QC":
        # Quebec: 1 Letter (matches surname) + 12 digits (Total 13 chars)
        return _repair_quebec_dl(cleaned, surname_initial)

    elif prov == "SK":
        # Saskatchewan: Exactly 8 digits
        return _repair_numeric_dl(cleaned, expected_len=8)

    elif prov == "MB":
        # Manitoba: 1 Letter + 13 digits (Total 14 chars)
        return _repair_letter_then_digits(cleaned, expected_digits=13, surname_initial=surname_initial)

    elif prov == "NS":
        # Nova Scotia: 1 Letter + 13 digits (Total 14 chars) or 7 digits
        if len(cleaned) <= 8:
            return _repair_numeric_dl(cleaned, expected_len=7)
        return _repair_letter_then_digits(cleaned, expected_digits=13, surname_initial=surname_initial)

    elif prov in ("NB", "NL", "PE", "PEI", "YT", "NT", "NU"):
        # Default numeric or letter-prefixed standard lengths
        if prov == "NL":
            return _repair_letter_then_digits(cleaned, expected_digits=9, surname_initial=surname_initial)
        return _repair_numeric_dl(cleaned, expected_len=7)

    # Fallback generic repair
    return cleaned, 0.8, True


def _repair_ontario_dl(cleaned: str, surname_initial: str = "") -> Tuple[str, float, bool]:
    """
    Ontario format: 1 Letter + 14 Digits.
    First letter is the initial of the driver's surname.
    Remaining 14 positions are strictly digits.
    """
    chars = list(cleaned)
    repairs_made = 0

    # Ensure length is 15 if possible
    if len(chars) > 15:
        chars = chars[:15]

    # Slot 0: Must be a letter
    if len(chars) >= 1:
        if chars[0].isdigit():
            # If we know the surname initial, use it; otherwise cast digit to letter
            if surname_initial and surname_initial.isalpha():
                chars[0] = surname_initial
                repairs_made += 1
            elif chars[0] in CONFUSION_TO_LETTER:
                chars[0] = CONFUSION_TO_LETTER[chars[0]]
                repairs_made += 1
        elif surname_initial and surname_initial.isalpha() and chars[0] != surname_initial:
            # If the OCR picked up an ambiguous letter (e.g. 'O' vs 'D'), and surname is 'D', prefer surname initial
            if chars[0] in ("O", "Q", "B", "C") and surname_initial in ("D", "O", "B"):
                chars[0] = surname_initial
                repairs_made += 1

    # Slots 1 to 14: Must be digits
    for i in range(1, len(chars)):
        ch = chars[i]
        if not ch.isdigit():
            if ch in CONFUSION_TO_DIGIT:
                chars[i] = CONFUSION_TO_DIGIT[ch]
                repairs_made += 1
            else:
                # If cannot convert, keep as is
                pass

    repaired = "".join(chars)
    is_valid = len(repaired) == 15 and repaired[0].isalpha() and repaired[1:].isdigit()
    confidence = max(0.5, 1.0 - (repairs_made * 0.05)) if is_valid else 0.5
    return repaired, confidence, is_valid


def _repair_bc_dl(cleaned: str) -> Tuple[str, float, bool]:
    """British Columbia: Exactly 7 digits."""
    return _repair_numeric_dl(cleaned, expected_len=7)


def _repair_alberta_dl(cleaned: str) -> Tuple[str, float, bool]:
    """Alberta: 7 to 9 digits."""
    chars = list(cleaned)
    repairs_made = 0

    # Convert letters to digits
    for i in range(len(chars)):
        if not chars[i].isdigit():
            if chars[i] in CONFUSION_TO_DIGIT:
                chars[i] = CONFUSION_TO_DIGIT[chars[i]]
                repairs_made += 1

    repaired = "".join(chars)
    is_valid = 7 <= len(repaired) <= 9 and repaired.isdigit()
    confidence = max(0.5, 1.0 - (repairs_made * 0.05)) if is_valid else 0.5
    return repaired, confidence, is_valid


def _repair_quebec_dl(cleaned: str, surname_initial: str = "") -> Tuple[str, float, bool]:
    """Quebec: 1 Letter + 12 digits (Total 13 chars)."""
    return _repair_letter_then_digits(cleaned, expected_digits=12, surname_initial=surname_initial)


def _repair_letter_then_digits(
    cleaned: str, expected_digits: int, surname_initial: str = ""
) -> Tuple[str, float, bool]:
    """Generic repair for: 1 Letter + N Digits."""
    chars = list(cleaned)
    total_len = expected_digits + 1
    repairs_made = 0

    if len(chars) > total_len:
        chars = chars[:total_len]

    # Slot 0: Letter
    if len(chars) >= 1:
        if chars[0].isdigit():
            if surname_initial and surname_initial.isalpha():
                chars[0] = surname_initial
                repairs_made += 1
            elif chars[0] in CONFUSION_TO_LETTER:
                chars[0] = CONFUSION_TO_LETTER[chars[0]]
                repairs_made += 1

    # Slots 1+: Digits
    for i in range(1, len(chars)):
        if not chars[i].isdigit():
            if chars[i] in CONFUSION_TO_DIGIT:
                chars[i] = CONFUSION_TO_DIGIT[chars[i]]
                repairs_made += 1

    repaired = "".join(chars)
    is_valid = len(repaired) == total_len and repaired[0].isalpha() and repaired[1:].isdigit()
    confidence = max(0.5, 1.0 - (repairs_made * 0.05)) if is_valid else 0.5
    return repaired, confidence, is_valid


def _repair_numeric_dl(cleaned: str, expected_len: int) -> Tuple[str, float, bool]:
    """Repair strictly numeric licenses."""
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


def format_canadian_dl(dl: str, province: str) -> str:
    """
    Formats a cleaned Canadian DL into its canonical display format with standard dashes.
    Example: Ontario D12345678901234 -> D1234-56789-01234
    """
    prov = normalize_province_code(province)
    cleaned = clean_dl_string(dl)
    if prov == "ON" and len(cleaned) == 15:
        return f"{cleaned[0:5]}-{cleaned[5:10]}-{cleaned[10:15]}"
    elif prov == "QC" and len(cleaned) == 13:
        return f"{cleaned[0]} {cleaned[1:5]} {cleaned[5:11]} {cleaned[11:13]}"
    elif prov == "MB" and len(cleaned) == 14:
        return f"{cleaned[0:3]}-{cleaned[3:5]}-{cleaned[5:8]}-{cleaned[8:14]}"
    return cleaned
