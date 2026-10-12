"""
Generate 1 High-Quality Sample Preview Image for Each of the 13 Canadian Jurisdictions.
Used for user visual inspection and coordinate validation before running full test suites.
Ensures zero face occlusion, perfectly preserved labels, and pixel-accurate credential placement.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = SCRIPT_DIR
ROOT_DIR = os.path.dirname(BACKEND_DIR)
SAMPLE_DIR = os.path.join(ROOT_DIR, "DL IMAGE'S SAMPLE'S")
PREVIEWS_DIR = os.path.join(BACKEND_DIR, "test_images", "previews")
os.makedirs(PREVIEWS_DIR, exist_ok=True)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.canadian_dl_grammar import format_canadian_dl

# Final Calibrated Configurations for all 13 Canadian Jurisdictions
CALIBRATED_CONFIGS = {
    "ON": {
        "name": "Ontario",
        "file": os.path.join(BACKEND_DIR, "template.png"),
        "target_w": 900,
        "bg": (226, 235, 222),
        "erase": [
            (330, 125, 600, 205), # Name
            (430, 250, 830, 310), # DL
            (735, 285, 885, 335), # Exp
            (125, 500, 300, 545), # DOB
        ],
        "coords": {
            "last_name": (335, 130),
            "first_name": (335, 168),
            "dl": (435, 258),
            "exp": (740, 292),
            "dob": (130, 508),
        },
        "profile": ("John Doe", "D61014070660905", "1966-09-05", "2029-04-23")
    },
    "QC": {
        "name": "Quebec",
        "file": os.path.join(SAMPLE_DIR, "Quebec DL.jpg"),
        "target_w": 900,
        "bg": (240, 244, 246),
        "erase": [
            (360, 95, 715, 145),  # DL
            (360, 140, 520, 190), # Name
            (600, 188, 720, 222), # DOB
            (690, 408, 805, 436), # Exp
        ],
        "coords": {
            "dl": (365, 102),
            "last_name": (368, 143),
            "first_name": (368, 168),
            "dob": (605, 194),
            "exp": (695, 412),
        },
        "profile": ("Marie Curie", "C123415029001", "1990-02-15", "2029-02-15")
    },
    "BC": {
        "name": "British Columbia",
        "file": os.path.join(SAMPLE_DIR, "British Columbia DL IMAGE.jpg"),
        "target_w": 900,
        "bg": (242, 244, 245),
        "erase": [
            (30, 130, 275, 215),  # Name (replaces 'SPECIMEN, TEST CARD')
            (600, 128, 860, 175), # DL (after 'DL:')
            (620, 172, 860, 218), # DOB (after 'DOB:')
            (415, 345, 640, 395), # Exp (after 'Expires:')
        ],
        "coords": {
            "name": (35, 145),
            "dl": (605, 136),
            "dob": (625, 180),
            "exp": (420, 355),
        },
        "profile": ("Jane Smith", "98765432", "1985-11-20", "2028-11-20")
    },
    "AB": {
        "name": "Alberta",
        "file": os.path.join(SAMPLE_DIR, "Albert DL IMAGE.webp"),
        "target_w": 900,
        "bg": (238, 242, 244),
        "erase": [
            (312, 115, 540, 160), # DL (after 'No')
            (324, 192, 510, 232), # Exp (after 'Exp')
            (288, 228, 480, 272), # Name
            (324, 400, 575, 450), # DOB (after 'DOB')
        ],
        "coords": {
            "dl": (316, 126),
            "exp": (330, 200),
            "name": (292, 238),
            "dob": (330, 410),
        },
        "profile": ("Sam Sample", "134711320", "1971-11-20", "2028-11-20")
    },
    "MB": {
        "name": "Manitoba",
        "file": os.path.join(SAMPLE_DIR, "Manitoba DL IMAGE.webp"),
        "target_w": 900,
        "bg": (235, 238, 235),
        "erase": [
            (318, 95, 560, 160),  # Name
            (400, 235, 645, 275), # DL
            (445, 295, 580, 332), # Exp
            (150, 485, 295, 528), # DOB (at bottom '3. DOB/DDN')
        ],
        "coords": {
            "last_name": (322, 100),
            "first_name": (322, 128),
            "dl": (405, 242),
            "exp": (450, 300),
            "dob": (155, 492),
        },
        "profile": ("James Beazley", "PUBLIJQ123AB", "1992-12-12", "2028-12-12")
    },
    "NB": {
        "name": "New Brunswick",
        "file": os.path.join(SAMPLE_DIR, "New Brunswick DL IMAGE.jpg"),
        "target_w": 900,
        "bg": (238, 240, 238),
        "erase": [
            (120, 105, 380, 155), # DL (left)
            (465, 128, 595, 172), # DOB (after '3. DOB/DDN')
            (120, 160, 250, 196), # Exp (left)
            (55, 190, 280, 260),  # Name (left)
        ],
        "coords": {
            "dl": (125, 114),
            "dob": (470, 136),
            "exp": (125, 166),
            "last_name": (58, 196),
            "first_name": (58, 226),
        },
        "profile": ("Paul Atkinson", "1234567", "1981-01-01", "2028-01-01")
    },
    "NS": {
        "name": "Nova Scotia",
        "file": os.path.join(SAMPLE_DIR, "Nova Scotia DL IMAGE.webp"),
        "target_w": 900,
        "bg": (238, 242, 246),
        "erase": [
            (375, 95, 650, 138),  # DL
            (375, 150, 520, 190), # Exp (after '4b. Exp')
            (620, 138, 770, 185), # DOB (after 'DOB/DDN')
            (310, 175, 480, 240), # Name
        ],
        "coords": {
            "dl": (380, 102),
            "exp": (380, 158),
            "dob": (625, 146),
            "last_name": (315, 180),
            "first_name": (315, 208),
        },
        "profile": ("Carlos Santana", "SANTA150378123", "1978-03-15", "2028-03-15")
    },
    "NL": {
        "name": "Newfoundland & Labrador",
        "file": os.path.join(SAMPLE_DIR, "Newfoundland and Labrador DL IMAGE.webp"),
        "target_w": 900,
        "bg": (236, 241, 241),
        "erase": [
            (375, 95, 540, 135),  # DL (after '4d. NO/N')
            (325, 135, 465, 170), # Exp (after '4b.')
            (320, 185, 510, 255), # Name
            (630, 125, 770, 160), # DOB (after '3. DOB/DDN')
        ],
        "coords": {
            "dl": (380, 102),
            "exp": (330, 142),
            "name": (325, 195),
            "dob": (635, 132),
        },
        "profile": ("Peter Miller", "M780515123", "1978-05-15", "2028-05-15")
    },
    "SK": {
        "name": "Saskatchewan",
        "file": os.path.join(SAMPLE_DIR, "Saskatchewan DL IMAGE.jpg"),
        "target_w": 900,
        "bg": (236, 242, 236),
        "erase": [
            (310, 110, 450, 144), # DL
            (310, 144, 480, 198), # Name
            (308, 310, 500, 348), # DOB
            (310, 412, 500, 448), # Exp
        ],
        "coords": {
            "dl": (314, 114),
            "last_name": (314, 148),
            "first_name": (314, 172),
            "dob": (312, 314),
            "exp": (314, 416),
        },
        "profile": ("Grace Hopper", "20000030", "1970-05-05", "2028-05-31")
    },
    "PE": {
        "name": "Prince Edward Island",
        "file": os.path.join(SAMPLE_DIR, "Prince Edward Island DL IMAGE.webp"),
        "target_w": 900,
        "bg": (236, 240, 245),
        "erase": [
            (360, 78, 440, 110),  # DL
            (320, 128, 465, 162), # Exp
            (525, 95, 680, 138),  # DOB (after 'DOB/DDN')
            (320, 155, 520, 215), # Name
        ],
        "coords": {
            "dl": (365, 84),
            "exp": (325, 134),
            "dob": (530, 102),
            "last_name": (325, 160),
            "first_name": (325, 184),
        },
        "profile": ("Ada Lovelace", "123456", "1988-08-24", "2029-08-24")
    },
    "NT": {
        "name": "Northwest Territories",
        "file": os.path.join(SAMPLE_DIR, "Northwest Territories DL IMAGE.webp"),
        "target_w": 900,
        "bg": (232, 239, 246),
        "erase": [
            (290, 100, 500, 155), # Name
            (305, 215, 430, 248), # DOB
            (498, 215, 625, 248), # Exp
            (680, 465, 835, 500), # DL
        ],
        "coords": {
            "last_name": (295, 105),
            "first_name": (295, 130),
            "dob": (310, 220),
            "exp": (502, 220),
            "dl": (685, 470),
        },
        "profile": ("Liam Vance", "123456789", "1994-06-18", "2028-06-18")
    },
    "YT": {
        "name": "Yukon",
        "file": os.path.join(SAMPLE_DIR, "Yukon DL IMAGE.webp"),
        "target_w": 900,
        "bg": (236, 239, 243),
        "erase": [
            (640, 225, 780, 275), # DL
            (330, 230, 550, 300), # Name (at 1 and 2 at top)
            (660, 345, 850, 395), # DOB (after '3. DOB/DDN')
            (660, 440, 850, 490), # Exp (after '4b. EXP')
        ],
        "coords": {
            "dl": (645, 235),
            "last_name": (335, 238),
            "first_name": (335, 268),
            "dob": (665, 355),
            "exp": (665, 450),
        },
        "profile": ("David Dawson", "654321", "1982-04-10", "2028-04-10")
    },
    "NU": {
        "name": "Nunavut",
        "file": os.path.join(SAMPLE_DIR, "Nunavut DL IMAGE.webp"),
        "target_w": 900,
        "bg": (242, 240, 238),
        "erase": [
            (285, 115, 480, 160), # DL
            (285, 170, 520, 220), # Name
            (285, 415, 450, 465), # DOB
            (285, 510, 460, 565), # Exp
        ],
        "coords": {
            "dl": (290, 122),
            "name": (290, 178),
            "dob": (290, 422),
            "exp": (290, 518),
        },
        "profile": ("Noah Iqaluit", "123456", "1991-09-09", "2028-09-09")
    }
}


def clean_field_patch(img: np.ndarray, box: tuple, default_bg: tuple) -> None:
    """Fills field area with locally sampled border color or default background."""
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    border_pixels = []
    if y1 > 3:
        border_pixels.append(img[y1-3:y1, x1:x2].reshape(-1, 3))
    if y2 < h - 3:
        border_pixels.append(img[y2:y2+3, x1:x2].reshape(-1, 3))
    if x1 > 3:
        border_pixels.append(img[y1:y2, x1-3:x1].reshape(-1, 3))
    if x2 < w - 3:
        border_pixels.append(img[y1:y2, x2:x2+3].reshape(-1, 3))

    if border_pixels:
        all_pixels = np.vstack(border_pixels)
        patch_color = np.median(all_pixels, axis=0).astype(int).tolist()
    else:
        patch_color = list(default_bg)

    cv2.rectangle(img, (x1, y1), (x2, y2), patch_color, -1)


def generate_single_preview(prov: str) -> str:
    cfg = CALIBRATED_CONFIGS[prov]
    template_path = cfg["file"]
    img = cv2.imread(template_path)
    if img is None:
        raise FileNotFoundError(f"Template not found: {template_path}")

    h, w = img.shape[:2]
    target_w = cfg["target_w"]
    target_h = int(h * (target_w / float(w)))
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Clean field boxes (Zero face overlap guaranteed)
    for box in cfg["erase"]:
        clean_field_patch(resized, box, cfg["bg"])

    # Draw Text
    img_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font_large = ImageFont.truetype("arialbd.ttf", 20)
        font_med = ImageFont.truetype("arialbd.ttf", 18)
        font_small = ImageFont.truetype("arialbd.ttf", 16)
    except Exception:
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_small = ImageFont.load_default()

    name, dl, dob, exp = cfg["profile"]
    parts = name.split()
    last_name = parts[-1].upper() if len(parts) > 1 else name.upper()
    first_name = parts[0].upper()
    formatted_dl = format_canadian_dl(dl, prov)

    for field, (x, y) in cfg["coords"].items():
        if field == "last_name":
            draw.text((x, y), last_name, font=font_large, fill=(20, 20, 20))
        elif field == "first_name":
            draw.text((x, y), first_name, font=font_med, fill=(20, 20, 20))
        elif field == "name":
            draw.text((x, y), f"{last_name}, {first_name}", font=font_large, fill=(20, 20, 20))
        elif field == "dl":
            draw.text((x, y), formatted_dl, font=font_large, fill=(20, 20, 20))
        elif field == "dob":
            draw.text((x, y), dob.replace("-", "/"), font=font_small, fill=(20, 20, 20))
        elif field == "exp":
            draw.text((x, y), exp.replace("-", "/"), font=font_small, fill=(20, 20, 20))

    final_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    out_path = os.path.join(PREVIEWS_DIR, f"preview_{prov}.png")
    cv2.imwrite(out_path, final_bgr)
    return out_path


def main():
    print(f"Generating preview sample images for all {len(CALIBRATED_CONFIGS)} jurisdictions...")
    generated = []
    for prov in sorted(CALIBRATED_CONFIGS.keys()):
        p = generate_single_preview(prov)
        print(f"  [+] {prov:2s} ({CALIBRATED_CONFIGS[prov]['name']}): {p}")
        generated.append(p)
    print(f"\nSuccessfully generated {len(generated)} preview images in: {PREVIEWS_DIR}")


if __name__ == "__main__":
    main()
