"""
Provincial Templates & Ground Truth Credential Configurations.
Calibrated for all 13 Canadian provinces and territories using real sample licenses.
Zero face occlusion, perfectly preserved labels, and pixel-accurate credential placement.
"""

import os
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(BACKEND_DIR)
SAMPLE_DIR = os.path.join(ROOT_DIR, "DL IMAGE'S SAMPLE'S")

PROVINCIAL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ON": {
        "name": "Ontario",
        "file": os.path.join(BACKEND_DIR, "template.png"),
        "target_w": 900,
        "erase": [
            (330, 125, 600, 205),
            (430, 250, 830, 310),
            (735, 285, 885, 335),
            (125, 500, 300, 545),
        ],
        "coords": {
            "last_name": (335, 130),
            "first_name": (335, 168),
            "dl": (435, 258),
            "exp": (740, 292),
            "dob": (130, 508),
        },
        "bg": (226, 235, 222),
        "profiles": [
            ("John Doe", "D61014070660905", "1966-09-05", "2029-04-23"),
            ("Alice Smith", "S12345678850412", "1985-04-12", "2028-04-12"),
            ("Michael Brown", "B54321098781125", "1978-11-25", "2029-11-25"),
            ("Emma Wilson", "W98765432920708", "1992-07-08", "2028-07-08"),
            ("David Taylor", "T11223344800315", "1980-03-15", "2029-03-15"),
            ("Sarah Davis", "D99887766951030", "1995-10-30", "2028-10-30"),
            ("James Miller", "M44556677721205", "1972-12-05", "2029-12-05"),
            ("Emily Anderson", "A12341234880820", "1988-08-20", "2028-08-20"),
        ]
    },
    "QC": {
        "name": "Quebec",
        "file": os.path.join(SAMPLE_DIR, "Quebec DL.jpg"),
        "target_w": 900,
        "erase": [
            (360, 95, 715, 145),
            (360, 140, 520, 190),
            (600, 188, 720, 222),
            (690, 408, 805, 436),
        ],
        "coords": {
            "dl": (365, 102),
            "last_name": (368, 143),
            "first_name": (368, 168),
            "dob": (605, 194),
            "exp": (695, 412),
        },
        "bg": (240, 244, 246),
        "profiles": [
            ("Marie Curie", "C123415029001", "1990-02-15", "2029-02-15"),
            ("Jean Tremblay", "T567812048502", "1985-04-12", "2028-04-12"),
            ("Pierre Gagnon", "G901225117803", "1978-11-25", "2029-11-25"),
            ("Sophie Roy", "R345608079204", "1992-07-08", "2028-07-08"),
            ("Marc Bouchard", "B789015038005", "1980-03-15", "2029-03-15"),
            ("Isabelle Gauthier", "G234530109506", "1995-10-30", "2028-10-30"),
            ("Francois Morin", "M678905127207", "1972-12-05", "2029-12-05"),
            ("Helene Lavoie", "L012320088808", "1988-08-20", "2028-08-20"),
        ]
    },
    "BC": {
        "name": "British Columbia",
        "file": os.path.join(SAMPLE_DIR, "British Columbia DL IMAGE.jpg"),
        "target_w": 900,
        "erase": [
            (30, 130, 275, 215),
            (600, 128, 860, 175),
            (620, 172, 860, 218),
            (415, 345, 640, 395),
        ],
        "coords": {
            "name": (35, 145),
            "dl": (605, 136),
            "dob": (625, 180),
            "exp": (420, 355),
        },
        "bg": (242, 244, 245),
        "profiles": [
            ("Jane Smith", "98765432", "1985-11-20", "2028-11-20"),
            ("Robert Chen", "12345678", "1990-05-15", "2029-05-15"),
            ("Liam Wilson", "87654321", "1982-08-22", "2028-08-22"),
            ("Olivia Taylor", "23456789", "1995-01-30", "2029-01-30"),
            ("Noah Davis", "76543210", "1988-09-10", "2028-09-10"),
            ("Emma Martin", "34567890", "1993-04-05", "2029-04-05"),
            ("William Clark", "65432109", "1979-12-18", "2028-12-18"),
            ("Sophia Hall", "45678901", "1986-07-27", "2029-07-27"),
        ]
    },
    "AB": {
        "name": "Alberta",
        "file": os.path.join(SAMPLE_DIR, "Albert DL IMAGE.webp"),
        "target_w": 900,
        "erase": [
            (312, 115, 540, 160),
            (324, 192, 510, 232),
            (288, 228, 480, 272),
            (324, 400, 575, 450),
        ],
        "coords": {
            "dl": (316, 126),
            "exp": (330, 200),
            "name": (292, 238),
            "dob": (330, 410),
        },
        "bg": (238, 242, 244),
        "profiles": [
            ("Sam Sample", "134711320", "1971-11-20", "2028-11-20"),
            ("Gordon Vance", "987654321", "1985-06-15", "2029-06-15"),
            ("Travis Scott", "123456789", "1990-03-22", "2028-03-22"),
            ("Amber Heard", "456789123", "1988-09-12", "2029-09-12"),
            ("Brett Wilson", "789123456", "1976-12-05", "2028-12-05"),
            ("Chelsea Green", "321654987", "1994-04-18", "2029-04-18"),
            ("Darren Young", "654987321", "1982-08-30", "2028-08-30"),
            ("Fiona Gallagher", "159753468", "1991-01-25", "2029-01-25"),
        ]
    },
    "MB": {
        "name": "Manitoba",
        "file": os.path.join(SAMPLE_DIR, "Manitoba DL IMAGE.webp"),
        "target_w": 900,
        "erase": [
            (318, 95, 560, 160),
            (400, 235, 645, 275),
            (445, 295, 580, 332),
            (150, 485, 295, 528),
        ],
        "coords": {
            "last_name": (322, 100),
            "first_name": (322, 128),
            "dl": (405, 242),
            "exp": (450, 300),
            "dob": (155, 492),
        },
        "bg": (235, 238, 235),
        "profiles": [
            ("James Beazley", "BEAZLJB123AB", "1992-12-12", "2028-12-12"),
            ("Akira Kurosawa", "KUROSAK123CD", "1985-05-10", "2029-05-10"),
            ("Bradley Cooper", "COOPEBC678EF", "1980-08-15", "2028-08-15"),
            ("Diane Kruger", "KRUGEDK135GH", "1991-02-20", "2029-02-20"),
            ("Ethan Hawke", "HAWKEEH246IJ", "1978-11-06", "2028-11-06"),
            ("Freida Pinto", "PINTOFP987KL", "1989-10-18", "2029-10-18"),
            ("George Lucas", "LUCASGL543MN", "1974-05-14", "2028-05-14"),
            ("Helen Mirren", "MIRREHM112OP", "1983-07-26", "2029-07-26"),
        ]
    },
    "NB": {
        "name": "New Brunswick",
        "file": os.path.join(SAMPLE_DIR, "New Brunswick DL IMAGE.jpg"),
        "target_w": 900,
        "erase": [
            (120, 105, 380, 155),
            (465, 128, 595, 172),
            (120, 160, 250, 196),
            (55, 190, 280, 260),
        ],
        "coords": {
            "dl": (125, 114),
            "dob": (470, 136),
            "exp": (125, 166),
            "last_name": (58, 196),
            "first_name": (58, 226),
        },
        "bg": (238, 240, 238),
        "profiles": [
            ("Paul Atkinson", "1234567", "1981-01-01", "2028-01-01"),
            ("Martin LeBlanc", "7654321", "1986-06-15", "2029-06-15"),
            ("Claude Cormier", "2345678", "1990-09-20", "2028-09-20"),
            ("Denise Richard", "8765432", "1977-04-12", "2029-04-12"),
            ("Eric Savoie", "3456789", "1989-12-05", "2028-12-05"),
            ("Giselle Landry", "9876543", "1993-03-25", "2029-03-25"),
            ("Henri Daigle", "4567890", "1982-08-18", "2028-08-18"),
            ("Jeanne Doucet", "6543210", "1995-11-30", "2029-11-30"),
        ]
    },
    "NS": {
        "name": "Nova Scotia",
        "file": os.path.join(SAMPLE_DIR, "Nova Scotia DL IMAGE.webp"),
        "target_w": 900,
        "erase": [
            (375, 95, 650, 138),
            (375, 150, 520, 190),
            (620, 138, 770, 185),
            (310, 175, 480, 240),
        ],
        "coords": {
            "dl": (380, 102),
            "exp": (380, 158),
            "dob": (625, 146),
            "last_name": (315, 180),
            "first_name": (315, 208),
        },
        "bg": (238, 242, 246),
        "profiles": [
            ("Carlos Santana", "SANTA150378123", "1978-03-15", "2028-03-15"),
            ("Gill Mahi", "MAHI*020280456", "1980-02-02", "2029-02-02"),
            ("Angus Young", "YOUNG120485789", "1985-04-12", "2028-04-12"),
            ("Brenda Fraser", "FRASE251178321", "1978-11-25", "2029-11-25"),
            ("Colin Campbell", "CAMPB080792654", "1992-07-08", "2028-07-08"),
            ("Donna MacLeod", "MACLE301095987", "1995-10-30", "2029-10-30"),
            ("Evan MacDonald", "MACDO051272147", "1972-12-05", "2028-12-05"),
            ("Faith Morrison", "MORRI200888258", "1988-08-20", "2029-08-20"),
        ]
    },
    "NL": {
        "name": "Newfoundland & Labrador",
        "file": os.path.join(SAMPLE_DIR, "Newfoundland and Labrador DL IMAGE.webp"),
        "target_w": 900,
        "erase": [
            (375, 95, 540, 135),
            (325, 135, 465, 170),
            (320, 185, 510, 255),
            (630, 125, 770, 160),
        ],
        "coords": {
            "dl": (380, 102),
            "exp": (330, 142),
            "name": (325, 195),
            "dob": (635, 132),
        },
        "bg": (236, 241, 241),
        "profiles": [
            ("Peter Miller", "M780515123", "1978-05-15", "2028-05-15"),
            ("Gordon Pike", "P990917999", "1999-09-17", "2029-09-17"),
            ("Arthur Brown", "B850412456", "1985-04-12", "2028-04-12"),
            ("Cathy Smith", "S920708789", "1992-07-08", "2029-07-08"),
            ("Donald Ryan", "R800315321", "1980-03-15", "2028-03-15"),
            ("Ellen Power", "P951030654", "1995-10-30", "2029-10-30"),
            ("Frank Walsh", "W721205987", "1972-12-05", "2028-12-05"),
            ("Grace Murphy", "M880820147", "1988-08-20", "2029-08-20"),
        ]
    },
    "SK": {
        "name": "Saskatchewan",
        "file": os.path.join(SAMPLE_DIR, "Saskatchewan DL IMAGE.jpg"),
        "target_w": 900,
        "erase": [
            (310, 110, 450, 144),
            (310, 144, 480, 198),
            (308, 310, 500, 348),
            (310, 412, 500, 448),
        ],
        "coords": {
            "dl": (314, 114),
            "last_name": (314, 148),
            "first_name": (314, 172),
            "dob": (312, 314),
            "exp": (314, 416),
        },
        "bg": (236, 242, 236),
        "profiles": [
            ("Grace Hopper", "20000030", "1970-05-05", "2028-05-31"),
            ("Scott Sample", "87654321", "1983-08-15", "2029-08-31"),
            ("Barry Bonds", "12345678", "1985-04-12", "2028-04-30"),
            ("Cindy Crawford", "98765432", "1992-07-08", "2029-07-31"),
            ("Derek Jeter", "34567890", "1980-03-15", "2028-03-31"),
            ("Evelyn Hart", "76543210", "1995-10-30", "2029-10-31"),
            ("Gordie Howe", "45678901", "1972-12-05", "2028-12-31"),
            ("Holly Hunter", "65432109", "1988-08-20", "2029-08-31"),
        ]
    },
    "PE": {
        "name": "Prince Edward Island",
        "file": os.path.join(SAMPLE_DIR, "Prince Edward Island DL IMAGE.webp"),
        "target_w": 900,
        "erase": [
            (360, 78, 440, 110),
            (320, 128, 465, 162),
            (525, 95, 680, 138),
            (320, 155, 520, 215),
        ],
        "coords": {
            "dl": (365, 84),
            "exp": (325, 134),
            "dob": (530, 102),
            "last_name": (325, 160),
            "first_name": (325, 184),
        },
        "bg": (236, 240, 245),
        "profiles": [
            ("Ada Lovelace", "123456", "1988-08-24", "2029-08-24"),
            ("Lucy Montgomery", "654321", "1985-04-12", "2028-04-12"),
            ("Anne Shirley", "234567", "1992-07-08", "2029-07-08"),
            ("Gilbert Blythe", "765432", "1980-03-15", "2028-03-15"),
            ("Matthew Cuthbert", "345678", "1975-11-20", "2029-11-20"),
            ("Marilla Cuthbert", "876543", "1978-06-18", "2028-06-18"),
            ("Diana Barry", "456789", "1994-09-30", "2029-09-30"),
            ("Rachel Lynde", "987654", "1972-12-05", "2028-12-05"),
        ]
    },
    "NT": {
        "name": "Northwest Territories",
        "file": os.path.join(SAMPLE_DIR, "Northwest Territories DL IMAGE.webp"),
        "target_w": 900,
        "erase": [
            (290, 100, 500, 155),
            (305, 215, 430, 248),
            (498, 215, 625, 248),
            (680, 465, 835, 500),
        ],
        "coords": {
            "last_name": (295, 105),
            "first_name": (295, 130),
            "dob": (310, 220),
            "exp": (502, 220),
            "dl": (685, 470),
        },
        "bg": (232, 239, 246),
        "profiles": [
            ("Liam Vance", "123456789", "1994-06-18", "2028-06-18"),
            ("Ethan Frost", "987654321", "1987-03-22", "2029-03-22"),
            ("Chloe Winter", "234567891", "1991-09-14", "2028-09-14"),
            ("Mason Cole", "876543219", "1983-11-05", "2029-11-05"),
            ("Aurora Borealis", "345678912", "1996-01-20", "2028-01-20"),
            ("Logan Pine", "765432198", "1979-08-30", "2029-08-30"),
            ("Maya Snow", "456789123", "1993-04-12", "2028-04-12"),
            ("Hunter Birch", "654321987", "1985-07-25", "2029-07-25"),
        ]
    },
    "YT": {
        "name": "Yukon",
        "file": os.path.join(SAMPLE_DIR, "Yukon DL IMAGE.webp"),
        "target_w": 900,
        "erase": [
            (640, 225, 780, 275),
            (330, 230, 550, 300),
            (660, 345, 850, 395),
            (660, 440, 850, 490),
        ],
        "coords": {
            "dl": (645, 235),
            "last_name": (335, 238),
            "first_name": (335, 268),
            "dob": (665, 355),
            "exp": (665, 450),
        },
        "bg": (236, 239, 243),
        "profiles": [
            ("David Dawson", "654321", "1982-04-10", "2028-04-10"),
            ("Jack London", "123456", "1986-09-15", "2029-09-15"),
            ("Robert Service", "234567", "1979-01-16", "2028-01-16"),
            ("Sam McGee", "765432", "1990-12-04", "2029-12-04"),
            ("Klondike Kate", "345678", "1984-07-28", "2028-07-28"),
            ("Dan McGrew", "876543", "1977-03-09", "2029-03-09"),
            ("Yukon Eric", "456789", "1992-06-21", "2028-06-21"),
            ("Martha Black", "987654", "1988-11-14", "2029-11-14"),
        ]
    },
    "NU": {
        "name": "Nunavut",
        "file": os.path.join(SAMPLE_DIR, "Nunavut DL IMAGE.webp"),
        "target_w": 900,
        "erase": [
            (285, 115, 480, 160),
            (285, 170, 520, 220),
            (285, 415, 450, 465),
            (285, 510, 460, 565),
        ],
        "coords": {
            "dl": (290, 122),
            "name": (290, 178),
            "dob": (290, 422),
            "exp": (290, 518),
        },
        "bg": (242, 240, 238),
        "profiles": [
            ("Noah Iqaluit", "123456", "1991-09-09", "2028-09-09"),
            ("Mary Simon", "654321", "1984-08-21", "2029-08-21"),
            ("Paul Quassa", "234567", "1978-01-12", "2028-01-12"),
            ("Eva Aariak", "765432", "1986-06-10", "2029-06-10"),
            ("Peter Taptuna", "345678", "1980-10-25", "2028-10-25"),
            ("Nellie Cournoyea", "876543", "1975-03-04", "2029-03-04"),
            ("Tagak Curley", "456789", "1982-12-18", "2028-12-18"),
            ("Susan Aglukark", "987654", "1989-05-15", "2029-05-15"),
        ]
    }
}
