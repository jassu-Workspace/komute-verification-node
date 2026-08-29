import os
import sys
import time
import math

# Thread clamping to prevent 98% CPU spike
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["ORT_INTRA_OP_NUM_THREADS"] = "2"
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import cv2
import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(2)

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, on_color=None, attrs=None):
        return text

from app.schemas import PersonalInfo, LicenseDetails
from core.dl_ocr import dl_ocr_engine

process = psutil.Process(os.getpid())


def get_system_metrics():
    """Returns (ram_mb, cpu_percent)."""
    try:
        ram = process.memory_info().rss / (1024 * 1024)
        cpu = psutil.cpu_percent(interval=None)
        return ram, cpu
    except Exception:
        return 0.0, 0.0


def generate_base_license(name, dl_number, dob, expiry, province):
    template_path = os.path.join(SCRIPT_DIR, 'template.png')
    img = cv2.imread(template_path)
    if img is None:
        img = np.full((351, 569, 3), 240, dtype=np.uint8)

    parts = name.split()
    last_name = parts[-1].upper() if len(parts) > 1 else name.upper()
    first_name = parts[0].upper()

    dob_fmt = dob.replace('-', '/')
    exp_fmt = expiry.replace('-', '/')

    # Erase field bounding boxes with license background tint
    cv2.rectangle(img, (215, 80), (350, 130), (225, 235, 215), -1)
    cv2.rectangle(img, (275, 160), (520, 190), (235, 245, 225), -1)
    cv2.rectangle(img, (85, 315), (200, 345), (220, 205, 185), -1)
    cv2.rectangle(img, (470, 188), (560, 215), (240, 250, 230), -1)

    # Convert to PIL for crisp font rendering
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font_large = ImageFont.truetype("arialbd.ttf", 18)
        font_small = ImageFont.truetype("arialbd.ttf", 14)
    except IOError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((215, 85), last_name, font=font_large, fill=(0, 0, 0))
    draw.text((215, 110), first_name, font=font_large, fill=(0, 0, 0))
    draw.text((275, 165), dl_number, font=font_large, fill=(0, 0, 0))
    draw.text((85, 320), dob_fmt, font=font_small, fill=(0, 0, 0))
    draw.text((470, 190), exp_fmt, font=font_small, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def apply_blur(img, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def apply_noise(img, severity):
    noise = np.random.normal(0, severity, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_rotation(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(240, 240, 240))


def apply_glare(img):
    overlay = img.copy()
    cv2.circle(overlay, (350, 200), 100, (255, 255, 255), -1)
    return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)


def apply_low_contrast(img):
    return cv2.convertScaleAbs(img, alpha=0.5, beta=128)


def apply_perspective(img):
    pts1 = np.float32([[0, 0], [569, 0], [0, 351], [569, 351]])
    pts2 = np.float32([[40, 40], [530, 20], [20, 330], [550, 310]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (569, 351), borderValue=(240, 240, 240))


base_data = [
    ("John Doe", "D61014070660905", "1966-09-05", "2029-04-23", "Ontario"),
    ("Jane Smith", "BC987654", "1985-11-20", "2026-11-20", "British Columbia"),
    ("Robert Chen", "AB555123", "2000-01-01", "2025-01-01", "Alberta"),
    ("Marie Curie", "QC111999", "1995-07-08", "2030-07-08", "Quebec"),
    ("Carlos Santana", "NS444777", "1978-03-15", "2028-03-15", "Nova Scotia"),
    ("Akira Kurosawa", "MB222333", "1992-12-12", "2027-12-12", "Manitoba"),
    ("Ada Lovelace", "PE888000", "1988-08-24", "2029-08-24", "Prince Edward Island"),
    ("Grace Hopper", "SK666555", "1970-05-05", "2025-05-05", "Saskatchewan"),
]

test_suite = []

# 1-10: Easy
for i in range(10):
    test_suite.append({'category': 'Clean Base', 'desc': f'Easy Standard License #{i+1}', 'data': base_data[i % len(base_data)], 'ops': []})

# 11-30: Blur (Mild to Heavy)
for i in range(20):
    k = (i // 2) + 1
    test_suite.append({'category': 'Gaussian Blur', 'desc': f'Blur Level {k}', 'data': base_data[i % len(base_data)], 'ops': [('blur', k * 2)]})

# 31-50: Noise
for i in range(20):
    sev = (i + 1) * 5
    test_suite.append({'category': 'Sensor Noise', 'desc': f'Gaussian Noise Severity {sev}', 'data': base_data[i % len(base_data)], 'ops': [('noise', sev)]})

# 51-70: Rotation
for i in range(20):
    ang = (-10 + i) if i != 10 else 15
    test_suite.append({'category': 'Skew Rotation', 'desc': f'Rotation {ang} degrees', 'data': base_data[i % len(base_data)], 'ops': [('rotate', ang)]})

# 71-80: Glare & Low Contrast
for i in range(10):
    if i < 5:
        test_suite.append({'category': 'Optical Glare', 'desc': f'Intense Glare #{i+1}', 'data': base_data[i % len(base_data)], 'ops': [('glare', None)]})
    else:
        test_suite.append({'category': 'Low Contrast', 'desc': f'Low Contrast #{i+1}', 'data': base_data[i % len(base_data)], 'ops': [('contrast', None)]})

# 81-90: Perspective Skew
for i in range(10):
    test_suite.append({'category': 'Perspective Warp', 'desc': f'Perspective Skew / Angled Photo #{i+1}', 'data': base_data[i % len(base_data)], 'ops': [('perspective', None)]})

# 91-102: Combinations (BRUTAL)
for i in range(12):
    test_suite.append({'category': 'Brutal Combo', 'desc': f'BRUTAL Combo #{i+1} (Blur + Noise + Rotate + Skew)', 'data': base_data[i % len(base_data)], 'ops': [('blur', 3), ('noise', 20), ('rotate', -5), ('perspective', None)]})


def run_102_dl_tests():
    print("\n" + "=" * 100, flush=True)
    print("      DRIVING LICENSE BRUTAL TEST SUITE (102 CASES - Fine-Tuned RapidOCR ONNX)      ", flush=True)
    print("=" * 100 + "\n", flush=True)

    passed_count = 0
    total_cases = len(test_suite)
    test_img_dir = os.path.join(SCRIPT_DIR, 'test_images')
    os.makedirs(test_img_dir, exist_ok=True)

    category_stats = {}
    peak_ram = 0.0
    peak_cpu = 0.0

    t_start = time.time()

    for idx, tc in enumerate(test_suite):
        category = tc['category']
        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}
        category_stats[category]["total"] += 1

        name, dl, dob, exp, prov = tc['data']
        img = generate_base_license(name, dl, dob, exp, prov)

        ops_desc_list = []
        for op, val in tc['ops']:
            if op == 'blur':
                img = apply_blur(img, val)
                ops_desc_list.append(f"blur(k={val})")
            elif op == 'noise':
                img = apply_noise(img, val)
                ops_desc_list.append(f"noise(s={val})")
            elif op == 'rotate':
                img = apply_rotation(img, val)
                ops_desc_list.append(f"rotate({val}°)")
            elif op == 'glare':
                img = apply_glare(img)
                ops_desc_list.append("glare")
            elif op == 'contrast':
                img = apply_low_contrast(img)
                ops_desc_list.append("low_contrast")
            elif op == 'perspective':
                img = apply_perspective(img)
                ops_desc_list.append("perspective_warp")

        distortion_str = ", ".join(ops_desc_list) if ops_desc_list else "Clean / None"

        personal = PersonalInfo(full_name=name, date_of_birth=dob, email_address='test@test.com', mobile_number='123', role='Driver')
        license_d = LicenseDetails(license_number=dl, issuing_province=prov, license_expiry_date=exp)

        # Save test sample
        cv2.imwrite(os.path.join(test_img_dir, f'test_{idx+1:03d}.png'), img)

        t0 = time.time()
        res = dl_ocr_engine.verify_license(img, personal, license_d)
        dt = (time.time() - t0) * 1000.0

        ram_mb, cpu_pct = get_system_metrics()
        peak_ram = max(peak_ram, ram_mb)
        peak_cpu = max(peak_cpu, cpu_pct)

        is_pass = res.passed
        if is_pass:
            passed_count += 1
            category_stats[category]["passed"] += 1
            status_str = colored("PASSED", "green", attrs=["bold"])
        else:
            status_str = colored("FAILED", "red", attrs=["bold"])

        avg_score = (res.name_similarity + res.number_similarity) / 2.0 * 100.0
        running_rate = (passed_count / (idx + 1)) * 100.0

        # Detailed Terminal Output for this test case
        print(f"┌─ Case [{idx+1:03d}/102] ── {tc['desc']} ({category})", flush=True)
        print(f"│  • Distortions: {distortion_str}", flush=True)
        print(f"│  • Expected   : Name: {name} | DL: {dl} | DOB: {dob} | EXP: {exp} | Prov: {prov}", flush=True)
        print(f"│  • Extracted  : {res.details}", flush=True)
        print(f"│  • Verdict    : {status_str} | Conf: {avg_score:5.1f}% | Latency: {dt:6.1f}ms | RAM: {ram_mb:5.1f}MB | CPU: {cpu_pct:4.1f}%", flush=True)
        print(f"└─ Running Total: {passed_count}/{idx+1} Passed ({running_rate:5.1f}%)\n", flush=True)

        # Small pause to yield CPU scheduler and prevent throttling
        time.sleep(0.02)

    elapsed = time.time() - t_start
    passing_perc = (passed_count / total_cases) * 100.0

    print("=" * 100, flush=True)
    print("                       102 DRIVING LICENSE TEST SUITE SUMMARY                        ", flush=True)
    print("=" * 100, flush=True)
    print(f"  • Total Test Cases Executed : {total_cases}", flush=True)
    print(f"  • Overall Passed            : {passed_count}/{total_cases} ({passing_perc:.1f}%)", flush=True)
    print(f"  • Total Elapsed Time        : {elapsed:.2f}s (Avg: {elapsed/total_cases*1000:.1f}ms/test)", flush=True)
    print(f"  • Peak Process RAM          : {peak_ram:.1f} MB (Target < 600 MB)", flush=True)
    print(f"  • Peak CPU Utilization      : {peak_cpu:.1f} %", flush=True)
    print("-" * 100, flush=True)
    print("  Category Performance Breakdown:", flush=True)
    for cat, stat in category_stats.items():
        cat_rate = (stat["passed"] / stat["total"]) * 100.0
        bar = "█" * int(cat_rate / 5) + "░" * (20 - int(cat_rate / 5))
        print(f"    - {cat:<20}: {stat['passed']:2d}/{stat['total']:2d} ({cat_rate:5.1f}%) [{bar}]", flush=True)
    print("=" * 100 + "\n", flush=True)


if __name__ == "__main__":
    run_102_dl_tests()
