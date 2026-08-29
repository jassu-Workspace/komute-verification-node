import os
import sys
import time

# Thread clamping to prevent 98% CPU spike
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["ORT_INTRA_OP_NUM_THREADS"] = "2"
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import cv2
import psutil
from rapidfuzz import fuzz

cv2.setNumThreads(2)

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, on_color=None, attrs=None):
        return text

from tests.plate_verification.run_plate_tests import extract_plate_text
from core.vehicle_alpr import calculate_plate_match_score, normalize_plate_string

process = psutil.Process(os.getpid())


def get_system_metrics():
    try:
        ram = process.memory_info().rss / (1024 * 1024)
        cpu = psutil.cpu_percent(interval=None)
        return ram, cpu
    except Exception:
        return 0.0, 0.0


BASE_EXPECTED = [
    "CMBY815",
    "627BEJ",
    "743FYK",
    "CDA3946",
]

EFFECT_NAMES = [
    "Clean / Standard",
    "Mild Gaussian Blur",
    "Heavy Gaussian Blur",
    "Mild Sensor Noise",
    "Heavy Sensor Noise",
    "Optical Specular Glare",
    "Low Contrast / Shadows",
    "Perspective Skew Angle",
    "Blur + Noise Distortion",
    "Perspective + Glare Warp",
]


def run_102_tests():
    print("\n" + "=" * 100, flush=True)
    print("      VEHICLE LICENSE PLATE BRUTAL TEST SUITE (102 CASES - Fine-Tuned RapidOCR ONNX)      ", flush=True)
    print("=" * 100 + "\n", flush=True)

    test_dir = os.path.join(SCRIPT_DIR, "plate_test_images")
    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) < 102:
        print(f"Generating 102 brutal test images in {test_dir}...", flush=True)
        from tests.plate_verification.generate_brutal_plate_tests import generate_102_tests
        generate_102_tests()

    passed = 0
    total = 0
    peak_ram = 0.0
    peak_cpu = 0.0

    category_stats = {}

    t_start = time.time()

    for i in range(1, 103):
        filename = f"test_{i:03d}.webp"
        filepath = os.path.join(test_dir, filename)

        if not os.path.exists(filepath):
            continue

        distortion = EFFECT_NAMES[(i - 1) % len(EFFECT_NAMES)]
        if distortion not in category_stats:
            category_stats[distortion] = {"total": 0, "passed": 0}
        category_stats[distortion]["total"] += 1

        expected = BASE_EXPECTED[(i - 1) % 4]
        total += 1

        t0 = time.time()
        extracted = extract_plate_text(filepath, save_crop_dir=os.path.join(SCRIPT_DIR, "cropped_dataset"))
        dt = (time.time() - t0) * 1000.0

        ram_mb, cpu_pct = get_system_metrics()
        peak_ram = max(peak_ram, ram_mb)
        peak_cpu = max(peak_cpu, cpu_pct)

        # Evaluate match using ALPR character confusion scoring and Levenshtein distance
        matched, match_score = calculate_plate_match_score(extracted, expected)
        score_pct = match_score * 100.0

        is_pass = matched or (score_pct >= 85.0) or (normalize_plate_string(extracted, apply_confusion=True) == normalize_plate_string(expected, apply_confusion=True))

        if is_pass:
            passed += 1
            category_stats[distortion]["passed"] += 1
            status_str = colored("PASSED", "green", attrs=["bold"])
        else:
            status_str = colored("FAILED", "red", attrs=["bold"])

        running_rate = (passed / total) * 100.0

        # Detailed Terminal Output for this test case
        print(f"┌─ Case [{i:03d}/102] ── {filename} ({distortion})", flush=True)
        print(f"│  • Expected Plate : {expected}", flush=True)
        print(f"│  • Extracted Plate: {extracted or '(none)'}", flush=True)
        print(f"│  • Verdict        : {status_str} | Score: {score_pct:5.1f}% | Latency: {dt:6.1f}ms | RAM: {ram_mb:5.1f}MB | CPU: {cpu_pct:4.1f}%", flush=True)
        print(f"└─ Running Total    : {passed}/{total} Passed ({running_rate:5.1f}%)\n", flush=True)

        # Small pause to yield CPU scheduler
        time.sleep(0.02)

    elapsed = time.time() - t_start
    passing_rate = (passed / total * 100.0) if total > 0 else 0.0

    print("=" * 100, flush=True)
    print("                    102 VEHICLE LICENSE PLATE TEST SUITE SUMMARY                     ", flush=True)
    print("=" * 100, flush=True)
    print(f"  • Total Test Cases Executed : {total}", flush=True)
    print(f"  • Overall Passed            : {passed}/{total} ({passing_rate:.1f}%)", flush=True)
    print(f"  • Total Elapsed Time        : {elapsed:.2f}s (Avg: {elapsed/total*1000:.1f}ms/test)", flush=True)
    print(f"  • Peak Process RAM          : {peak_ram:.1f} MB (Target < 600 MB)", flush=True)
    print(f"  • Peak CPU Utilization      : {peak_cpu:.1f} %", flush=True)
    print("-" * 100, flush=True)
    print("  Category Performance Breakdown:", flush=True)
    for cat, stat in category_stats.items():
        cat_rate = (stat["passed"] / stat["total"]) * 100.0
        bar = "█" * int(cat_rate / 5) + "░" * (20 - int(cat_rate / 5))
        print(f"    - {cat:<26}: {stat['passed']:2d}/{stat['total']:2d} ({cat_rate:5.1f}%) [{bar}]", flush=True)
    print("=" * 100 + "\n", flush=True)


if __name__ == "__main__":
    run_102_tests()
