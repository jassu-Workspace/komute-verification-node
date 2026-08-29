import os
import cv2
import re
import time
import numpy as np
import sys
import psutil

# Ensure backend root is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.ocr_engine import shared_ocr
from core.vehicle_alpr import vehicle_alpr_engine, calculate_plate_match_score, normalize_plate_string
from core.image_utils import apply_clahe, upscale_and_sharpen
from rapidfuzz import fuzz

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, on_color=None, attrs=None):
        return text

process = psutil.Process(os.getpid())


def get_system_metrics():
    try:
        ram = process.memory_info().rss / (1024 * 1024)
        cpu = psutil.cpu_percent(interval=None)
        return ram, cpu
    except Exception:
        return 0.0, 0.0


def extract_plate_text(img_path: str, save_crop_dir: str = None) -> str:
    """
    Extracts license plate text from vehicle image using RapidOCR (ONNX Runtime).
    """
    engine = shared_ocr.get_engine()
    img = cv2.imread(img_path)
    if img is None:
        return ""

    res, _ = engine(img)
    best_crop = None

    # If full vehicle image didn't detect plate directly, test candidate crops
    if not res:
        candidates = vehicle_alpr_engine.detect_plate_candidate_regions(img)
        for crop, bbox in candidates[:4]:
            if crop.size == 0:
                continue
            enhanced = apply_clahe(crop)
            c_res, _ = engine(enhanced)
            if c_res:
                res = c_res
                best_crop = crop
                break

    if not res:
        return ""

    candidates = []
    for line in res:
        if len(line) >= 2:
            box = line[0]
            text = str(line[1])
            conf = float(line[2]) if len(line) >= 3 else 1.0

            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
            if cleaned in ("ALAMY", "WWW", "STOCK", "GETTY", "SHUTTERSTOCK", "PHOTO"):
                continue

            if 3 <= len(cleaned) <= 10:
                candidates.append((cleaned, conf, box))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: (len(x[0]) >= 5, x[1]), reverse=True)
    best_cleaned, best_conf, best_box = candidates[0]

    # Save cropped plate image if requested
    if save_crop_dir:
        os.makedirs(save_crop_dir, exist_ok=True)
        filename = os.path.basename(img_path)
        save_path = os.path.join(save_crop_dir, f"cropped_{filename}")
        if best_crop is not None and best_crop.size > 0:
            cv2.imwrite(save_path, best_crop)
        elif best_box is not None and len(best_box) == 4:
            try:
                pts = np.array(best_box, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                pad = 10
                img_h, img_w = img.shape[:2]
                x_start, y_start = max(0, x - pad), max(0, y - pad)
                x_end, y_end = min(img_w, x + w + pad), min(img_h, y + h + pad)
                cropped_plate = img[y_start:y_end, x_start:x_end]
                if cropped_plate.size > 0:
                    cv2.imwrite(save_path, cropped_plate)
            except Exception:
                pass

    return best_cleaned


def run_tests():
    print("\n" + "=" * 100, flush=True)
    print("      VEHICLE LICENSE PLATE VERIFICATION (Fine-Tuned RapidOCR ONNX Engine)      ", flush=True)
    print("=" * 100 + "\n", flush=True)

    test_dir = os.path.dirname(__file__)
    test_cases = {
        "1.webp": "CMBY815",
        "2.webp": "627BEJ",
        "3.webp": "743FYK",
        "4.webp": "CDA3946",
    }

    passed = 0
    total = 0

    t_start = time.time()

    for idx, (filename, expected) in enumerate(test_cases.items()):
        filepath = os.path.join(test_dir, filename)
        if not os.path.exists(filepath):
            continue

        total += 1
        t0 = time.time()
        extracted = extract_plate_text(filepath)
        dt = (time.time() - t0) * 1000.0

        ram_mb, cpu_pct = get_system_metrics()

        matched, match_score = calculate_plate_match_score(extracted, expected)
        score_pct = match_score * 100.0
        is_pass = matched or score_pct >= 85.0

        if is_pass:
            passed += 1
            status_str = colored("PASSED", "green", attrs=["bold"])
        else:
            status_str = colored("FAILED", "red", attrs=["bold"])

        print(f"┌─ Image [{total}/{len(test_cases)}] ── {filename}", flush=True)
        print(f"│  • Expected Plate : {expected}", flush=True)
        print(f"│  • Extracted Plate: {extracted}", flush=True)
        print(f"│  • Match Score    : {score_pct:5.1f}%", flush=True)
        print(f"│  • Diagnostics    : Verdict: {status_str} | Latency: {dt:6.1f}ms | RAM Used: {ram_mb:5.1f}MB | CPU: {cpu_pct:4.1f}%", flush=True)
        print(f"└─ Running Total    : {passed}/{total} Passed ({(passed/total)*100:.1f}%)\n", flush=True)

    elapsed = time.time() - t_start
    passing_rate = (passed / total * 100.0) if total > 0 else 0.0

    print("=" * 100, flush=True)
    print(f" FINAL RESULT: {passed}/{total} Passed ({passing_rate:.1f}%) | Total Time: {elapsed:.2f}s", flush=True)
    print("=" * 100 + "\n", flush=True)


if __name__ == "__main__":
    run_tests()
