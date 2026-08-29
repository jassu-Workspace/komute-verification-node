import sys
import os
import cv2
import time

# Thread clamping
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["ORT_INTRA_OP_NUM_THREADS"] = "2"
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"

sys.path.append(os.path.abspath('.'))

import psutil
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
    try:
        ram = process.memory_info().rss / (1024 * 1024)
        cpu = psutil.cpu_percent(interval=None)
        return ram, cpu
    except Exception:
        return 0.0, 0.0


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


def run_batch_verification():
    folder = 'test_images'
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' not found.", flush=True)
        return

    images = sorted([f for f in os.listdir(folder) if f.endswith('.png')])

    if not images:
        print("No images found in the folder.", flush=True)
        return

    test_suite = []
    for i in range(10): test_suite.append(base_data[i % len(base_data)])
    for i in range(20): test_suite.append(base_data[i % len(base_data)])
    for i in range(20): test_suite.append(base_data[i % len(base_data)])
    for i in range(10): test_suite.append(base_data[i % len(base_data)])
    for i in range(10): test_suite.append(base_data[i % len(base_data)])
    for i in range(10): test_suite.append(base_data[i % len(base_data)])
    for i in range(12): test_suite.append(base_data[i % len(base_data)])

    print("\n" + "=" * 100, flush=True)
    print(f"      BATCH DRIVING LICENSE OCR VERIFICATION ({len(images)} Images - RapidOCR ONNX)      ", flush=True)
    print("=" * 100 + "\n", flush=True)

    passed_count = 0
    t_start = time.time()

    for idx, img_name in enumerate(images):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        try:
            case_idx = int(img_name.replace('test_', '').replace('.jpg', '').replace('.png', '')) - 1
            if case_idx >= len(test_suite):
                continue
            name, dl, dob, exp, prov = test_suite[case_idx]
        except Exception:
            continue

        personal = PersonalInfo(full_name=name, date_of_birth=dob, email_address='test@test.com', mobile_number='123', role='Driver')
        license_d = LicenseDetails(license_number=dl, issuing_province=prov, license_expiry_date=exp)

        t0 = time.time()
        res = dl_ocr_engine.verify_license(img, personal, license_d)
        dt = (time.time() - t0) * 1000.0

        ram_mb, cpu_pct = get_system_metrics()

        avg_score = (res.name_similarity + res.number_similarity) / 2.0 * 100.0
        if res.passed:
            passed_count += 1
            status_str = colored("PASSED", "green", attrs=["bold"])
        else:
            status_str = colored("FAILED", "red", attrs=["bold"])

        running_rate = (passed_count / (idx + 1)) * 100.0

        print(f"┌─ Image [{idx+1:03d}/{len(images)}] ── {img_name}", flush=True)
        print(f"│  • Expected   : Name: {name} | DL: {dl} | DOB: {dob} | EXP: {exp} | Prov: {prov}", flush=True)
        print(f"│  • Extracted  : {res.details}", flush=True)
        print(f"│  • Diagnostics: Verdict: {status_str} | Conf: {avg_score:5.1f}% | Latency: {dt:6.1f}ms | RAM Used: {ram_mb:5.1f}MB | CPU: {cpu_pct:4.1f}%", flush=True)
        print(f"└─ Running Total: {passed_count}/{idx+1} Passed ({running_rate:5.1f}%)\n", flush=True)

        time.sleep(0.02)

    elapsed = time.time() - t_start
    passing_perc = (passed_count / len(images)) * 100.0

    print("=" * 100, flush=True)
    print(f" FINAL RESULT: {passed_count}/{len(images)} Passed ({passing_perc:.1f}%) | Total Time: {elapsed:.2f}s", flush=True)
    print("=" * 100 + "\n", flush=True)


if __name__ == "__main__":
    run_batch_verification()
