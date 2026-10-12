"""
Generate 102 Brutal Driving License Test Images for All 13 Canadian Jurisdictions.
(13 provinces/territories x 102 images = 1,326 test images).
Includes real-time progress reporting, speed, elapsed time, and ETA calculations.
"""

import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = SCRIPT_DIR
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.provincial_templates import PROVINCIAL_CONFIGS
from core.canadian_dl_grammar import format_canadian_dl

TEST_IMAGES_DIR = os.path.join(SCRIPT_DIR, "test_images")


def apply_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def apply_noise(img: np.ndarray, severity: int) -> np.ndarray:
    noise = np.random.normal(0, severity, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_rotation(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(240, 240, 240))


def apply_glare(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    center = (int(w * 0.6), int(h * 0.5))
    radius = int(min(w, h) * 0.28)
    cv2.circle(overlay, center, radius, (255, 255, 255), -1)
    return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)


def apply_low_contrast(img: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=0.5, beta=128)


def apply_perspective(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[w * 0.07, h * 0.1], [w * 0.93, h * 0.05], [w * 0.035, h * 0.94], [w * 0.965, h * 0.88]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(240, 240, 240))


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


def generate_base_card(prov: str, profile_idx: int = 0):
    cfg = PROVINCIAL_CONFIGS[prov]
    template_path = cfg["file"]
    img = cv2.imread(template_path)
    if img is None:
        raise FileNotFoundError(f"Template image not found: {template_path}")

    h, w = img.shape[:2]
    target_w = cfg["target_w"]
    target_h = int(h * (target_w / float(w)))
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Inpaint field erase boxes (Zero face overlap guaranteed)
    for box in cfg["erase"]:
        clean_field_patch(resized, box, cfg["bg"])

    img_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font_large = ImageFont.truetype("arialbd.ttf", 20)
        font_med = ImageFont.truetype("arialbd.ttf", 18)
        font_small = ImageFont.truetype("arialbd.ttf", 19)
    except Exception:
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_small = ImageFont.load_default()

    profiles = cfg["profiles"]
    profile = profiles[profile_idx % len(profiles)]
    name, dl, dob, exp = profile

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
    return final_bgr, profile


def build_102_distortion_specs():
    suite = []
    # 1-10: Clean Base
    for i in range(10):
        suite.append({"category": "Clean Base", "desc": f"Clean Baseline #{i+1}", "profile_idx": i, "ops": []})
    # 11-30: Blur (Mild to Heavy)
    for i in range(20):
        k = (i // 2) + 1
        suite.append({"category": "Gaussian Blur", "desc": f"Blur Level {k}", "profile_idx": i, "ops": [("blur", k * 2)]})
    # 31-50: Noise (Severity 5 to 100)
    for i in range(20):
        sev = (i + 1) * 5
        suite.append({"category": "Sensor Noise", "desc": f"Gaussian Noise {sev}", "profile_idx": i, "ops": [("noise", sev)]})
    # 51-70: Rotation (-10 to +15 degrees)
    for i in range(20):
        ang = (-10 + i) if i != 10 else 15
        suite.append({"category": "Skew Rotation", "desc": f"Rotation {ang}°", "profile_idx": i, "ops": [("rotate", ang)]})
    # 71-80: Glare (5) & Low Contrast (5)
    for i in range(10):
        if i < 5:
            suite.append({"category": "Optical Glare", "desc": f"Optical Glare #{i+1}", "profile_idx": i, "ops": [("glare", None)]})
        else:
            suite.append({"category": "Low Contrast", "desc": f"Low Contrast #{i-4}", "profile_idx": i, "ops": [("contrast", None)]})
    # 81-90: Perspective Homography Warp
    for i in range(10):
        suite.append({"category": "Perspective Warp", "desc": f"Perspective Skew #{i+1}", "profile_idx": i, "ops": [("perspective", None)]})
    # 91-102: Brutal Combination
    for i in range(12):
        suite.append({"category": "Brutal Combo", "desc": f"Brutal Combo #{i+1}", "profile_idx": i, "ops": [("blur", 3), ("noise", 20), ("rotate", -5), ("perspective", None)]})
    return suite


def generate_all_provincial_images(target_regions=None, force=False):
    import json
    import tempfile
    import shutil

    print("=" * 100, flush=True)
    print("      FAST 102 BRUTAL TEST IMAGE GENERATION (13 JURISDICTIONS - 1,326 TOTAL IMAGES)      ", flush=True)
    print("=" * 100 + "\n", flush=True)

    specs = build_102_distortion_specs()
    all_provs = sorted(PROVINCIAL_CONFIGS.keys())
    provinces = [p for p in all_provs if p in target_regions] if target_regions else all_provs
    total_to_generate = len(provinces) * len(specs)
    total_generated = 0
    all_ground_truth = []
    t_start = time.time()

    # Fast PNG compression parameters
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    # Fast local SSD buffer directory (avoids USB FAT32 sync locking)
    temp_dir = os.path.join(tempfile.gettempdir(), "komute_prov_gen")
    os.makedirs(temp_dir, exist_ok=True)

    progress_file = os.path.join(BACKEND_DIR, "generation_progress.json")

    for p_num, prov in enumerate(provinces):
        cfg = PROVINCIAL_CONFIGS[prov]
        prov_dir = os.path.join(TEST_IMAGES_DIR, prov)
        os.makedirs(prov_dir, exist_ok=True)

        num_profiles = len(cfg["profiles"])
        base_cards = [generate_base_card(prov, profile_idx=p_idx)[0] for p_idx in range(num_profiles)]

        prov_ground_truth = []

        for idx, spec in enumerate(specs):
            profile = cfg["profiles"][spec["profile_idx"] % num_profiles]
            name, dl, dob, exp = profile
            parts = name.split()
            last_name = parts[-1].upper() if len(parts) > 1 else name.upper()
            first_name = parts[0].upper()
            formatted_dl = format_canadian_dl(dl, prov)

            out_path = os.path.join(prov_dir, f"test_{idx+1:03d}.png")
            temp_path = os.path.join(temp_dir, f"{prov}_test_{idx+1:03d}.png")

            # Check if file already exists and is valid
            if force or not (os.path.exists(out_path) and os.path.getsize(out_path) > 1000):
                base_img = base_cards[spec["profile_idx"] % num_profiles]
                img = base_img.copy()

                for op, val in spec["ops"]:
                    if op == "blur":
                        img = apply_blur(img, val)
                    elif op == "noise":
                        img = apply_noise(img, val)
                    elif op == "rotate":
                        img = apply_rotation(img, val)
                    elif op == "glare":
                        img = apply_glare(img)
                    elif op == "contrast":
                        img = apply_low_contrast(img)
                    elif op == "perspective":
                        img = apply_perspective(img)

                # Write to high-speed NVMe SSD buffer first
                cv2.imwrite(temp_path, img, png_params)
                # Stream copy directly to destination
                shutil.copyfile(temp_path, out_path)

                if prov == "ON":
                    shutil.copyfile(temp_path, os.path.join(TEST_IMAGES_DIR, f"test_{idx+1:03d}.png"))

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            gt_entry = {
                "test_idx": idx + 1,
                "filename": f"test_{idx+1:03d}.png",
                "category": spec["category"],
                "description": spec["desc"],
                "province": prov,
                "province_name": cfg["name"],
                "expected": {
                    "name": name,
                    "first_name": first_name,
                    "last_name": last_name,
                    "dl_number": dl,
                    "formatted_dl": formatted_dl,
                    "dob": dob,
                    "expiry": exp,
                }
            }
            prov_ground_truth.append(gt_entry)
            all_ground_truth.append(gt_entry)

            total_generated += 1

            # Real-time progress update every 10 images or at end of province
            if (idx + 1) % 10 == 0 or (idx + 1) == len(specs):
                elapsed = time.time() - t_start
                rate = total_generated / elapsed if elapsed > 0 else 0
                remaining = total_to_generate - total_generated
                eta = remaining / rate if rate > 0 else 0
                pct = (total_generated / total_to_generate) * 100.0

                prog_bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(
                    f"[{prog_bar}] {total_generated:4d}/{total_to_generate} ({pct:5.1f}%) | "
                    f"Region: {prov} [{idx+1:3d}/102] | "
                    f"Speed: {rate:4.1f} img/s | "
                    f"Elapsed: {elapsed:4.1f}s | "
                    f"Remaining: {remaining:4d} (ETA: {eta:4.1f}s)",
                    flush=True
                )

                # Update progress JSON
                progress_info = {
                    "total_generated": total_generated,
                    "total_to_generate": total_to_generate,
                    "pct": round(pct, 1),
                    "current_province": prov,
                    "current_province_name": cfg["name"],
                    "province_count": f"{idx+1}/102",
                    "speed_fps": round(rate, 2),
                    "elapsed_seconds": round(elapsed, 1),
                    "remaining_seconds": round(eta, 1),
                    "status": "RUNNING" if total_generated < total_to_generate else "COMPLETED"
                }
                try:
                    with open(progress_file, "w", encoding="utf-8") as pf:
                        json.dump(progress_info, pf, indent=2)
                except Exception:
                    pass

        # Save provincial ground truth JSON
        with open(os.path.join(prov_dir, "ground_truth.json"), "w", encoding="utf-8") as f:
            json.dump(prov_ground_truth, f, indent=2)

        print(f"  [OK] Completed Region {prov} ({cfg['name']}): 102/102 images ready + ground_truth.json saved.\n", flush=True)

    # Save consolidated ground truth JSON
    with open(os.path.join(TEST_IMAGES_DIR, "ground_truth_all.json"), "w", encoding="utf-8") as f:
        json.dump(all_ground_truth, f, indent=2)

    total_time = time.time() - t_start
    final_info = {
        "total_generated": total_generated,
        "total_to_generate": total_to_generate,
        "pct": 100.0,
        "status": "COMPLETED",
        "total_time_seconds": round(total_time, 2),
        "speed_fps": round(total_generated / total_time, 2) if total_time > 0 else 0
    }
    with open(progress_file, "w", encoding="utf-8") as pf:
        json.dump(final_info, pf, indent=2)

    print("=" * 100, flush=True)
    print(f"SUCCESS: Generated all {total_generated} brutal test images in {total_time:.2f}s ({total_generated/total_time:.1f} img/s)!", flush=True)
    print("=" * 100 + "\n", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Brutal Provincial Test Image Generator")
    parser.add_argument("--regions", type=str, default=None, help="Comma-separated provinces to generate (e.g. 'ON,MB')")
    parser.add_argument("--force", action="store_true", help="Force regenerate existing images")
    args = parser.parse_args()
    regs = [r.strip().upper() for r in args.regions.split(",")] if args.regions else None
    generate_all_provincial_images(target_regions=regs, force=args.force)
