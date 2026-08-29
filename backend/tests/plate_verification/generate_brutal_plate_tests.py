import os
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plate_test_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base images and their true plate numbers
BASE_IMAGES = {
    0: ("1.webp", "CMBY815"),
    1: ("2.webp", "627BEJ"),
    2: ("3.webp", "743FYK"),
    3: ("4.webp", "CDA3946"),
}


def apply_blur(img, severity=5):
    if severity % 2 == 0:
        severity += 1
    return cv2.GaussianBlur(img, (severity, severity), 0)


def apply_noise(img, severity=0.05):
    noisy = np.copy(img)
    num_pepper = np.ceil(severity * img.size * 0.5)
    num_salt = np.ceil(severity * img.size * 0.5)

    # Add salt
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy[tuple(coords)] = 255
    # Add pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy[tuple(coords)] = 0
    return noisy


def apply_glare(img):
    overlay = img.copy()
    h, w = img.shape[:2]
    cv2.circle(overlay, (w // 2, h // 2), int(h * 0.4), (255, 255, 255), -1)
    return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)


def apply_low_contrast(img):
    return cv2.convertScaleAbs(img, alpha=0.5, beta=128)


def apply_perspective(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[0, int(h * 0.2)], [w, 0], [0, int(h * 0.8)], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(128, 128, 128))


effects = [
    ("clean", lambda x: x),
    ("mild_blur", lambda x: apply_blur(x, 5)),
    ("heavy_blur", lambda x: apply_blur(x, 15)),
    ("mild_noise", lambda x: apply_noise(x, 0.02)),
    ("heavy_noise", lambda x: apply_noise(x, 0.10)),
    ("glare", apply_glare),
    ("low_contrast", apply_low_contrast),
    ("perspective", apply_perspective),
    ("blur_and_noise", lambda x: apply_noise(apply_blur(x, 7), 0.03)),
    ("perspective_and_glare", lambda x: apply_glare(apply_perspective(x))),
]


def generate_102_tests():
    print(f"Generating 102 brutal test cases in {OUTPUT_DIR}...", flush=True)
    count = 1

    for i in range(102):
        base_idx = i % 4
        base_filename, expected_text = BASE_IMAGES[base_idx]
        base_path = os.path.join(SCRIPT_DIR, base_filename)

        img = cv2.imread(base_path)
        if img is None:
            print(f"Error loading {base_path}", flush=True)
            continue

        effect_name, effect_fn = effects[i % len(effects)]
        processed = effect_fn(img)

        out_name = f"test_{count:03d}.webp"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), processed)
        count += 1

    print(f"Done! Successfully generated {count-1} test images.", flush=True)


if __name__ == "__main__":
    generate_102_tests()
