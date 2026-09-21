"""
Postman Test Helper — Generate Base64 test data for API testing.
Run this script to get base64-encoded images ready for Postman requests.

Usage:
    cd dl_verification_node
    python backend/postman_test_helper.py

Outputs ready-to-paste Base64 strings and sample request bodies.
"""

import os
import sys
import base64
import json

# Ensure backend is in path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_test_images():
    """Locate available test images in the repo."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_images_dir = os.path.join(root, "backend", "test_images")
    plate_dir = os.path.join(root, "backend", "tests", "plate_verification")

    images = {}
    if os.path.exists(test_images_dir):
        for f in os.listdir(test_images_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                images[f] = os.path.join(test_images_dir, f)

    if os.path.exists(plate_dir):
        for f in os.listdir(plate_dir):
            if f.lower().endswith((".webp", ".png", ".jpg")):
                images[f"plate_{f}"] = os.path.join(plate_dir, f)

    return images


def generate_sample_verify_body(selfie_b64: str, dl_b64: str, vehicle_b64: str) -> dict:
    """Generate a complete /api/v1/verify request body."""
    return {
        "request_id": "req_postman_001",
        "driver_id": "driver_postman_001",
        "personal_info": {
            "full_name": "John Smith",
            "email_address": "john.smith@email.com",
            "mobile_number": "+14165551234",
            "date_of_birth": "1990-05-15",
            "gender": "Male",
            "role": "Driver"
        },
        "license_details": {
            "license_number": "DL1234567",
            "issuing_province": "ON",
            "license_expiry_date": "2028-12-31"
        },
        "vehicle_details": {
            "make": "Toyota",
            "model": "Camry",
            "year": 2022,
            "color": "White",
            "plate": "CMBY815"
        },
        "images": {
            "selfie_base64": selfie_b64,
            "license_image_base64": dl_b64,
            "vehicle_photo_base64": vehicle_b64
        }
    }


def main():
    print("=" * 80)
    print("  KOMUTE POSTMAN TEST HELPER — Base64 Image Encoder")
    print("=" * 80)
    print()

    images = find_test_images()

    if not images:
        print("[ERROR] No test images found!")
        print(f"  Searched: {os.path.join(os.getcwd(), 'backend', 'test_images')}")
        sys.exit(1)

    print(f"[OK] Found {len(images)} test images:")
    for name in sorted(images.keys()):
        size_kb = os.path.getsize(images[name]) / 1024
        print(f"  - {name} ({size_kb:.1f} KB)")
    print()

    # Pick one image for each role
    image_list = sorted(images.keys())

    # Use first available for each slot
    selfie_img = images.get(image_list[0], "")
    dl_img = images.get(image_list[1], image_list[0])
    vehicle_img = images.get(image_list[min(2, len(image_list)-1)], image_list[0])

    # Allow user to pick
    print("Available images:")
    for i, name in enumerate(image_list):
        print(f"  [{i}] {name}")
    print()

    try:
        selfie_idx = int(input(f"Select SELFIE image index [0]: ").strip() or "0")
        dl_idx = int(input(f"Select DL IMAGE index [1]: ").strip() or "1")
        vehicle_idx = int(input(f"Select VEHICLE IMAGE index [2]: ").strip() or "2")
    except (ValueError, EOFError):
        selfie_idx, dl_idx, vehicle_idx = 0, 1, 2

    selfie_path = images[image_list[selfie_idx]]
    dl_path = images[image_list[dl_idx]]
    vehicle_path = images[image_list[vehicle_idx]]

    print()
    print(f"Encoding SELFIE:  {image_list[selfie_idx]}...")
    selfie_b64 = encode_image_to_base64(selfie_path)
    print(f"  → {len(selfie_b64)} characters")

    print(f"Encoding DL:      {image_list[dl_idx]}...")
    dl_b64 = encode_image_to_base64(dl_path)
    print(f"  → {len(dl_b64)} characters")

    print(f"Encoding VEHICLE: {image_list[vehicle_idx]}...")
    vehicle_b64 = encode_image_to_base64(vehicle_path)
    print(f"  → {len(vehicle_b64)} characters")

    print()
    print("=" * 80)
    print("  COPY-PASTE BASE64 STRINGS FOR POSTMAN")
    print("=" * 80)

    print()
    print("--- SELFIE BASE64 (for selfie_base64 field) ---")
    print(selfie_b64)
    print()
    print("--- DL IMAGE BASE64 (for license_image_base64 field) ---")
    print(dl_b64)
    print()
    print("--- VEHICLE IMAGE BASE64 (for vehicle_photo_base64 field) ---")
    print(vehicle_b64)

    # Generate complete request body
    body = generate_sample_verify_body(selfie_b64, dl_b64, vehicle_b64)

    print()
    print("=" * 80)
    print("  COMPLETE /api/v1/verify REQUEST BODY (ready to paste in Postman)")
    print("=" * 80)
    print()
    print(json.dumps(body, indent=2))

    # Also save to file for convenience
    output_path = os.path.join(os.path.dirname(__file__), "postman_sample_request.json")
    with open(output_path, "w") as f:
        json.dump(body, f, indent=2)
    print()
    print(f"[SAVED] Complete request body saved to: {output_path}")

    # Generate privacy crop preview body
    print()
    print("=" * 80)
    print("  PRIVACY CROP PREVIEW REQUEST BODY")
    print("=" * 80)
    preview_body = {
        "image_base64": dl_b64,
        "margin_ratio": 0.15
    }
    print(json.dumps(preview_body, indent=2))

    preview_path = os.path.join(os.path.dirname(__file__), "postman_privacy_preview_request.json")
    with open(preview_path, "w") as f:
        json.dump(preview_body, f, indent=2)
    print(f"[SAVED] Privacy preview request body saved to: {preview_path}")

    print()
    print("=" * 80)
    print("  POSTMAN QUICK START")
    print("=" * 80)
    print("""
1. Start the server:
   cd dl_verification_node/backend
   python -m app.main

2. Import Postman Collection:
   File → Import → Upload Files →
   Select: Komute_Driver_Verification_v2.postman_collection.json

3. Import Postman Environment:
   File → Import → Upload Files →
   Select: Komute_Local.postman_environment.json

4. Select environment "Komüte Local" in top-right dropdown

5. Update image base64 values in requests:
   Replace "REPLACE_WITH_BASE64_SELFIE" with the selfie base64 above
   Replace "REPLACE_WITH_BASE64_DL_IMAGE" with the DL base64 above
   Replace "REPLACE_WITH_BASE64_VEHICLE_IMAGE" with the vehicle base64 above

6. Run requests in order:
   a) GET /health (no auth needed)
   b) GET /ping (no auth needed)
   c) POST /api/v1/verify (with images)
   d) POST /api/v1/privacy-crop-preview
   e) GET /api/v1/telemetry/sessions
   f) POST /api/v1/telemetry/self-test

7. Check Tests tab for auto-passing assertions
""")


if __name__ == "__main__":
    main()
