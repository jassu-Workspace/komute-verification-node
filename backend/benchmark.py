import os
import psutil
import time
import asyncio
import cv2
import numpy as np

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print(f"Memory at startup: {get_memory_mb():.2f} MB")

# Now load libraries
print("Loading libraries...")
from core.dl_ocr import dl_ocr_engine
from core.face_privacy_cropper import face_privacy_cropper
from core.vehicle_alpr import vehicle_alpr_engine
from app.schemas import PersonalInfo, LicenseDetails, VehicleDetails

print(f"Memory after loading libraries & models: {get_memory_mb():.2f} MB")

def create_dummy_image(w=800, h=600):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return img

async def run_benchmark():
    # Load actual image if possible
    img_path = "test_upload.png"
    if os.path.exists(img_path):
        test_img = cv2.imread(img_path)
    else:
        test_img = create_dummy_image()

    print(f"Memory after loading test image: {get_memory_mb():.2f} MB")

    personal_info = PersonalInfo(
        full_name="Jaswanth Sri Sai Venkat Dangeti",
        email_address="test@test.com",
        role="Driver",
        mobile_number="123",
        date_of_birth="2005-12-14",
        gender="Male"
    )
    license_details = LicenseDetails(
        license_number="DDNPV1331Q",
        issuing_province="ON",
        license_expiry_date="2030-12-14"
    )
    vehicle_details = VehicleDetails(
        make="Hyundai",
        model="Venue",
        year="2020",
        color="Black",
        plate="DL 7CQ 1939"
    )

    # 1. OCR
    t0 = time.time()
    res1 = dl_ocr_engine.verify_license(test_img, personal_info, license_details)
    t1 = time.time()
    print(f"OCR Latency: {(t1 - t0)*1000:.2f} ms")
    print(f"Memory after OCR: {get_memory_mb():.2f} MB")

    # 2. Face
    t0 = time.time()
    res2 = face_privacy_cropper.extract_isolated_face(test_img)
    t1 = time.time()
    print(f"Face Cropper Latency: {(t1 - t0)*1000:.2f} ms")
    print(f"Memory after Face Cropper: {get_memory_mb():.2f} MB")

    # 3. ALPR
    t0 = time.time()
    res3 = vehicle_alpr_engine.verify_vehicle(test_img, vehicle_details)
    t1 = time.time()
    print(f"ALPR Latency: {(t1 - t0)*1000:.2f} ms")
    print(f"Memory after ALPR: {get_memory_mb():.2f} MB")
    
    print(f"PEAK MEMORY (current): {get_memory_mb():.2f} MB")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
