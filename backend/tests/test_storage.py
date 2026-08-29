import os
import shutil
import tempfile
import numpy as np
import pytest
from storage.storage import StorageManager, sanitize_folder_name


def test_sanitize_folder_name():
    assert sanitize_folder_name("driver/..\\123") == "driver_.._123"
    assert sanitize_folder_name("drv_test_01") == "drv_test_01"
    assert sanitize_folder_name("") == "unknown"


def test_storage_manager_save_verification_session():
    temp_dir = tempfile.mkdtemp()
    try:
        manager = StorageManager(base_dir=temp_dir)

        # Create dummy images
        selfie_img = np.zeros((100, 100, 3), dtype=np.uint8)
        license_img = np.ones((200, 300, 3), dtype=np.uint8) * 128
        vehicle_img = np.ones((150, 250, 3), dtype=np.uint8) * 200

        selfie_comp = np.zeros((80, 80, 3), dtype=np.uint8)
        license_comp = np.ones((160, 240, 3), dtype=np.uint8) * 128
        vehicle_comp = np.ones((120, 200, 3), dtype=np.uint8) * 200

        selfie_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        dl_crop = np.ones((60, 60, 3), dtype=np.uint8) * 100
        plate_crop = np.ones((30, 90, 3), dtype=np.uint8) * 255

        saved = manager.save_verification_session_images(
            driver_id="drv_jaswanth_01",
            request_id="req_test_1001",
            original_images={
                "selfie": selfie_img,
                "license": license_img,
                "vehicle": vehicle_img,
            },
            compressed_images={
                "selfie": selfie_comp,
                "license": license_comp,
                "vehicle": vehicle_comp,
            },
            cropped_images={
                "selfie_face": selfie_crop,
                "dl_face": dl_crop,
                "vehicle_plate": plate_crop,
            },
            metadata={"test_key": "test_val"},
        )

        assert "session_directory" in saved
        assert os.path.exists(saved["session_directory"])
        assert os.path.exists(saved["originals_folder"])
        assert os.path.exists(saved["compressed_folder"])
        assert os.path.exists(saved["cropped_folder"])

        # Check saved original files
        assert "selfie" in saved["original_files"]
        assert "license" in saved["original_files"]
        assert "vehicle" in saved["original_files"]
        assert os.path.exists(saved["original_files"]["selfie"]["absolute_path"])
        assert os.path.exists(saved["original_files"]["license"]["absolute_path"])
        assert os.path.exists(saved["original_files"]["vehicle"]["absolute_path"])

        # Check saved compressed files
        assert "selfie" in saved["compressed_files"]
        assert "license" in saved["compressed_files"]
        assert "vehicle" in saved["compressed_files"]
        assert os.path.exists(saved["compressed_files"]["selfie"]["absolute_path"])
        assert os.path.exists(saved["compressed_files"]["license"]["absolute_path"])
        assert os.path.exists(saved["compressed_files"]["vehicle"]["absolute_path"])

        # Check saved cropped files
        assert "selfie_face" in saved["cropped_files"]
        assert "dl_face" in saved["cropped_files"]
        assert "vehicle_plate" in saved["cropped_files"]
        assert os.path.exists(saved["cropped_files"]["selfie_face"]["absolute_path"])
        assert os.path.exists(saved["cropped_files"]["dl_face"]["absolute_path"])
        assert os.path.exists(saved["cropped_files"]["vehicle_plate"]["absolute_path"])

        # Check metadata JSON file
        assert "metadata_file" in saved
        assert os.path.exists(saved["metadata_file"])

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_storage_manager_preview_images():
    temp_dir = tempfile.mkdtemp()
    try:
        manager = StorageManager(base_dir=temp_dir)
        orig_img = np.ones((100, 100, 3), dtype=np.uint8) * 50
        comp_img = np.ones((80, 80, 3), dtype=np.uint8) * 50
        crop_img = np.ones((40, 40, 3), dtype=np.uint8) * 150

        saved = manager.save_preview_images(
            preview_id="prev_test_99",
            original_img=orig_img,
            cropped_img=crop_img,
            compressed_img=comp_img,
        )

        assert saved["original_file"] is not None
        assert saved["compressed_file"] is not None
        assert saved["cropped_file"] is not None
        assert os.path.exists(os.path.join(temp_dir, "previews", "prev_test_99", "originals", "license_original.jpg"))
        assert os.path.exists(os.path.join(temp_dir, "previews", "prev_test_99", "compressed", "license_compressed.webp"))
        assert os.path.exists(os.path.join(temp_dir, "previews", "prev_test_99", "cropped", "dl_face_crop_sanitized.webp"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
