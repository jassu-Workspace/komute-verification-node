# Context
Identity and vehicle verification system. Addresses high RAM spikes and face detection accuracy.

# Approach
1. **Memory Management**: Pre-load models (RetinaFace, OCR) at startup, NOT on-request. Use `multiprocessing` for heavy model inference if RAM spike persists (isolate model memory).
2. **Face Verification**:
   - Use `RetinaFace` (MobileNet0.25 backbone) for robust detection.
   - **Verification Layer**: Check face detection confidence + landmarks in cropped image.
   - **Re-crop Logic**: If confidence < threshold, adjust bbox, re-crop. Max 3 attempts.
   - **Security**: Strict cropping (no DL data).
3. **Vehicle Verification**:
   - OCR plate number (EasyOCR).
   - Match plate number pattern.

# Critical Files
- `core/face_privacy_cropper.py`: RetinaFace logic + re-crop loop.
- `core/dl_ocr.py`: OCR wrapper.
- `core/vlm_face_verifier.py`: Anthropic API integration (PII filtering).

# Verification
- Test: DL image with face vs PAN card with QR. Verify face detection fails/re-crops on PAN.
- Check: RAM usage monitoring (e.g., `psutil` in `main.py`).
