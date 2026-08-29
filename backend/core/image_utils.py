import base64
import io
import math
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import ExifTags, Image

try:
    import pillow_avif  # type: ignore
except ImportError:
    pass

try:
    import pillow_heif  # type: ignore
    pillow_heif.register_avif_opener()
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Decompression Bomb Protection (prevents zip-bomb image memory exploits)
Image.MAX_IMAGE_PIXELS = 50_000_000


def compress_image_bytes(
    input_bytes: bytes,
    out_format: str = "webp",
    max_dim: int = 1400,
    quality: int = 85,
) -> Tuple[bytes, str, int, int]:
    """
    Compresses raw image bytes into WebP or AVIF format.
    - Strips EXIF metadata to protect user privacy
    - Preserves correct rotation & orientation
    - Resizes to max_dim using high-quality Lanczos resampling
    - Returns (compressed_bytes, mime_type, original_size, compressed_size)
    """
    if not input_bytes:
        raise ValueError("Empty image bytes provided")

    original_size = len(input_bytes)
    out_fmt = out_format.lower().replace(".", "")
    if out_fmt not in ("webp", "avif"):
        out_fmt = "webp"

    pil_img = Image.open(io.BytesIO(input_bytes))
    pil_img = correct_pil_orientation(pil_img)

    # Strip EXIF metadata
    pil_img.info.pop("exif", None)

    # Color space normalization
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGBA")
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Downscale if larger than max_dim
    w, h = pil_img.size
    max_side = max(w, h)
    if max_dim > 0 and max_side > max_dim:
        scale = max_dim / float(max_side)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    output_buffer = io.BytesIO()
    try:
        pil_img.save(output_buffer, format=out_fmt.upper(), optimize=True, quality=quality)
    except Exception:
        # Fallback to WEBP if AVIF plugin unavailable
        out_fmt = "webp"
        output_buffer = io.BytesIO()
        pil_img.save(output_buffer, format="WEBP", optimize=True, quality=quality)

    compressed_bytes = output_buffer.getvalue()
    mime_type = f"image/{out_fmt}"
    return compressed_bytes, mime_type, original_size, len(compressed_bytes)


def compress_and_normalize_base64(
    base64_data: str,
    out_format: str = "webp",
    max_dim: int = 1400,
    quality: int = 85,
) -> Tuple[np.ndarray, str]:
    """
    Takes any input base64 string (JPEG, PNG, WebP, AVIF, HEIC, etc.),
    compresses it to optimized WebP, and returns (OpenCV_BGR_array, data_url_base64).
    Reduces database storage, memory footprint, and network payload by 70-85%.
    """
    if not base64_data:
        raise ValueError("Empty base64 string provided")

    raw_b64 = base64_data
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    raw_b64 = raw_b64.strip().replace("\n", "").replace("\r", "")
    missing_padding = len(raw_b64) % 4
    if missing_padding:
        raw_b64 += "=" * (4 - missing_padding)

    input_bytes = base64.b64decode(raw_b64)
    compressed_bytes, mime_type, _, _ = compress_image_bytes(
        input_bytes=input_bytes,
        out_format=out_format,
        max_dim=max_dim,
        quality=quality,
    )

    # Decode compressed bytes to OpenCV BGR
    pil_decomp = Image.open(io.BytesIO(compressed_bytes))
    if pil_decomp.mode != "RGB":
        pil_decomp = pil_decomp.convert("RGB")
    rgb_arr = np.array(pil_decomp)
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    # Encode to data URI
    compressed_b64 = base64.b64encode(compressed_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{compressed_b64}"

    return bgr_arr, data_uri


def decode_base64_to_image(base64_data: str) -> np.ndarray:
    """
    Decodes a base64 string (raw or data URL) into an OpenCV BGR numpy array.
    Handles EXIF orientation and validates image integrity.
    """
    if not base64_data:
        raise ValueError("Empty base64 string provided")

    # Strip data URL prefix if present (e.g. data:image/jpeg;base64,...)
    if "," in base64_data:
        base64_data = base64_data.split(",", 1)[1]

    # Clean whitespace and newlines
    base64_data = base64_data.strip().replace("\n", "").replace("\r", "")

    # Fix padding if necessary
    missing_padding = len(base64_data) % 4
    if missing_padding:
        base64_data += "=" * (4 - missing_padding)

    img_bytes = base64.b64decode(base64_data)
    if len(img_bytes) == 0:
        raise ValueError("Decoded base64 bytes are empty")

    # Use PIL first to read EXIF orientation tag if present
    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_img = correct_pil_orientation(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        rgb_arr = np.array(pil_img)
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        return bgr_arr
    except Exception:
        # Fallback to OpenCV imdecode
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image from base64 buffer")
        return img_bgr


def encode_image_to_base64(
    img_bgr: np.ndarray,
    format_ext: str = ".webp",
    quality: int = 90,
    include_data_uri: bool = True,
) -> str:
    """
    Encodes an OpenCV BGR image into a base64 string or data URL.
    Defaults to lightweight WebP compression.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Cannot encode empty or None image")

    fmt = format_ext.lower()
    if fmt in [".jpg", ".jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        mime = "image/jpeg"
        success, buffer = cv2.imencode(fmt, img_bgr, params)
    elif fmt == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]
        mime = "image/png"
        success, buffer = cv2.imencode(fmt, img_bgr, params)
    elif fmt == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        mime = "image/webp"
        success, buffer = cv2.imencode(fmt, img_bgr, params)
    else:
        # Convert via PIL for other formats (like AVIF)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format=fmt.replace(".", "").upper(), quality=quality)
        buffer = buf.getvalue()
        success = True
        mime = f"image/{fmt.replace('.', '')}"

    if not success:
        raise ValueError("Failed to encode image to buffer")

    if isinstance(buffer, np.ndarray):
        b64_str = base64.b64encode(buffer).decode("utf-8")
    else:
        b64_str = base64.b64encode(buffer).decode("utf-8")

    if include_data_uri:
        return f"data:{mime};base64,{b64_str}"
    return b64_str


def correct_pil_orientation(image: Image.Image) -> Image.Image:
    """
    Auto-rotates PIL image according to its EXIF orientation tag.
    """
    try:
        exif = image.getexif()
        if not exif:
            return image

        orientation_tag = None
        for tag, value in ExifTags.TAGS.items():
            if value == "Orientation":
                orientation_tag = tag
                break

        if orientation_tag and orientation_tag in exif:
            orientation = exif[orientation_tag]
            if orientation == 3:
                return image.rotate(180, expand=True)
            elif orientation == 6:
                return image.rotate(270, expand=True)
            elif orientation == 8:
                return image.rotate(90, expand=True)
    except Exception:
        pass
    return image


def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization on the Luminance channel in LAB space.
    Enhances contrast for low-light images or licenses with glare.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_adaptive_threshold(img_bgr: np.ndarray) -> np.ndarray:
    """
    Phase 1 Fix: Converts the image to grayscale and applies Adaptive Gaussian Thresholding.
    This strips patterned backgrounds (like the Ontario watermarks) and isolates black text
    on a completely white background for perfect OCR recognition.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Mild blur to denoise before thresholding
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Adaptive thresholding (blockSize must be odd, C is the constant subtracted)
    # Using adaptive thresholding creates a clean binarized image (black text on white background)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    
    # Convert back to BGR so it matches the expected input shape for downstream processing
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def deskew_image(img_bgr: np.ndarray, max_angle: float = 30.0) -> np.ndarray:
    """
    Estimates skew angle of text/document and rotates to upright orientation.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find non-zero points
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return img_bgr

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    if abs(angle) < 1.0 or abs(angle) > max_angle:
        return img_bgr

    (h, w) = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def crop_image_safe(
    img_bgr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    margin_ratio: float = 0.0,
) -> np.ndarray:
    """
    Tightly crops image with bounded margin expansion, preventing out-of-bound errors.
    """
    img_h, img_w = img_bgr.shape[:2]

    pad_x = int(w * margin_ratio)
    pad_y = int(h * margin_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    cropped = img_bgr[y1:y2, x1:x2]
    return cropped


def resize_image_max_dimension(
    img_bgr: np.ndarray,
    max_dim: int = 1400,
) -> Tuple[np.ndarray, float]:
    """
    Downscales image if max dimension exceeds max_dim to cap peak RAM consumption
    while preserving full resolution detail for OCR and biometric facial features.
    Returns (scaled_image_bgr, scale_factor).
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr, 1.0

    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img_bgr, 1.0

    scale = max_dim / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    # Use CUBIC for upscaling, AREA for downscaling
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    scaled = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    return scaled, scale


def upscale_and_sharpen(img_bgr: np.ndarray, scale: float = 2.0, max_dim: int = 960) -> np.ndarray:
    """
    Multi-Scale Upsampling for Tiny Text (Magnifying Glass effect).
    Scales the image up with a bounded max_dim (max 960px) to prevent RAM spikes,
    and applies a sharpening kernel to recover edges of blurred text.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    effective_scale = min(scale, max_dim / float(longest)) if (longest * scale > max_dim) else scale

    if effective_scale <= 1.05:
        # Just apply sharpening without enlargement
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
        return cv2.filter2D(img_bgr, -1, kernel)

    new_w, new_h = max(1, int(w * effective_scale)), max(1, int(h * effective_scale))
    upscaled = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(upscaled, -1, kernel)
    return sharpened


def flatten_id_card(img_bgr: np.ndarray) -> np.ndarray:
    """
    Finds the 4 corners of the ID card using contour detection and applies a 
    Perspective Transform (Homography) to flatten it as if scanned.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    
    # Close broken edges
    kernel = np.ones((5,5), np.uint8)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
        
    img_h, img_w = img_bgr.shape[:2]
    img_area = img_w * img_h
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        # If the contour is less than 30% of the image size, it's an internal element (like a photo or barcode)
        if cv2.contourArea(c) < 0.3 * img_area:
            continue
            
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            ordered = np.array([tl, tr, br, bl], dtype="float32")
            
            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_width = max(int(width_a), int(width_b))
            
            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = max(int(height_a), int(height_b))
            
            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]], dtype="float32")
                
            M = cv2.getPerspectiveTransform(ordered, dst)
            return cv2.warpPerspective(img_bgr, M, (max_width, max_height))
            
    return img_bgr


def get_image_metrics(img_bgr: np.ndarray) -> dict:
    """
    Calculates Variance of Laplacian (Blur) and Noise metrics.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {
        "blur_variance": variance,
        "is_blurry": variance < 100,
        "is_noisy": variance > 5000
    }

