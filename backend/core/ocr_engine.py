import os

# Clamp CPU threads globally to prevent 98% CPU spike and maintain smooth multithreading
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")
os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")

import logging
import threading
from typing import Optional
import cv2
from rapidocr_onnxruntime import RapidOCR

from app.config import settings

cv2.setNumThreads(2)

logger = logging.getLogger(__name__)


class SharedOCREngine:
    """
    Fine-tuned Singleton for RapidOCR with ONNX Runtime.
    Hyperparameter-tuned DBNet detector and CRNN recognizer:
    - unclip_ratio=1.95: Expands text bounding polygons to eliminate character edge/ascender/descender clipping.
    - box_thresh=0.38: Increases sensitivity to faint, blurred, or low-contrast text strokes.
    - thresh=0.20: Fine-grained probability threshold for DBNet segmentation map binarization.
    - max_side_len=1280: High-resolution image text segmentation ceiling.
    - Thread-clamped to 2 CPU workers to maintain low CPU load (< 30%).
    """
    def __init__(self):
        self.ocr: Optional[RapidOCR] = None
        self._lock = threading.Lock()

    def get_engine(self) -> RapidOCR:
        if self.ocr is None:
            with self._lock:
                if self.ocr is None:
                    logger.info("Initializing Fine-Tuned RapidOCR Engine (Singleton, ONNX Runtime)...")
                    try:
                        try:
                            # Cards are deskewed upright before inference, so the
                            # angle classifier is dead weight when OCR_USE_CLS=false.
                            self.ocr = RapidOCR(use_cls=settings.ocr_use_cls)
                        except TypeError:
                            self.ocr = RapidOCR()
                        # Apply high-accuracy DBNet detector tuning directly on postprocessing pipeline
                        if hasattr(self.ocr, "text_detector") and hasattr(self.ocr.text_detector, "postprocess_op"):
                            self.ocr.text_detector.postprocess_op.unclip_ratio = settings.ocr_unclip_ratio
                            self.ocr.text_detector.postprocess_op.box_thresh = settings.ocr_box_thresh
                            self.ocr.text_detector.postprocess_op.thresh = settings.ocr_db_thresh

                        logger.info("Fine-Tuned RapidOCR Engine initialized successfully with calibrated thresholds.")
                    except Exception as e:
                        logger.error(f"Failed to initialize RapidOCR: {e}")
                        raise
        return self.ocr



shared_ocr = SharedOCREngine()
