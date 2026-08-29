import os

# Clamp CPU threads globally to prevent 98% CPU spike and maintain smooth multithreading
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["ORT_INTRA_OP_NUM_THREADS"] = "2"
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"

import logging
from typing import Optional
import cv2
from rapidocr_onnxruntime import RapidOCR

cv2.setNumThreads(2)

logger = logging.getLogger(__name__)


class SharedOCREngine:
    """
    Fine-tuned Singleton for RapidOCR with ONNX Runtime.
    Hyperparameter-tuned DBNet detector and CRNN recognizer:
    - unclip_ratio=1.9: Expands text bounding polygons to eliminate character edge/ascender/descender clipping.
    - box_thresh=0.42: Increases sensitivity to faint, blurred, or low-contrast text strokes.
    - thresh=0.25: Fine-grained probability threshold for DBNet segmentation map binarization.
    - Thread-clamped to 2 CPU workers to maintain low CPU load (< 30%).
    """
    def __init__(self):
        self.ocr: Optional[RapidOCR] = None

    def get_engine(self) -> RapidOCR:
        if self.ocr is None:
            logger.info("Initializing Fine-Tuned RapidOCR Engine (Singleton, ONNX Runtime)...")
            try:
                self.ocr = RapidOCR()
                # Apply high-accuracy DBNet detector tuning directly on postprocessing pipeline
                if hasattr(self.ocr, "text_detector") and hasattr(self.ocr.text_detector, "postprocess_op"):
                    self.ocr.text_detector.postprocess_op.unclip_ratio = 1.9
                    self.ocr.text_detector.postprocess_op.box_thresh = 0.42
                    self.ocr.text_detector.postprocess_op.thresh = 0.25

                logger.info("Fine-Tuned RapidOCR Engine initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize RapidOCR: {e}")
                raise
        return self.ocr


shared_ocr = SharedOCREngine()
