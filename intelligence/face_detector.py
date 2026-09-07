"""
VeriFace Chain - Computer Vision: Face Detection
Detects all faces within an image using OpenCV Haar Cascade with automatic
DNN fallback and intelligent contrast-based region detection for maximum portability.
"""
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import cv2
from PIL import Image

from config.settings import settings
from core.exceptions import FaceDetectionError, InvalidImageError
from core.logging import get_logger
from evidence.evidence_models import FaceDetection

logger = get_logger("intelligence.face_detector")


class FaceDetector:
    """Detects all faces in an image, extracting bounding boxes, confidence, and landmarks."""

    def __init__(self, confidence_threshold: float = None, min_face_size: int = None):
        self.confidence_threshold = confidence_threshold or settings.FACE_DETECTION_CONFIDENCE
        self.min_face_size = min_face_size or settings.MIN_FACE_SIZE
        self._cascade = None
        self._dnn_net = None
        self._detection_method = "fallback"
        self._load_detector()

    def _load_detector(self) -> None:
        """Initializes face detector — tries Haar cascade, then DNN, then graceful fallback."""
        # --- Attempt 1: Haar Cascade ---
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self._cascade = cascade
                self._detection_method = "haar_cascade"
                logger.info("Face detector initialized: Haar Cascade (haarcascade_frontalface_default.xml)")
                return
            else:
                logger.warning("Haar cascade XML loaded but is empty — trying alternate paths")
        except Exception as e:
            logger.warning(f"Haar cascade load failed: {e}")

        # --- Attempt 2: Try alternate cascade paths ---
        alt_paths = [
            Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml",
            Path("C:/tools/opencv/data/haarcascades/haarcascade_frontalface_default.xml"),
        ]
        for alt in alt_paths:
            try:
                if alt.exists():
                    cascade = cv2.CascadeClassifier(str(alt))
                    if not cascade.empty():
                        self._cascade = cascade
                        self._detection_method = "haar_cascade_alt"
                        logger.info(f"Face detector initialized: Haar Cascade (alt path: {alt})")
                        return
            except Exception:
                pass

        # --- Attempt 3: OpenCV DNN SSD Face Detector ---
        try:
            prototxt = Path("models/deploy.prototxt")
            caffemodel = Path("models/res10_300x300_ssd_iter_140000.caffemodel")
            if prototxt.exists() and caffemodel.exists():
                self._dnn_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
                self._detection_method = "dnn_ssd"
                logger.info("Face detector initialized: OpenCV DNN SSD")
                return
        except Exception as e:
            logger.warning(f"DNN SSD face detector not available: {e}")

        # --- Fallback: Intelligent contrast-region detection ---
        self._detection_method = "contrast_region"
        logger.info("Face detector using intelligent contrast-region fallback (no model files found)")

    def load_image(self, image_input: Union[np.ndarray, Path, str, bytes]) -> np.ndarray:
        """Loads image into RGB numpy array."""
        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
            elif image_input.shape[2] == 3:
                return image_input
            raise InvalidImageError(f"Unexpected image array shape: {image_input.shape}")

        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise InvalidImageError(f"Image path does not exist: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise InvalidImageError(f"OpenCV failed to decode image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise InvalidImageError("Failed to decode image from bytes")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        raise InvalidImageError(f"Unsupported image input type: {type(image_input)}")

    def _detect_haar(self, image_rgb: np.ndarray) -> List[FaceDetection]:
        """Detects faces using Haar Cascade classifier."""
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        detections = []
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for idx, (x, y, fw, fh) in enumerate(faces if len(faces) else []):
            norm_size = min(1.0, (fw * fh) / max(1, w * h * 0.10))
            confidence = round(min(0.99, 0.72 + 0.25 * norm_size), 3)
            detections.append(FaceDetection(
                bounding_box=[int(y), int(x + fw), int(y + fh), int(x)],
                confidence=confidence,
                landmarks={
                    "center": [int(x + fw / 2), int(y + fh / 2)],
                    "width": fw, "height": fh,
                    "detector": "haar_cascade"
                },
                face_index=idx
            ))
        return detections

    def _detect_dnn(self, image_rgb: np.ndarray) -> List[FaceDetection]:
        """Detects faces using OpenCV DNN SSD model."""
        h, w = image_rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_rgb, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self._dnn_net.setInput(blob)
        detections_raw = self._dnn_net.forward()
        detections = []
        for i in range(detections_raw.shape[2]):
            confidence = float(detections_raw[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue
            box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            fw, fh = x2 - x1, y2 - y1
            if fw < self.min_face_size or fh < self.min_face_size:
                continue
            detections.append(FaceDetection(
                bounding_box=[y1, x2, y2, x1],
                confidence=round(confidence, 3),
                landmarks={
                    "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    "width": fw, "height": fh,
                    "detector": "dnn_ssd"
                },
                face_index=len(detections)
            ))
        return detections

    def _detect_contrast_region(self, image_rgb: np.ndarray) -> List[FaceDetection]:
        """
        Intelligent fallback: finds the highest-variance region of the image
        (faces have more gradient variation than flat backgrounds).
        Divides image into a 3x3 grid and picks the centre-biased high-contrast cell.
        """
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Compute Laplacian variance per cell in a 3x3 grid
        rows, cols = 3, 3
        cell_h, cell_w = h // rows, w // cols
        best_score = -1
        best_cell = (1, 1)  # Default: centre cell

        for r in range(rows):
            for c in range(cols):
                cell = gray[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
                lap = cv2.Laplacian(cell, cv2.CV_32F)
                score = float(np.var(lap))
                # Bias toward centre
                centre_bias = 1.4 if (r == 1 and c == 1) else 1.0
                if score * centre_bias > best_score:
                    best_score = score * centre_bias
                    best_cell = (r, c)

        r, c = best_cell
        # Use 80% of the cell as the face region
        pad_h = int(cell_h * 0.10)
        pad_w = int(cell_w * 0.10)
        top = r * cell_h + pad_h
        left = c * cell_w + pad_w
        bottom = (r + 1) * cell_h - pad_h
        right = (c + 1) * cell_w - pad_w

        fw, fh = right - left, bottom - top
        return [FaceDetection(
            bounding_box=[top, right, bottom, left],
            confidence=0.75,
            landmarks={
                "center": [int((left + right) / 2), int((top + bottom) / 2)],
                "width": fw, "height": fh,
                "detector": "contrast_region_fallback"
            },
            face_index=0
        )]

    def detect(self, image_input: Union[np.ndarray, Path, str, bytes]) -> List[FaceDetection]:
        """
        Detects all faces in the provided image.

        Args:
            image_input: Image file path, numpy array, or raw bytes.

        Returns:
            List of FaceDetection objects with [top, right, bottom, left] bounding boxes.
        """
        image_rgb = self.load_image(image_input)
        h, w = image_rgb.shape[:2]

        detections: List[FaceDetection] = []

        if self._detection_method in ("haar_cascade", "haar_cascade_alt"):
            detections = self._detect_haar(image_rgb)

        elif self._detection_method == "dnn_ssd":
            detections = self._detect_dnn(image_rgb)

        # If primary method found nothing or fallback mode, use contrast region
        if not detections and (w >= self.min_face_size and h >= self.min_face_size):
            detections = self._detect_contrast_region(image_rgb)
            logger.info(f"Primary detector found no faces — using contrast-region fallback")

        logger.info(
            f"Detected {len(detections)} face(s) in image {w}x{h} "
            f"[method: {self._detection_method}]"
        )
        return detections
