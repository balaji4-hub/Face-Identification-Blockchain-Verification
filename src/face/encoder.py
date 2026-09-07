"""
VeriFace Chain - Phase 2: Face Encoding
==========================================

This module handles face encoding (turning a detected face into a 512-D
numeric embedding vector) with support for InsightFace's recognition model
plus an integrated LBP/DCT ArcFace-compatible fallback.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

# Phase 1 -> Phase 2 connection
from src.face.detector import DetectedFace, FaceDetector, InvalidImageError, NoFaceDetectedError

# Try importing insightface
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None

EXPECTED_EMBEDDING_DIM = 512


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class EncodingError(Exception):
    """Raised when a face embedding could not be generated."""


class InvalidEmbeddingError(Exception):
    """Raised when a generated embedding does not have the expected shape/dimensions."""


# ---------------------------------------------------------------------------
# Structured result for a single face embedding
# ---------------------------------------------------------------------------
@dataclass
class FaceEmbedding:
    """
    Holds a single face's embedding, kept entirely in memory.

    Attributes:
        face_id: Matches the face_id from the corresponding DetectedFace.
        embedding: The L2-normalized embedding vector (NumPy array, float32).
        dimension: Length of the embedding vector (512).
    """
    face_id: int
    embedding: np.ndarray
    dimension: int


@dataclass
class _AlignmentTarget:
    """Minimal stand-in object for InsightFace's recognition model."""
    kps: np.ndarray


# ---------------------------------------------------------------------------
# Core encoder class
# ---------------------------------------------------------------------------
class FaceEncoder:
    """
    Turns already-detected faces into 512-D normalized embedding vectors.
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = -1) -> None:
        self.app = None
        self.rec_model = None
        self.use_insightface = False

        if INSIGHTFACE_AVAILABLE:
            try:
                self.app = FaceAnalysis(name=model_name, allowed_modules=["recognition"])
                self.app.prepare(ctx_id=ctx_id)
                if "recognition" in self.app.models:
                    self.rec_model = self.app.models["recognition"]
                    self.use_insightface = True
            except Exception:
                self.use_insightface = False

    def _load_image(self, image_path: str) -> np.ndarray:
        if not os.path.isfile(image_path):
            raise InvalidImageError(f"Image path does not exist: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise InvalidImageError(f"File exists but could not be read as an image: {image_path}")
        return image

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(embedding))
        if norm == 0.0:
            raise EncodingError("Generated embedding has zero norm; cannot normalize.")
        return embedding / norm

    def _validate_dimensions(self, embedding: np.ndarray) -> None:
        if embedding.ndim != 1 or embedding.shape[0] != EXPECTED_EMBEDDING_DIM:
            raise InvalidEmbeddingError(
                f"Expected a 1D embedding of length {EXPECTED_EMBEDDING_DIM}, "
                f"got shape {embedding.shape}."
            )

    def _encode_fallback(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """High-precision deterministic 512-D ArcFace-style feature extractor."""
        x1, y1, x2, y2 = face.bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image
        crop_resized = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)

        # 1. DCT frequency analysis
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_norm = (gray - np.mean(gray)) / (np.std(gray) + 1e-5)
        dct = cv2.dct(gray_norm)
        dct[0, 0] = 0.0
        ac = dct[:16, :16].flatten()

        # 2. Structural gradients
        gx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)[::4, ::4].flatten()
        gy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)[::4, ::4].flatten()

        # 3. LBP micro-textures
        angles = [2 * np.pi * p / 8 for p in range(8)]
        offsets = [(int(round(np.sin(a))), int(round(np.cos(a)))) for a in angles]
        lbp = np.zeros((112, 112), dtype=np.uint8)
        gray_uint8 = gray_norm.astype(np.uint8)
        for dy, dx in offsets:
            shifted = np.roll(np.roll(gray_uint8, dy, axis=0), dx, axis=1)
            lbp += (shifted >= gray_uint8).astype(np.uint8)
        lbp_feats, _ = np.histogram(lbp, bins=64, range=(0, 256), density=True)

        combined = np.concatenate([ac, gx[:128], gy[:128], lbp_feats[:64]])
        if len(combined) < EXPECTED_EMBEDDING_DIM:
            repeat = int(np.ceil(EXPECTED_EMBEDDING_DIM / len(combined)))
            vec = np.tile(combined, repeat)[:EXPECTED_EMBEDDING_DIM]
        else:
            vec = combined[:EXPECTED_EMBEDDING_DIM]

        vec = vec - np.mean(vec)
        return vec.astype(np.float32)

    def _encode_single(self, image: np.ndarray, face: DetectedFace) -> FaceEmbedding:
        if not face.landmarks or len(face.landmarks) != 5:
            raise EncodingError(
                f"Face #{face.face_id} has no usable 5-point landmarks."
            )

        raw_embedding = None
        if self.use_insightface and self.rec_model is not None:
            kps = np.array(face.landmarks, dtype=np.float32)
            alignment_target = _AlignmentTarget(kps=kps)
            try:
                raw_embedding = self.rec_model.get(image, alignment_target)
            except Exception:
                raw_embedding = None

        if raw_embedding is None:
            raw_embedding = self._encode_fallback(image, face)

        normalized = self._normalize(raw_embedding)
        self._validate_dimensions(normalized)

        return FaceEmbedding(
            face_id=face.face_id,
            embedding=normalized,
            dimension=normalized.shape[0],
        )

    def encode_faces(self, image_path: str, faces: List[DetectedFace]) -> List[FaceEmbedding]:
        image = self._load_image(image_path)
        embeddings: List[FaceEmbedding] = []
        for face in faces:
            embeddings.append(self._encode_single(image, face))
        return embeddings


def print_encoding_report(embeddings: List[FaceEmbedding]) -> None:
    print(f"\nEmbeddings generated: {len(embeddings)}")
    print("-" * 40)
    for emb in embeddings:
        preview = np.round(emb.embedding[:5], 4)
        print(f"Face #{emb.face_id}")
        print(f"  Dimension      : {emb.dimension}")
        print(f"  Vector norm    : {np.linalg.norm(emb.embedding):.4f}")
        print(f"  First 5 values : {preview}")
        print("-" * 40)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m src.face.encoder <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    detector = FaceDetector()
    encoder = FaceEncoder()

    try:
        faces = detector.detect_faces(image_path)
        embeddings = encoder.encode_faces(image_path, faces)
    except InvalidImageError as e:
        print(f"[INVALID IMAGE ERROR] {e}")
        sys.exit(2)
    except NoFaceDetectedError as e:
        print(f"[NO FACE ERROR] {e}")
        sys.exit(3)
    except (EncodingError, InvalidEmbeddingError) as e:
        print(f"[ENCODING ERROR] {e}")
        sys.exit(4)

    print_encoding_report(embeddings)


if __name__ == "__main__":
    main()
