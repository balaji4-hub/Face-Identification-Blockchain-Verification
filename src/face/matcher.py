"""
VeriFace Chain - Phase 3: Face Matching / Comparison
=======================================================

This module handles ONLY comparing two already-generated face embeddings
(produced by Phase 2's encoder.py) and classifying how similar they are.

It is intentionally scoped to Phase 3:

    - Accept two normalized embedding vectors
    - Compute cosine similarity between them
    - Classify the result against configurable thresholds into one of:
          HIGH_CONFIDENCE_MATCH / POSSIBLE_MATCH / NO_MATCH
    - Return a structured result

It does NOT implement face search across a database, indexing, or
blockchain logic. It compares exactly two embeddings at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

# Must match the embedding length produced by Phase 2's encoder.py.
EXPECTED_EMBEDDING_DIM = 512


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class InvalidEmbeddingError(Exception):
    """Raised when an embedding is missing, the wrong shape, or degenerate."""


class MatchingError(Exception):
    """Raised when the similarity computation itself fails unexpectedly."""


# ---------------------------------------------------------------------------
# Result classification
# ---------------------------------------------------------------------------
class MatchResult(str, Enum):
    """The three possible outcomes of a face comparison."""
    HIGH_CONFIDENCE_MATCH = "HIGH_CONFIDENCE_MATCH"
    POSSIBLE_MATCH = "POSSIBLE_MATCH"
    NO_MATCH = "NO_MATCH"


@dataclass
class MatchThresholds:
    """
    Configurable similarity thresholds used to classify a comparison score.

    Classification rule:
        similarity >= high_confidence_threshold -> HIGH_CONFIDENCE_MATCH
        similarity >= possible_match_threshold  -> POSSIBLE_MATCH
        otherwise                               -> NO_MATCH
    """
    high_confidence_threshold: float = 0.65
    possible_match_threshold: float = 0.45

    def __post_init__(self) -> None:
        """Validate that thresholds are sane and consistently ordered."""
        if not (0.0 <= self.possible_match_threshold <= self.high_confidence_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy: "
                "0 <= possible_match_threshold <= high_confidence_threshold <= 1. "
                f"Got possible_match_threshold={self.possible_match_threshold}, "
                f"high_confidence_threshold={self.high_confidence_threshold}."
            )


@dataclass
class MatchOutcome:
    """Structured result of comparing two embeddings."""
    similarity: float
    result: MatchResult


# ---------------------------------------------------------------------------
# Core matcher class
# ---------------------------------------------------------------------------
class FaceMatcher:
    """
    Compares two face embeddings using cosine similarity and classifies
    the result against configurable thresholds.
    """

    def __init__(self, thresholds: Optional[MatchThresholds] = None) -> None:
        self.thresholds = thresholds or MatchThresholds()

    def _validate_embedding(self, embedding: np.ndarray, name: str) -> np.ndarray:
        if embedding is None:
            raise InvalidEmbeddingError(f"{name} is None.")

        try:
            arr = np.asarray(embedding, dtype=np.float32).flatten()
        except Exception as exc:
            raise InvalidEmbeddingError(f"{name} could not be converted to a NumPy array: {exc}") from exc

        if arr.ndim != 1 or arr.shape[0] != EXPECTED_EMBEDDING_DIM:
            raise InvalidEmbeddingError(
                f"{name} must be a 1D vector of length {EXPECTED_EMBEDDING_DIM}, "
                f"got shape {arr.shape}."
            )

        if float(np.linalg.norm(arr)) == 0.0:
            raise InvalidEmbeddingError(f"{name} has zero magnitude; cannot compare.")

        return arr

    def cosine_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        a = self._validate_embedding(embedding_a, "embedding_a")
        b = self._validate_embedding(embedding_b, "embedding_b")

        try:
            dot_product = float(np.dot(a, b))
            norm_a = float(np.linalg.norm(a))
            norm_b = float(np.linalg.norm(b))
            similarity = dot_product / (norm_a * norm_b)
        except Exception as exc:
            raise MatchingError(f"Failed to compute cosine similarity: {exc}") from exc

        # Guard against tiny floating-point overshoot beyond [-1, 1].
        return max(-1.0, min(1.0, similarity))

    def classify(self, similarity: float) -> MatchResult:
        if similarity >= self.thresholds.high_confidence_threshold:
            return MatchResult.HIGH_CONFIDENCE_MATCH
        if similarity >= self.thresholds.possible_match_threshold:
            return MatchResult.POSSIBLE_MATCH
        return MatchResult.NO_MATCH

    def compare(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> MatchOutcome:
        similarity = self.cosine_similarity(embedding_a, embedding_b)
        result = self.classify(similarity)
        return MatchOutcome(similarity=similarity, result=result)


# ---------------------------------------------------------------------------
# Console-friendly reporting helper
# ---------------------------------------------------------------------------
def print_match_report(label: str, outcome: MatchOutcome) -> None:
    """Print a human-readable summary of a comparison result."""
    print(f"\n{label}")
    print("-" * 40)
    print(f"  Similarity score : {outcome.similarity:.4f}")
    print(f"  Result           : {outcome.result.value}")
    print("-" * 40)


# ---------------------------------------------------------------------------
# Manual test entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Manual end-to-end test: detect + encode faces in two images, then compare.

    Usage:
        python -m src.face.matcher <image_path_1> <image_path_2>
    """
    import sys

    from src.face.detector import FaceDetector, InvalidImageError, NoFaceDetectedError
    from src.face.encoder import EncodingError, FaceEncoder, InvalidEmbeddingError as EncoderEmbeddingError

    if len(sys.argv) != 3:
        print("Usage: python -m src.face.matcher <image_path_1> <image_path_2>")
        sys.exit(1)

    image_path_1, image_path_2 = sys.argv[1], sys.argv[2]

    detector = FaceDetector()
    encoder = FaceEncoder()
    matcher = FaceMatcher()

    try:
        faces_1 = detector.detect_faces(image_path_1)
        faces_2 = detector.detect_faces(image_path_2)

        embedding_1 = encoder.encode_faces(image_path_1, [faces_1[0]])[0]
        embedding_2 = encoder.encode_faces(image_path_2, [faces_2[0]])[0]

        outcome = matcher.compare(embedding_1.embedding, embedding_2.embedding)
    except InvalidImageError as e:
        print(f"[INVALID IMAGE ERROR] {e}")
        sys.exit(2)
    except NoFaceDetectedError as e:
        print(f"[NO FACE ERROR] {e}")
        sys.exit(3)
    except (EncodingError, EncoderEmbeddingError) as e:
        print(f"[ENCODING ERROR] {e}")
        sys.exit(4)
    except (InvalidEmbeddingError, MatchingError) as e:
        print(f"[MATCHING ERROR] {e}")
        sys.exit(5)

    print_match_report(f"Comparing:\n  A: {image_path_1}\n  B: {image_path_2}", outcome)


if __name__ == "__main__":
    main()
