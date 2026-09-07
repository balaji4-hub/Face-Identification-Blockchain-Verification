"""
VeriFace Chain - Candidate Analysis
=====================================

Algorithm:
    INPUT EMBEDDING
    FOR EACH CANDIDATE:
        Download image          (Phase 5: candidate_fetcher.py)
        Detect faces            (Phase 1: detector.py)
        FOR EACH FACE:
            Generate embedding  (Phase 2: encoder.py)
            Compare to input    (Phase 3: matcher.py)
        Keep highest score
    Rank all candidates by their highest score
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.face.detector import FaceDetector, InvalidImageError, NoFaceDetectedError
from src.face.encoder import EncodingError, FaceEncoder, InvalidEmbeddingError as EncoderEmbeddingError
from src.face.matcher import (
    FaceMatcher,
    InvalidEmbeddingError as MatcherEmbeddingError,
    MatchingError,
    MatchResult,
)
from src.search.candidate_fetcher import (
    CandidateFetcher,
    DownloadError,
    FileTooLargeError,
    InvalidContentTypeError,
    InvalidImageContentError,
    fetched_candidate_image,
)
from src.search.search_provider import CandidateResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class CandidateAnalysisError(Exception):
    """Raised for unexpected failures in the analyzer itself."""


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------
@dataclass
class CandidateAnalysis:
    """The fully analyzed result for one candidate image."""
    candidate_id: str
    source: str
    page_url: str
    image_url: str
    faces_detected: int
    best_face_index: Optional[int]
    highest_similarity: Optional[float]
    decision: MatchResult
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core analyzer class
# ---------------------------------------------------------------------------
class CandidateAnalyzer:
    """
    Given an input face embedding and a list of search candidates,
    downloads each candidate image, detects and encodes every face in it,
    compares each face to the input embedding, and ranks all candidates by
    their best score.
    """

    def __init__(
        self,
        detector: FaceDetector,
        encoder: FaceEncoder,
        matcher: FaceMatcher,
        fetcher: CandidateFetcher,
    ) -> None:
        self.detector = detector
        self.encoder = encoder
        self.matcher = matcher
        self.fetcher = fetcher

    def analyze_candidates(
        self, input_embedding: np.ndarray, candidates: List[CandidateResult]
    ) -> List[CandidateAnalysis]:
        results: List[CandidateAnalysis] = []

        for candidate in candidates:
            analysis = self._analyze_single_candidate(input_embedding, candidate)
            results.append(analysis)

        results.sort(
            key=lambda a: a.highest_similarity if a.highest_similarity is not None else -1.0,
            reverse=True,
        )
        return results

    def _analyze_single_candidate(
        self, input_embedding: np.ndarray, candidate: CandidateResult
    ) -> CandidateAnalysis:
        candidate_id = self._make_candidate_id(candidate)
        logger.info("Analyzing candidate %s (%s)", candidate_id, candidate.image_url)

        try:
            with fetched_candidate_image(self.fetcher, candidate) as fetched:
                faces = self._safe_detect(fetched.local_path, candidate_id)
                if not faces:
                    logger.info("Candidate %s: no faces detected", candidate_id)
                    return CandidateAnalysis(
                        candidate_id=candidate_id,
                        source=candidate.source,
                        page_url=candidate.page_url,
                        image_url=candidate.image_url,
                        faces_detected=0,
                        best_face_index=None,
                        highest_similarity=None,
                        decision=MatchResult.NO_MATCH,
                    )

                embeddings = self.encoder.encode_faces(fetched.local_path, faces)

                best_face_index, highest_similarity = self._find_best_match(
                    input_embedding, embeddings
                )
                decision = self.matcher.classify(highest_similarity)

                logger.info(
                    "Candidate %s: %d face(s), best_face_index=%s, "
                    "highest_similarity=%.4f, decision=%s",
                    candidate_id,
                    len(faces),
                    best_face_index,
                    highest_similarity,
                    decision.value,
                )

                return CandidateAnalysis(
                    candidate_id=candidate_id,
                    source=candidate.source,
                    page_url=candidate.page_url,
                    image_url=candidate.image_url,
                    faces_detected=len(faces),
                    best_face_index=best_face_index,
                    highest_similarity=highest_similarity,
                    decision=decision,
                )

        except (
            DownloadError,
            FileTooLargeError,
            InvalidContentTypeError,
            InvalidImageContentError,
            InvalidImageError,
            EncodingError,
            EncoderEmbeddingError,
            MatcherEmbeddingError,
            MatchingError,
        ) as exc:
            logger.warning("Candidate %s failed to process: %s", candidate_id, exc)
            return CandidateAnalysis(
                candidate_id=candidate_id,
                source=candidate.source,
                page_url=candidate.page_url,
                image_url=candidate.image_url,
                faces_detected=0,
                best_face_index=None,
                highest_similarity=None,
                decision=MatchResult.NO_MATCH,
                error=str(exc),
            )

    def _safe_detect(self, local_path: str, candidate_id: str) -> list:
        try:
            return self.detector.detect_faces(local_path)
        except NoFaceDetectedError:
            return []

    def _find_best_match(
        self, input_embedding: np.ndarray, embeddings: list
    ) -> tuple[int, float]:
        best_index = -1
        best_similarity = -1.0

        for emb in embeddings:
            similarity = self.matcher.cosine_similarity(input_embedding, emb.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = emb.face_id

        return best_index, best_similarity

    @staticmethod
    def _make_candidate_id(candidate: CandidateResult) -> str:
        digest = hashlib.sha256(candidate.image_url.encode("utf-8")).hexdigest()
        return f"candidate_{digest[:12]}"


# ---------------------------------------------------------------------------
# Console-friendly reporting helper
# ---------------------------------------------------------------------------
def print_ranked_report(analyses: List[CandidateAnalysis]) -> None:
    print(f"\nRanked candidates: {len(analyses)}")
    print("=" * 60)
    for rank, analysis in enumerate(analyses, start=1):
        print(f"#{rank}  {analysis.candidate_id}")
        print(f"    Source             : {analysis.source}")
        print(f"    Page URL           : {analysis.page_url}")
        print(f"    Image URL          : {analysis.image_url}")
        print(f"    Faces detected     : {analysis.faces_detected}")
        print(f"    Best face index    : {analysis.best_face_index}")
        if analysis.highest_similarity is not None:
            print(f"    Highest similarity : {analysis.highest_similarity:.4f}")
        else:
            print(f"    Highest similarity : N/A")
        print(f"    Decision           : {analysis.decision.value}")
        if analysis.error:
            print(f"    Error              : {analysis.error}")
        print("-" * 60)


# ---------------------------------------------------------------------------
# Manual test entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.face.candidate_analyzer "
            "<probe_image_path> <candidate_url_1> [<candidate_url_2> ...]"
        )
        sys.exit(1)

    probe_image_path = sys.argv[1]
    candidate_urls = sys.argv[2:]

    detector = FaceDetector()
    encoder = FaceEncoder()
    matcher = FaceMatcher()
    fetcher = CandidateFetcher()
    analyzer = CandidateAnalyzer(detector, encoder, matcher, fetcher)

    try:
        probe_faces = detector.detect_faces(probe_image_path)
        probe_embedding = encoder.encode_faces(probe_image_path, [probe_faces[0]])[0]
    except (InvalidImageError, NoFaceDetectedError) as e:
        print(f"[PROBE IMAGE ERROR] {e}")
        sys.exit(2)
    except (EncodingError, EncoderEmbeddingError) as e:
        print(f"[PROBE ENCODING ERROR] {e}")
        sys.exit(3)

    candidates = [
        CandidateResult(source="manual_test", page_url=url, image_url=url, title=None)
        for url in candidate_urls
    ]

    analyses = analyzer.analyze_candidates(probe_embedding.embedding, candidates)
    print_ranked_report(analyses)


if __name__ == "__main__":
    main()
