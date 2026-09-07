"""Core package for VeriFace Chain."""
from core.exceptions import (
    VeriFaceError,
    InputValidationError,
    InvalidImageError,
    FaceDetectionError,
    FaceQualityError,
    SearchProviderError,
    CandidateFetchError,
    CanonicalizationError,
    BlockchainError,
    BlockchainUnavailableError,
    DuplicateEvidenceError,
    VerificationError
)
from core.logging import get_logger
from core.result_types import StageStatus, MatchDecision, VerificationState, StageResult

__all__ = [
    "VeriFaceError",
    "InputValidationError",
    "InvalidImageError",
    "FaceDetectionError",
    "FaceQualityError",
    "SearchProviderError",
    "CandidateFetchError",
    "CanonicalizationError",
    "BlockchainError",
    "BlockchainUnavailableError",
    "DuplicateEvidenceError",
    "VerificationError",
    "get_logger",
    "StageStatus",
    "MatchDecision",
    "VerificationState",
    "StageResult"
]
