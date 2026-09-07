"""
VeriFace Chain - Domain Exceptions
Defines typed exception hierarchy across all subsystems.
"""


class VeriFaceError(Exception):
    """Base exception for all VeriFace Chain errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InputValidationError(VeriFaceError):
    """Raised when an input image file fails pre-flight validation."""
    pass


class InvalidImageError(InputValidationError):
    """Raised when an image is corrupt, unreadable, or not a supported format."""
    pass


class FaceDetectionError(VeriFaceError):
    """Raised when face detection pipeline fails."""
    pass


class FaceQualityError(VeriFaceError):
    """Raised when face quality analysis fails."""
    pass


class SearchProviderError(VeriFaceError):
    """Raised when an external or local search provider experiences failure."""
    pass


class CandidateFetchError(VeriFaceError):
    """Raised when candidate image download violates security or network constraints."""
    pass


class CanonicalizationError(VeriFaceError):
    """Raised when evidence dictionary canonicalization fails."""
    pass


class BlockchainError(VeriFaceError):
    """Raised when interaction with blockchain or smart contract fails."""
    pass


class BlockchainUnavailableError(BlockchainError):
    """Raised when RPC endpoint cannot be reached."""
    pass


class DuplicateEvidenceError(BlockchainError):
    """Raised when evidence hash is already registered on chain."""
    pass


class VerificationError(VeriFaceError):
    """Raised when evidence verification fails or detects tampering."""
    pass
