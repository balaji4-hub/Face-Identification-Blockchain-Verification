"""
VeriFace Chain - Evidence and Domain Data Models
Strongly-typed Pydantic models for face detection, candidates, evidence provenance, and verification.
Strict Privacy Standard: Biometric vectors are never included in evidence or blockchain models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from core.result_types import MatchDecision, VerificationState


class FaceDetection(BaseModel):
    """Bounding box, confidence, and facial landmarks for a detected face."""
    bounding_box: List[int] = Field(
        ...,
        description="Bounding box coordinates in [top, right, bottom, left] order"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    landmarks: Optional[Dict[str, Any]] = Field(default=None, description="Facial landmarks if available")
    face_index: int = Field(default=0, description="Index of face in image")


class FaceEmbedding(BaseModel):
    """In-memory transient representation of a normalized biometric face embedding."""
    embedding: List[float] = Field(..., description="Biometric feature vector")
    dimension: int = Field(default=512, description="Dimension of embedding vector")
    normalized: bool = Field(default=True, description="Whether vector is L2 normalized")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FaceQualityResult(BaseModel):
    """Evaluation result from the face quality analyzer."""
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score (0.0 - 1.0)")
    accepted: bool = Field(..., description="Whether the face meets minimum quality standards")
    blur_score: float = Field(default=0.0, description="Variance of Laplacian blur estimate")
    face_ratio: float = Field(default=0.0, description="Face area relative to image area")
    resolution: Dict[str, int] = Field(default_factory=dict, description="Width and height of face")
    issues: List[str] = Field(default_factory=list, description="Quality deficiencies if rejected")


class CandidateResult(BaseModel):
    """Candidate post/image discovered during visual search."""
    source: str = Field(..., description="Search provider or origin domain")
    page_url: str = Field(..., description="Source web page or article URL")
    image_url: str = Field(..., description="Direct publicly accessible candidate image URL")
    title: Optional[str] = Field(default=None, description="Page title or caption")
    discovered_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 discovery timestamp"
    )


class CandidateAnalysis(BaseModel):
    """Detailed intelligence analysis of a fetched candidate image."""
    candidate_id: str = Field(..., description="Unique identifier for the candidate")
    source: str = Field(..., description="Provider source")
    page_url: str = Field(..., description="Page URL")
    image_url: str = Field(..., description="Image URL")
    image_sha256: Optional[str] = Field(default=None, description="Cryptographic SHA-256 of candidate image")
    faces_detected: int = Field(default=0, description="Count of faces detected in candidate image")
    highest_similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Maximum cosine similarity score")
    best_face_index: Optional[int] = Field(default=None, description="Index of candidate face with highest match")
    match_status: MatchDecision = Field(default=MatchDecision.NO_MATCH, description="Decision classification")
    rank: int = Field(default=0, description="Ranking position (1-indexed)")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")


class EvidenceRecord(BaseModel):
    """
    Structured tamper-evident digital evidence provenance record.
    NOTE: Biometric embeddings are strictly excluded to enforce user privacy.
    """
    evidence_id: str = Field(..., description="Unique UUID for this evidence record")
    source: str = Field(..., description="Origin provider or platform where candidate was discovered")
    page_url: str = Field(..., description="Source page URL")
    image_url: str = Field(..., description="Candidate image URL")
    image_sha256: str = Field(..., description="SHA-256 fingerprint of the candidate image")
    input_image_sha256: str = Field(..., description="SHA-256 fingerprint of the consented input image")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score against query face")
    match_threshold: float = Field(..., description="Configured threshold required for match")
    match_decision: MatchDecision = Field(..., description="Final confidence decision classification")
    model_name: str = Field(..., description="Computer vision model architecture identifier")
    discovered_at: str = Field(..., description="ISO 8601 timestamp when candidate was discovered")
    pipeline_version: str = Field(..., description="Semantic version of VeriFace Chain pipeline")


class BlockchainReceipt(BaseModel):
    """Receipt confirming registration of evidence fingerprint on blockchain."""
    transaction_hash: str = Field(..., description="Transaction hash on blockchain")
    block_number: int = Field(..., description="Block number containing transaction")
    contract_address: str = Field(..., description="Address of EvidenceRegistry smart contract")
    chain_id: int = Field(..., description="Network chain ID (e.g., 31337 for local)")
    registered_at: str = Field(..., description="Timestamp of registration")
    gas_used: Optional[int] = Field(default=None, description="Gas consumed by transaction")
    status: str = Field(default="SUCCESS", description="Transaction status")


class VerificationResult(BaseModel):
    """Independent verification result against blockchain state."""
    verified: bool = Field(..., description="True if evidence hash matches registered on-chain record")
    evidence_hash: str = Field(..., description="SHA-256 fingerprint calculated from canonical evidence")
    on_chain: bool = Field(..., description="Whether the evidence hash exists in smart contract")
    blockchain_timestamp: Optional[int] = Field(default=None, description="Unix timestamp recorded on chain")
    uploader: Optional[str] = Field(default=None, description="Address of the entity that registered evidence")
    verification_state: VerificationState = Field(..., description="Granular verification outcome")
    reason: str = Field(..., description="Detailed explanation of verification decision")
