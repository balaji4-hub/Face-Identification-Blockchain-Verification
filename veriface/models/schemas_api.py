"""
VERIFACE CHAIN Pydantic Schemas for API
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class VerificationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============ Input Acquisition Schemas ============

class InputUploadRequest(BaseModel):
    """Request for face image upload"""
    auto_process: bool = True
    metadata: Optional[Dict[str, Any]] = None


class InputUploadResponse(BaseModel):
    """Response after image upload"""
    session_id: str
    input_image_id: str
    filename: str
    file_size: int
    mime_type: str
    image_hash: str
    dimensions: Dict[str, int]
    status: str
    message: str


# ============ Biometric Processing Schemas ============

class FaceLocation(BaseModel):
    """Face bounding box coordinates"""
    top: int
    right: int
    bottom: int
    left: int


class FaceLandmark(BaseModel):
    """Facial landmark coordinates"""
    eye_left: Optional[tuple] = None
    eye_right: Optional[tuple] = None
    nose: Optional[tuple] = None
    mouth_left: Optional[tuple] = None
    mouth_right: Optional[tuple] = None


class BiometricResultResponse(BaseModel):
    """Response from biometric processing"""
    biometric_id: str
    face_detected: bool
    detection_confidence: float
    quality_score: float
    quality_check_passed: bool
    embedding_size: int
    face_count: int
    face_locations: Optional[List[FaceLocation]] = None
    landmarks: Optional[FaceLandmark] = None
    processing_time_ms: float
    error_message: Optional[str] = None


# ============ Discovery Engine Schemas ============

class SearchProviderConfig(BaseModel):
    """Configuration for a search provider"""
    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None
    max_results: int = 50


class DiscoveryRequest(BaseModel):
    """Request to start candidate discovery"""
    search_providers: Optional[List[str]] = None
    max_candidates: Optional[int] = None
    timeout_seconds: int = 30


class DiscoveryResponse(BaseModel):
    """Response from discovery engine"""
    session_id: str
    candidates_found: int
    providers_searched: List[str]
    discovery_time_ms: float
    status: str


# ============ Candidate Intelligence Schemas ============

class CandidateInfo(BaseModel):
    """Information about a candidate match"""
    candidate_id: str
    provider_name: str
    external_id: Optional[str] = None
    image_url: Optional[str] = None
    face_detected: bool = False
    face_count: int = 0
    detection_confidence: float = 0.0
    similarity_score: float
    distance: Optional[float] = None
    rank: int
    is_top_match: bool
    metadata: Optional[Dict[str, Any]] = None


class CandidateAnalysisResponse(BaseModel):
    """Response from candidate intelligence analysis"""
    session_id: str
    total_candidates: int
    analyzed_candidates: int
    candidates: List[CandidateInfo]
    best_match: Optional[CandidateInfo] = None
    analysis_time_ms: float


# ============ Confidence Engine Schemas ============

class ConfidenceDecision(BaseModel):
    """Decision from confidence engine"""
    match_found: bool
    confidence_score: float
    decision_threshold: float
    ranking_position: Optional[int] = None
    reasoning: str


class ConfidenceResponse(BaseModel):
    """Response from confidence engine"""
    session_id: str
    match_found: bool
    confidence_score: float
    best_candidate_id: Optional[str] = None
    decision: ConfidenceDecision
    all_candidates_ranked: List[CandidateInfo]
    processing_time_ms: float


# ============ Evidence Vault Schemas ============

class ProvenanceEntry(BaseModel):
    """Entry in provenance chain"""
    stage: str
    timestamp: datetime
    data_hash: str
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EvidenceResponse(BaseModel):
    """Response from evidence vault"""
    session_id: str
    evidence_id: str
    evidence_hash: str
    canonical_image_hash: str
    evidence_json: Dict[str, Any]
    provenance_chain: List[ProvenanceEntry]
    quality_scores: Dict[str, float]
    created_at: datetime
    verified: bool


# ============ Crypto Engine Schemas ============

class MerkleProof(BaseModel):
    """Merkle proof for verification"""
    leaf_index: int
    leaf_hash: str
    merkle_root: str
    proof: List[str]
    hash_algorithm: str


class CryptoResult(BaseModel):
    """Result from crypto engine"""
    sha256_hash: str
    merkle_root: Optional[str] = None
    merkle_proof: Optional[MerkleProof] = None
    is_canonical: bool
    processing_time_ms: float


# ============ Blockchain Schemas ============

class BlockchainTransaction(BaseModel):
    """Blockchain transaction record"""
    transaction_hash: str
    block_number: int
    block_hash: str
    timestamp: datetime
    evidence_hash: str
    previous_hash: Optional[str] = None
    confirmations: int = 0


class BlockchainResponse(BaseModel):
    """Response from blockchain service"""
    session_id: str
    transaction: BlockchainTransaction
    merkle_root: str
    recorded: bool
    recording_time_ms: float


# ============ Verification Schemas ============

class VerificationResult(BaseModel):
    """Final verification result"""
    session_id: str
    status: VerificationStatus
    match_found: bool
    confidence_score: float
    best_candidate: Optional[CandidateInfo] = None
    evidence_hash: str
    blockchain_transaction: Optional[BlockchainTransaction] = None
    is_verified: bool
    tamper_detected: bool
    total_processing_time_ms: float


class VerificationChainResponse(BaseModel):
    """Full verification chain response"""
    session_id: str
    input_acquisition: InputUploadResponse
    biometric_processing: BiometricResultResponse
    discovery_engine: DiscoveryResponse
    candidate_intelligence: CandidateAnalysisResponse
    confidence_engine: ConfidenceResponse
    evidence_vault: EvidenceResponse
    crypto_engine: CryptoResult
    blockchain: BlockchainResponse
    verification: VerificationResult
    complete_chain: List[str] = []
    created_at: datetime
    completed_at: datetime


# ============ Audit Schemas ============

class AuditLogResponse(BaseModel):
    """Audit log entry"""
    id: int
    timestamp: datetime
    session_id: Optional[str]
    action: str
    details: Optional[Dict[str, Any]] = None


# ============ Error Schemas ============

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    database: str
    services: Dict[str, str]


# ============ Evidence Data Schema ============

class EvidenceData(BaseModel):
    """Full structured evidence data payload used by EvidenceVaultService"""
    session_id: str
    timestamp: str
    input: Dict[str, Any]
    biometric: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    decision: Dict[str, Any]
    metadata: Dict[str, Any] = {}