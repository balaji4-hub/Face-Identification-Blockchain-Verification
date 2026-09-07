"""
VeriFace Chain - Evidence Record Builder
Constructs standardized, tamper-evident digital evidence records from candidate intelligence.
Strict Privacy Standard: Ensures no biometric embeddings or facial landmarks are embedded in evidence.
"""
import uuid
from datetime import datetime
from config.settings import settings
from evidence.evidence_models import EvidenceRecord, CandidateAnalysis
from core.result_types import MatchDecision


class EvidenceBuilder:
    """Builds cryptographic-ready EvidenceRecord objects."""
    
    @staticmethod
    def build_evidence(
        candidate: CandidateAnalysis,
        input_image_sha256: str,
        threshold: float = None,
        model_name: str = None
    ) -> EvidenceRecord:
        """
        Constructs an EvidenceRecord from candidate analysis results.
        
        Args:
            candidate: Analyzed candidate match.
            input_image_sha256: SHA-256 hash of the consented input face image.
            threshold: Minimum similarity threshold configured for matching.
            model_name: Identifier for the biometric model used.
            
        Returns:
            EvidenceRecord ready for canonicalization and blockchain notarization.
        """
        return EvidenceRecord(
            evidence_id=str(uuid.uuid4()),
            source=candidate.source or "unknown_provider",
            page_url=candidate.page_url or "unknown_page",
            image_url=candidate.image_url or "unknown_image",
            image_sha256=candidate.image_sha256 or "",
            input_image_sha256=input_image_sha256,
            similarity_score=round(float(candidate.highest_similarity), 4),
            match_threshold=threshold or settings.FACE_MATCH_THRESHOLD,
            match_decision=candidate.match_status,
            model_name=model_name or settings.FACE_RECOGNITION_MODEL,
            discovered_at=datetime.utcnow().isoformat() + "Z",
            pipeline_version=settings.PIPELINE_VERSION
        )
