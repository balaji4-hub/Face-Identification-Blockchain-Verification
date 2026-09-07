"""
STAGE 6: EVIDENCE VAULT
Handles evidence JSON, image hash, and provenance tracking
"""
import json
import time
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from veriface.core.config import settings
from veriface.models.database import AsyncSession
from veriface.models.schemas import VerificationSession, Evidence, InputImage, BiometricResult, Candidate
from veriface.models.schemas_api import EvidenceResponse, ProvenanceEntry


@dataclass
class EvidenceData:
    """Structured evidence data for a verification session"""
    session_id: str
    timestamp: str
    input: Dict[str, Any]
    biometric: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    decision: Dict[str, Any]
    metadata: Dict[str, Any]


class EvidenceVaultService:
    """Service for creating and managing evidence records"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    def create_evidence_json(
        self,
        session_id: str,
        input_image: InputImage,
        biometric_result: BiometricResult,
        candidates: List[Candidate],
        decision: Dict[str, Any]
    ) -> EvidenceData:
        """Create structured evidence JSON"""
        
        # Format input data
        input_data = {
            "id": input_image.id,
            "filename": input_image.filename,
            "file_size": input_image.file_size,
            "mime_type": input_image.mime_type,
            "image_hash": input_image.image_hash,
            "dimensions": {
                "width": input_image.width,
                "height": input_image.height,
                "channels": input_image.channels
            },
            "captured_at": input_image.captured_at.isoformat() if input_image.captured_at else None,
            "metadata": input_image.extra_metadata
        }
        
        # Format biometric data
        biometric_data = {
            "id": biometric_result.id,
            "face_detected": biometric_result.face_detected,
            "detection_confidence": biometric_result.detection_confidence,
            "quality_score": biometric_result.quality_score,
            "quality_check_passed": biometric_result.quality_check_passed,
            "embedding_size": biometric_result.embedding_size,
            "face_count": len(biometric_result.face_locations) if biometric_result.face_locations else 0,
            "processing_time_ms": biometric_result.processing_time_ms,
            "embedding_model": biometric_result.embedding_model,
            "error": biometric_result.error_message
        }
        
        # Format candidate data
        candidate_data = []
        for c in candidates:
            candidate_data.append({
                "id": c.id,
                "provider_name": c.provider_name,
                "external_id": c.external_id,
                "image_url": c.candidate_image_url,
                "face_detected": c.face_detected,
                "face_count": c.face_count,
                "detection_confidence": c.detection_confidence,
                "similarity_score": c.similarity_score,
                "distance": c.distance,
                "rank": c.rank,
                "is_top_match": c.is_top_match,
                "metadata": c.extra_metadata
            })
        
        # Create evidence data structure
        evidence = EvidenceData(
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            input=input_data,
            biometric=biometric_data,
            candidates=candidate_data,
            decision=decision,
            metadata={
                "version": settings.app_version,
                "evidence_version": "1.0"
            }
        )
        
        return evidence
    
    def calculate_evidence_hash(self, evidence) -> str:
        """Calculate SHA-256 hash of evidence JSON"""
        from dataclasses import asdict as dc_asdict
        import dataclasses
        if dataclasses.is_dataclass(evidence) and not isinstance(evidence, type):
            evidence_dict = dc_asdict(evidence)
        elif hasattr(evidence, "model_dump"):
            evidence_dict = evidence.model_dump()
        else:
            evidence_dict = dict(evidence)
        evidence_json = json.dumps(evidence_dict, sort_keys=True, default=str)
        return hashlib.sha256(evidence_json.encode()).hexdigest()
    
    def calculate_provenance_score(self, evidence: EvidenceData) -> float:
        """Calculate overall quality score from evidence"""
        # Biometric quality
        biometric_score = evidence.biometric.get("quality_score", 0.0) if evidence.biometric else 0.0
        
        # Check if face was detected
        face_detected = evidence.biometric.get("face_detected", False) if evidence.biometric else False
        
        # Candidate comparison quality
        candidates = evidence.candidates
        if candidates:
            # Average similarity of top matches
            avg_similarity = sum(c.get("similarity_score", 0) for c in candidates) / len(candidates)
            comparison_score = avg_similarity
        else:
            comparison_score = 0.0
        
        # Provenance completeness score
        provenance_score = 1.0 if (
            evidence.input and 
            evidence.biometric and 
            face_detected
        ) else 0.5
        
        # Weighted combination
        overall_score = (
            0.4 * biometric_score +
            0.4 * comparison_score +
            0.2 * provenance_score
        )
        
        return overall_score
    
    def build_provenance_chain(
        self,
        session_id: str,
        input_image: InputImage,
        biometric_result: BiometricResult,
        candidates: List[Candidate],
        decision: Dict[str, Any]
    ) -> List[ProvenanceEntry]:
        """Build provenance chain for the verification"""
        chain = []
        timestamp = datetime.utcnow()
        
        # Stage 1: Input acquisition
        chain.append(ProvenanceEntry(
            stage="INPUT_ACQUISITION",
            timestamp=timestamp,
            data_hash=input_image.image_hash,
            input_hash=None,
            output_hash=input_image.image_hash,
            metadata={
                "filename": input_image.filename,
                "file_size": input_image.file_size,
                "dimensions": {
                    "width": input_image.width,
                    "height": input_image.height
                }
            }
        ))
        
        # Stage 2: Biometric processing
        biometric_hash = biometric_result.error_message or hashlib.sha256(
            str(biometric_result.quality_score).encode()
        ).hexdigest()[:16]
        
        chain.append(ProvenanceEntry(
            stage="BIOMETRIC_PROCESSING",
            timestamp=timestamp,
            data_hash=biometric_hash,
            input_hash=input_image.image_hash,
            output_hash=biometric_hash,
            metadata={
                "face_detected": biometric_result.face_detected,
                "quality_score": biometric_result.quality_score,
                "embedding_size": biometric_result.embedding_size
            }
        ))
        
        # Stage 3: Discovery & Candidate Intelligence
        candidate_hash = hashlib.sha256(
            str(len(candidates)).encode()
        ).hexdigest()[:16]
        
        chain.append(ProvenanceEntry(
            stage="CANDIDATE_INTELLIGENCE",
            timestamp=timestamp,
            data_hash=candidate_hash,
            input_hash=biometric_hash,
            output_hash=candidate_hash,
            metadata={
                "candidates_found": len(candidates),
                "providers_searched": list(set(c.provider_name for c in candidates))
            }
        ))
        
        # Stage 4: Confidence Decision
        decision_hash = hashlib.sha256(
            json.dumps(decision).encode()
        ).hexdigest()
        
        chain.append(ProvenanceEntry(
            stage="CONFIDENCE_ENGINE",
            timestamp=timestamp,
            data_hash=decision_hash,
            input_hash=candidate_hash,
            output_hash=decision_hash,
            metadata=decision
        ))
        
        return chain
    
    async def create_evidence(
        self,
        session_id: str
    ) -> EvidenceResponse:
        """
        Create evidence record for a verification session
        
        Args:
            session_id: Verification session ID
            
        Returns:
            EvidenceResponse with evidence details
        """
        start_time = time.time()
        
        # Get session data
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one()
        
        # Get related data
        input_image = (await self.session.execute(
            select(InputImage).where(InputImage.session_id == session_id)
        )).scalar_one()
        
        biometric_result = (await self.session.execute(
            select(BiometricResult).where(BiometricResult.session_id == session_id)
        )).scalar_one()
        
        candidates = (await self.session.execute(
            select(Candidate).where(Candidate.session_id == session_id)
            .order_by(Candidate.rank)
        )).scalars().all()
        
        # Decision info
        decision = {
            "match_found": session.match_found,
            "confidence_score": session.match_confidence,
            "status": session.status
        }
        
        # Create evidence data
        evidence_data = self.create_evidence_json(
            session_id, input_image, biometric_result, candidates, decision
        )
        
        # Calculate hashes
        evidence_hash = self.calculate_evidence_hash(evidence_data)
        canonical_image_hash = input_image.image_hash
        
        # Build provenance chain
        provenance_chain = self.build_provenance_chain(
            session_id, input_image, biometric_result, candidates, decision
        )
        
        # Calculate quality scores
        overall_quality = self.calculate_provenance_score(evidence_data)
        biometric_quality = biometric_result.quality_score if biometric_result else 0.0
        
        if candidates:
            comparison_quality = max(c.similarity_score for c in candidates) if candidates else 0.0
        else:
            comparison_quality = 0.0
        
        # Create evidence record
        evidence_id = str(uuid.uuid4())
        evidence = Evidence(
            id=evidence_id,
            session_id=session_id,
            evidence_json=asdict(evidence_data),
            evidence_hash=evidence_hash,
            canonical_image_hash=canonical_image_hash,
            provenance_chain=[asdict(p) for p in provenance_chain],
            source_metadata=input_image.extra_metadata,
            overall_quality_score=overall_quality,
            biometric_quality_score=biometric_quality,
            comparison_quality_score=comparison_quality,
            created_at=datetime.utcnow(),
            verified_at=None
        )
        
        self.session.add(evidence)
        
        # Update session
        session.status = "evidence_created"
        
        await self.session.commit()
        
        processing_time = (time.time() - start_time) * 1000
        
        return EvidenceResponse(
            session_id=session_id,
            evidence_id=evidence_id,
            evidence_hash=evidence_hash,
            canonical_image_hash=canonical_image_hash,
            evidence_json=asdict(evidence_data),
            provenance_chain=provenance_chain,
            quality_scores={
                "overall": overall_quality,
                "biometric": biometric_quality,
                "comparison": comparison_quality
            },
            created_at=evidence.created_at,
            verified=False
        )
    
    async def get_evidence(self, session_id: str) -> Optional[Evidence]:
        """Get evidence for a session"""
        result = await self.session.execute(
            select(Evidence).where(Evidence.session_id == session_id)
        )
        return result.scalar_one_or_none()
    
    async def verify_evidence_integrity(self, evidence: Evidence) -> bool:
        """Verify that evidence hasn't been tampered"""
        # Re-calculate hash
        evidence_json = json.dumps(evidence.evidence_json, sort_keys=True, default=str)
        recalculated_hash = hashlib.sha256(evidence_json.encode()).hexdigest()
        
        # Compare with stored hash
        return recalculated_hash == evidence.evidence_hash


# Helper imports
import uuid
from sqlalchemy import select
from dataclasses import asdict