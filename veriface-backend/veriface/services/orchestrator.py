"""
VERIFACE CHAIN ORCHESTRATOR
Coordinates the complete verification pipeline through all 9 stages
"""
import time
import json
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

from veriface.core.config import settings
from veriface.models.database import AsyncSession
from veriface.models.schemas import VerificationSession, BiometricResult, Evidence
from veriface.models.schemas_api import VerificationChainResponse, VerificationStatus

from veriface.services.input_acquisition import InputAcquisitionService
from veriface.services.biometric_processing import BiometricProcessingService
from veriface.services.discovery_engine import DiscoveryEngineService
from veriface.services.candidate_intelligence import CandidateIntelligenceService
from veriface.services.confidence_engine import ConfidenceEngineService
from veriface.services.evidence_vault import EvidenceVaultService
from veriface.services.crypto_engine import CryptoEngineService
from veriface.services.blockchain import BlockchainService
from veriface.services.verification_engine import VerificationEngineService


class VerificationOrchestrator:
    """
    Orchestrates the complete VERIFACE CHAIN verification pipeline
    
    Coordinates all 9 stages:
    1. Input Acquisition
    2. Biometric Processing
    3. Discovery Engine
    4. Candidate Intelligence
    5. Confidence Engine
    6. Evidence Vault
    7. Crypto Engine
    8. Blockchain
    9. Verification
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.stages_completed: List[str] = []
        
        # Initialize services
        self.input_service = InputAcquisitionService(session)
        self.biometric_service = BiometricProcessingService(session)
        self.discovery_service = DiscoveryEngineService(session)
        self.candidate_service = CandidateIntelligenceService(session)
        self.confidence_service = ConfidenceEngineService(session)
        self.evidence_service = EvidenceVaultService(session)
        self.crypto_service = CryptoEngineService()
        self.blockchain_service = BlockchainService(session)
        self.verification_service = VerificationEngineService(session)
    
    async def run_full_pipeline(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Execute full verification pipeline
        
        Args:
            session_id: Verification session ID
            
        Returns:
            Dict with results from all stages
        """
        start_time = time.time()
        results = {"session_id": session_id, "stages": {}, "errors": []}
        
        try:
            # ===== STAGE 1: Input Acquisition =====
            stage_start = time.time()
            input_result = await self._get_input_result(session_id)
            results["stages"]["input_acquisition"] = {
                "status": "completed",
                "duration_ms": (time.time() - stage_start) * 1000,
                "data": input_result
            }
            
            # ===== STAGE 2: Biometric Processing =====
            stage_start = time.time()
            if input_result:
                biometric_result = await self._get_biometric_result(session_id)
                results["stages"]["biometric_processing"] = {
                    "status": "completed",
                    "duration_ms": (time.time() - stage_start) * 1000,
                    "data": biometric_result
                }
            else:
                results["errors"].append("No input image found")
            
            # ===== STAGE 3: Discovery Engine =====
            stage_start = time.time()
            if biometric_result and biometric_result.get("face_detected"):
                discovery_result = await self._run_discovery(session_id)
                results["stages"]["discovery_engine"] = {
                    "status": "completed",
                    "duration_ms": (time.time() - stage_start) * 1000,
                    "data": discovery_result
                }
            else:
                results["stages"]["discovery_engine"] = {
                    "status": "skipped",
                    "reason": "No face detected"
                }
            
            # ===== STAGE 4: Candidate Intelligence =====
            stage_start = time.time()
            if biometric_result and biometric_result.get("embedding"):
                candidate_result = await self._run_candidate_analysis(session_id)
                results["stages"]["candidate_intelligence"] = {
                    "status": "completed",
                    "duration_ms": (time.time() - stage_start) * 1000,
                    "data": candidate_result
                }
            else:
                results["stages"]["candidate_intelligence"] = {
                    "status": "skipped",
                    "reason": "No embedding available"
                }
            
            # ===== STAGE 5: Confidence Engine =====
            stage_start = time.time()
            confidence_result = await self._compute_confidence(session_id)
            results["stages"]["confidence_engine"] = {
                "status": "completed",
                "duration_ms": (time.time() - stage_start) * 1000,
                "data": confidence_result
            }
            
            # ===== STAGE 6: Evidence Vault =====
            stage_start = time.time()
            evidence_result = await self._create_evidence(session_id)
            results["stages"]["evidence_vault"] = {
                "status": "completed",
                "duration_ms": (time.time() - stage_start) * 1000,
                "data": evidence_result
            }
            
            # ===== STAGE 7: Crypto Engine =====
            stage_start = time.time()
            crypto_result = await self._run_crypto(evidence_result)
            results["stages"]["crypto_engine"] = {
                "status": "completed",
                "duration_ms": (time.time() - stage_start) * 1000,
                "data": crypto_result
            }
            
            # ===== STAGE 8: Blockchain =====
            stage_start = time.time()
            if evidence_result:
                blockchain_result = await self._record_blockchain(
                    session_id,
                    crypto_result.get("sha256_hash"),
                    crypto_result.get("merkle_root")
                )
                results["stages"]["blockchain"] = {
                    "status": "completed",
                    "duration_ms": (time.time() - stage_start) * 1000,
                    "data": blockchain_result
                }
            else:
                results["stages"]["blockchain"] = {
                    "status": "skipped",
                    "reason": "No evidence created"
                }
            
            # ===== STAGE 9: Verification =====
            stage_start = time.time()
            verification_result = await self.verification_service.run_full_verification(session_id)
            results["stages"]["verification"] = {
                "status": "completed",
                "duration_ms": (time.time() - stage_start) * 1000,
                "data": {
                    "match_found": verification_result.match_found,
                    "confidence_score": verification_result.confidence_score,
                    "is_verified": verification_result.is_verified,
                    "tamper_detected": verification_result.tamper_detected
                }
            }
            
            results["total_duration_ms"] = (time.time() - start_time) * 1000
            results["complete"] = True
            
            # Update session
            await self._update_session_final(session_id, verification_result)
            
        except Exception as e:
            results["errors"].append(str(e))
            results["complete"] = False
            
            # Update session with error
            await self._update_session_error(session_id, str(e))
        
        return results
    
    async def _get_input_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get input acquisition result"""
        from veriface.models.schemas import InputImage
        result = await self.session.execute(
            select(InputImage).where(InputImage.session_id == session_id)
        )
        img = result.scalar_one_or_none()
        if img:
            return {
                "id": img.id,
                "filename": img.filename,
                "image_hash": img.image_hash
            }
        return None
    
    async def _get_biometric_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get or run biometric processing"""
        from veriface.models.schemas import InputImage, BiometricResult
        
        # Check if already processed
        result = await self.session.execute(
            select(BiometricResult).where(BiometricResult.session_id == session_id)
        )
        bio = result.scalar_one_or_none()
        
        if bio:
            embedding = None
            if bio.face_embedding:
                import numpy as np
                embedding = np.frombuffer(bio.face_embedding, dtype=np.float64).tolist()
            
            return {
                "id": bio.id,
                "face_detected": bio.face_detected,
                "quality_score": bio.quality_score,
                "embedding_size": bio.embedding_size,
                "embedding": embedding
            }
        
        # Run biometric processing
        input_result = await self._get_input_result(session_id)
        if not input_result:
            return None
        
        input_img = await self.session.execute(
            select(InputImage).where(InputImage.session_id == session_id)
        )
        input_image = input_result  # Already have the dict
        
        result = await self.biometric_service.process_biometrics(
            session_id,
            input_image.get("file_path", "")
        )
        
        return {
            "id": result.biometric_id,
            "face_detected": result.face_detected,
            "quality_score": result.quality_score,
            "embedding_size": result.embedding_size,
            "embedding": []
        }
    
    async def _run_discovery(self, session_id: str) -> Dict[str, Any]:
        """Run candidate discovery"""
        # Get biometric result
        bio_result = await self._get_biometric_result(session_id)
        
        if not bio_result or not bio_result.get("embedding"):
            return {"candidates_found": 0}
        
        embedding = bio_result["embedding"]
        image_hash = bio_result.get("image_hash", "")
        
        from veriface.models.schemas_api import DiscoveryRequest
        request = DiscoveryRequest(max_candidates=50)
        
        result = await self.discovery_service.discover_candidates(
            session_id, embedding, image_hash, request
        )
        
        return {
            "candidates_found": result.candidates_found,
            "providers_searched": result.providers_searched
        }
    
    async def _run_candidate_analysis(self, session_id: str) -> Dict[str, Any]:
        """Run candidate intelligence analysis"""
        bio_result = await self._get_biometric_result(session_id)
        
        if not bio_result or not bio_result.get("embedding"):
            return {"analyzed_candidates": 0}
        
        embedding = bio_result["embedding"]
        import numpy as np
        embedding_array = np.array(embedding)
        
        result = await self.candidate_service.analyze_all_candidates(
            session_id, embedding_array
        )
        
        return {
            "total_candidates": result.total_candidates,
            "analyzed_candidates": result.analyzed_candidates,
            "best_match_id": result.best_match.candidate_id if result.best_match else None
        }
    
    async def _compute_confidence(self, session_id: str) -> Dict[str, Any]:
        """Compute confidence scores"""
        result = await self.confidence_service.compute_confidence(session_id)
        
        return {
            "match_found": result.match_found,
            "confidence_score": result.confidence_score,
            "best_candidate_id": result.best_candidate_id
        }
    
    async def _create_evidence(self, session_id: str) -> Dict[str, Any]:
        """Create evidence record"""
        result = await self.evidence_service.create_evidence(session_id)
        
        return {
            "evidence_id": result.evidence_id,
            "evidence_hash": result.evidence_hash
        }
    
    async def _run_crypto(self, evidence_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process crypto operations"""
        if not evidence_result:
            return {}
        
        # Get evidence JSON from database
        evidence = await self.evidence_service.get_evidence(
            evidence_result.get("session_id") or ""
        )
        
        if not evidence:
            return {}
        
        result = await self.crypto_service.process_evidence(
            evidence.evidence_json,
            evidence.canonical_image_hash
        )
        
        return {
            "sha256_hash": result["sha256_hash"],
            "merkle_root": result["merkle_root"]
        }
    
    async def _record_blockchain(
        self,
        session_id: str,
        evidence_hash: str,
        merkle_root: str
    ) -> Dict[str, Any]:
        """Record on blockchain"""
        if not evidence_hash:
            return {"recorded": False}
        
        result = await self.blockchain_service.record_evidence(
            session_id, evidence_hash, merkle_root or ""
        )
        
        return {
            "recorded": result.recorded,
            "transaction_hash": result.transaction.transaction_hash,
            "block_number": result.transaction.block_number
        }
    
    async def _update_session_final(
        self,
        session_id: str,
        verification_result: Any
    ):
        """Update session with final status"""
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one()
        
        if session:
            session.completed_at = datetime.utcnow()
            session.match_found = verification_result.match_found
            session.match_confidence = verification_result.confidence_score
        
        await self.session.commit()
    
    async def _update_session_error(self, session_id: str, error: str):
        """Update session with error status"""
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one()
        
        if session:
            session.status = "failed"
            session.completed_at = datetime.utcnow()
        
        await self.session.commit()
    
    async def get_pipeline_status(self, session_id: str) -> Dict[str, Any]:
        """Get current pipeline status"""
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one_or_none()
        
        if not session:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "status": session.status,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "match_found": session.match_found,
            "match_confidence": session.match_confidence
        }


# Helper imports
from sqlalchemy import select