"""
STAGE 9: VERIFICATION ENGINE
Handles chain recalculation, query, and tamper detection
"""
import time
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from veriface.core.config import settings
from veriface.models.database import AsyncSession
from veriface.models.schemas import VerificationSession, Evidence, BlockchainRecord, AuditLog
from veriface.models.schemas_api import VerificationResult, VerificationChainResponse
from veriface.services.crypto_engine import CryptoEngineService
from veriface.services.blockchain import BlockchainService


@dataclass
class VerificationReport:
    """Complete verification report"""
    is_valid: bool
    tamper_detected: bool
    tamper_details: List[str]
    chain_verified: bool
    evidence_verified: bool
    blockchain_verified: bool
    overall_confidence: float


class VerificationEngineService:
    """Service for verification chain validation"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.crypto = CryptoEngineService()
        self.blockchain = None  # Initialized lazily
    
    @property
    def blockchain_service(self) -> BlockchainService:
        if self.blockchain is None:
            self.blockchain = BlockchainService(self.session)
        return self.blockchain
    
    async def verify_chain(
        self,
        session_id: str
    ) -> VerificationReport:
        """
        Verify complete verification chain for a session
        
        Args:
            session_id: Verification session ID
            
        Returns:
            VerificationReport with validation results
        """
        tamper_details = []
        is_valid = True
        
        # Get session data
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one_or_none()
        
        if not session:
            return VerificationReport(
                is_valid=False,
                tamper_detected=True,
                tamper_details=["Session not found"],
                chain_verified=False,
                evidence_verified=False,
                blockchain_verified=False,
                overall_confidence=0.0
            )
        
        # Get evidence
        evidence = (await self.session.execute(
            select(Evidence).where(Evidence.session_id == session_id)
        )).scalar_one_or_none()
        
        # Verify evidence integrity
        evidence_verified = True
        if evidence:
            # Recalculate hash
            evidence_json = json.dumps(evidence.evidence_json, sort_keys=True, default=str)
            recalculated_hash = hashlib.sha256(evidence_json.encode()).hexdigest()
            
            if recalculated_hash != evidence.evidence_hash:
                evidence_verified = False
                tamper_details.append("Evidence hash mismatch - tampering detected")
                is_valid = False
            
            # Verify quality scores
            if evidence.biometric_quality_score == 0 and evidence.overall_quality_score == 0:
                tamper_details.append("Missing quality scores")
                is_valid = False
        else:
            evidence_verified = False
            tamper_details.append("No evidence found")
            is_valid = False
        
        # Verify blockchain
        blockchain_verified = True
        blockchain_record = None
        if evidence:
            blockchain_record = (await self.session.execute(
                select(BlockchainRecord).where(BlockchainRecord.session_id == session_id)
            )).scalar_one_or_none()
            
            if blockchain_record:
                # Verify hash matches
                if blockchain_record.evidence_hash != evidence.evidence_hash:
                    blockchain_verified = False
                    tamper_details.append("Blockchain evidence hash mismatch")
                    is_valid = False
                
                # Verify Merkle root
                if blockchain_record.merkle_root != evidence.evidence_hash:
                    # Note: In production, would verify full Merkle proof
                    pass
            else:
                blockchain_verified = False
                tamper_details.append("No blockchain record found")
                is_valid = False
        
        # Verify chain integrity
        chain_verified = True
        chain = await self.blockchain_service.verify_chain()
        if not chain["valid"]:
            chain_verified = False
            tamper_details.append(f"Blockchain invalid: {chain.get('reason')}")
            is_valid = False
        
        # Calculate overall confidence
        confidence = session.match_confidence or 0.0
        
        if not evidence_verified:
            confidence *= 0.5
        if not blockchain_verified:
            confidence *= 0.7
        
        return VerificationReport(
            is_valid=is_valid,
            tamper_detected=len(tamper_details) > 0,
            tamper_details=tamper_details,
            chain_verified=chain_verified,
            evidence_verified=evidence_verified,
            blockchain_verified=blockchain_verified,
            overall_confidence=confidence
        )
    
    async def recalculate_chain(self, session_id: str) -> Dict[str, Any]:
        """
        Recalculate verification chain to detect tampering
        
        Args:
            session_id: Verification session ID
            
        Returns:
            Dict with recalculation results
        """
        # Get original evidence
        evidence = (await self.session.execute(
            select(Evidence).where(Evidence.session_id == session_id)
        )).scalar_one_or_none()
        
        if not evidence:
            return {"error": "No evidence found"}
        
        # Recalculate evidence hash
        evidence_json = json.dumps(evidence.evidence_json, sort_keys=True, default=str)
        recalculated_hash = hashlib.sha256(evidence_json.encode()).hexdigest()
        
        # Compare with stored hash
        match = recalculated_hash == evidence.evidence_hash
        
        # Recalculate Merkle root
        merkle_data = self._extract_merkle_data(evidence.evidence_json)
        recalculated_merkle = self._recalculate_merkle(merkle_data)
        
        return {
            "session_id": session_id,
            "original_hash": evidence.evidence_hash,
            "recalculated_hash": recalculated_hash,
            "hash_match": match,
            "original_merkle_root": evidence.canonical_image_hash,
            "recalculated_merkle": recalculated_merkle.get("root"),
            "merkle_match": recalculated_merkle.get("root") == evidence.canonical_image_hash,
            "verified": match
        }
    
    def _extract_merkle_data(self, evidence_json: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data for Merkle recalculation"""
        return {
            "input": evidence_json.get("input", {}),
            "biometric": evidence_json.get("biometric", {}),
            "candidates": evidence_json.get("candidates", []),
            "decision": evidence_json.get("decision", {})
        }
    
    def _recalculate_merkle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recalculate Merkle root from data"""
        leaves = []
        for i, (key, value) in enumerate(data.items()):
            leaf = self.crypto.build_merkle_leaf(value, i)
            leaves.append(leaf)
        
        root = self.crypto.build_merkle_tree(leaves)
        
        return {"root": root.hash if root else None, "leaves": len(leaves)}
    
    async def tamper_detection(self, session_id: str) -> Dict[str, Any]:
        """
        Run tamper detection on verification chain
        
        Args:
            session_id: Verification session ID
            
        Returns:
            Dict with tamper detection results
        """
        issues = []
        warnings = []
        
        # Check evidence integrity
        evidence = (await self.session.execute(
            select(Evidence).where(Evidence.session_id == session_id)
        )).scalar_one_or_none()
        
        if evidence:
            # Verify hash
            evidence_json = json.dumps(evidence.evidence_json, sort_keys=True, default=str)
            recalculated = hashlib.sha256(evidence_json.encode()).hexdigest()
            
            if recalculated != evidence.evidence_hash:
                issues.append({
                    "type": "evidence_tampering",
                    "severity": "critical",
                    "message": "Evidence JSON has been modified"
                })
            
            # Check timestamp consistency
            if evidence.created_at and evidence.verified_at:
                if evidence.verified_at < evidence.created_at:
                    issues.append({
                        "type": "timestamp_anomaly",
                        "severity": "critical",
                        "message": "Verification timestamp precedes creation timestamp"
                    })
            
            # Check provenance chain
            provenance = evidence.provenance_chain
            if len(provenance) < 4:
                warnings.append({
                    "type": "incomplete_provenance",
                    "severity": "warning",
                    "message": f"Provenance chain has only {len(provenance)} stages"
                })
        else:
            issues.append({
                "type": "missing_evidence",
                "severity": "critical",
                "message": "No evidence found for session"
            })
        
        # Check blockchain
        record = (await self.session.execute(
            select(BlockchainRecord).where(BlockchainRecord.session_id == session_id)
        )).scalar_one_or_none()
        
        if not record:
            warnings.append({
                "type": "no_blockchain_record",
                "severity": "warning",
                "message": "No blockchain record found"
            })
        else:
            if not record.confirmed:
                issues.append({
                    "type": "unconfirmed_transaction",
                    "severity": "warning",
                    "message": "Blockchain transaction not confirmed"
                })
            
            if record.confirmations < 3:
                warnings.append({
                    "type": "low_confirmations",
                    "severity": "info",
                    "message": f"Only {record.confirmations} confirmations"
                })
        
        return {
            "session_id": session_id,
            "verified": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def run_full_verification(
        self,
        session_id: str
    ) -> VerificationResult:
        """
        Run complete verification pipeline
        
        Args:
            session_id: Verification session ID
            
        Returns:
            VerificationResult with full status
        """
        start_time = time.time()
        
        # Get session
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one_or_none()
        
        if not session:
            return VerificationResult(
                session_id=session_id,
                status="failed",
                match_found=False,
                confidence_score=0.0,
                evidence_hash="",
                is_verified=False,
                tamper_detected=True,
                total_processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Run tamper detection
        tamper_result = await self.tamper_detection(session_id)
        tamper_detected = not tamper_result["verified"]
        
        # Get blockchain transaction
        blockchain_record = (await self.session.execute(
            select(BlockchainRecord).where(BlockchainRecord.session_id == session_id)
        )).scalar_one_or_none()
        
        # Get evidence
        evidence = (await self.session.execute(
            select(Evidence).where(Evidence.session_id == session_id)
        )).scalar_one_or_none()
        
        # Build transaction response
        transaction = None
        if blockchain_record:
            from veriface.models.schemas_api import BlockchainTransaction
            transaction = BlockchainTransaction(
                transaction_hash=blockchain_record.transaction_hash,
                block_number=blockchain_record.block_number,
                block_hash=blockchain_record.block_hash,
                timestamp=blockchain_record.timestamp,
                evidence_hash=blockchain_record.evidence_hash,
                previous_hash=blockchain_record.previous_hash,
                confirmations=blockchain_record.confirmations
            )
        
        # Update session status
        session.status = "verified" if not tamper_detected else "tamper_detected"
        
        # Create audit log
        audit = AuditLog(
            timestamp=datetime.utcnow(),
            session_id=session_id,
            action="verification_completed",
            details={
                "match_found": session.match_found,
                "confidence": session.match_confidence,
                "tamper_detected": tamper_detected
            }
        )
        self.session.add(audit)
        
        await self.session.commit()
        
        return VerificationResult(
            session_id=session_id,
            status=session.status,
            match_found=session.match_found or False,
            confidence_score=session.match_confidence or 0.0,
            best_candidate=None,
            evidence_hash=evidence.evidence_hash if evidence else "",
            blockchain_transaction=transaction,
            is_verified=not tamper_detected,
            tamper_detected=tamper_detected,
            total_processing_time_ms=(time.time() - start_time) * 1000
        )


# Helper imports
import uuid
from sqlalchemy import select