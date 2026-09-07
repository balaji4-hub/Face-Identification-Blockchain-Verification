"""
STAGE 9: VERIFICATION ENGINE
Handles chain recalculation, query, and tamper detection
Implements full verification and re-verification capabilities
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
    re_verification: bool = False


class VerificationEngineService:
    """
    Service for verification chain validation
    Supports both initial verification and re-verification
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.crypto = CryptoEngineService()
        self.blockchain = None
    
    @property
    def blockchain_service(self) -> BlockchainService:
        if self.blockchain is None:
            self.blockchain = BlockchainService(self.session)
        return self.blockchain
    
    async def verify_chain(
        self,
        session_id: str,
        re_verification: bool = False
    ) -> VerificationReport:
        """
        Verify complete verification chain for a session
        
        Args:
            session_id: Verification session ID
            re_verification: If True, this is a re-verification
            
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
                overall_confidence=0.0,
                re_verification=re_verification
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
            
            # Verify provenance chain
            if evidence.provenance_chain:
                if len(evidence.provenance_chain) < 4:
                    tamper_details.append("Incomplete provenance chain")
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
                if blockchain_record.merkle_root:
                    # Merkle root should match canonical image hash
                    if blockchain_record.merkle_root != evidence.canonical_image_hash:
                        tamper_details.append("Merkle root mismatch")
                
                # Verify block exists
                block_info = await self.blockchain_service.get_block_info(blockchain_record.block_number)
                if block_info:
                    if block_info.evidence_hash != evidence.evidence_hash:
                        blockchain_verified = False
                        tamper_details.append("Block evidence hash mismatch")
                else:
                    blockchain_verified = False
                    tamper_details.append("Block not found in chain")
            else:
                blockchain_verified = False
                tamper_details.append("No blockchain record found")
        
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
            overall_confidence=confidence,
            re_verification=re_verification
        )
    
    async def recalculate_chain(self, session_id: str) -> Dict[str, Any]:
        """
        Recalculate verification chain to detect tampering
        Used for re-verification
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
        hash_match = recalculated_hash == evidence.evidence_hash
        
        # Recalculate Merkle root
        merkle_data = self._extract_merkle_data(evidence.evidence_json)
        recalculated_merkle = self._recalculate_merkle(merkle_data)
        
        # Get blockchain verification
        blockchain_record = (await self.session.execute(
            select(BlockchainRecord).where(BlockchainRecord.session_id == session_id)
        )).scalar_one_or_none()
        
        blockchain_valid = False
        if blockchain_record:
            # Verify the block
            block_info = await self.blockchain_service.get_block_info(
                blockchain_record.block_number
            )
            if block_info:
                target_prefix = '0' * settings.difficulty
                hash_valid = block_info.hash.startswith(target_prefix)
                prev_valid = block_info.evidence_hash == evidence.evidence_hash
                blockchain_valid = hash_valid and prev_valid
        
        return {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "original_hash": evidence.evidence_hash,
            "recalculated_hash": recalculated_hash,
            "hash_match": hash_match,
            "original_merkle_root": evidence.canonical_image_hash,
            "recalculated_merkle": recalculated_merkle.get("root"),
            "merkle_match": recalculated_merkle.get("root") == evidence.canonical_image_hash,
            "blockchain_valid": blockchain_valid,
            "verified": hash_match and blockchain_valid,
            "re_verification": True
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
        """Run comprehensive tamper detection"""
        issues = []
        warnings = []
        
        # Get session data
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one_or_none()
        
        if not session:
            return {
                "session_id": session_id,
                "verified": False,
                "issues": [{"type": "not_found", "severity": "critical", "message": "Session not found"}],
                "warnings": [],
                "timestamp": datetime.utcnow().isoformat()
            }
        
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
                    "message": "Evidence JSON has been modified",
                    "original_hash": evidence.evidence_hash,
                    "recalculated_hash": recalculated
                })
            
            # Verify timestamp consistency
            if evidence.created_at and evidence.verified_at:
                if evidence.verified_at < evidence.created_at:
                    issues.append({
                        "type": "timestamp_anomaly",
                        "severity": "critical",
                        "message": "Verification timestamp precedes creation timestamp"
                    })
            
            # Verify quality scores are reasonable
            if evidence.overall_quality_score == 0:
                warnings.append({
                    "type": "missing_quality_score",
                    "severity": "warning",
                    "message": "Missing overall quality score"
                })
            
            # Check provenance chain
            provenance = evidence.provenance_chain
            if provenance:
                expected_stages = ["INPUT_ACQUISITION", "BIOMETRIC_PROCESSING", 
                                  "CANDIDATE_INTELLIGENCE", "CONFIDENCE_ENGINE"]
                found_stages = [p.get("stage") for p in provenance]
                
                for expected in expected_stages:
                    if expected not in found_stages:
                        warnings.append({
                            "type": "incomplete_provenance",
                            "severity": "warning",
                            "message": f"Missing stage: {expected}"
                        })
            else:
                issues.append({
                    "type": "missing_provenance",
                    "severity": "critical",
                    "message": "No provenance chain found"
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
                "message": "No blockchain record found - evidence not timestamped"
            })
        else:
            # Verify block
            block = await self.blockchain_service.get_block_info(record.block_number)
            if block:
                target_prefix = '0' * settings.difficulty
                if not block.hash.startswith(target_prefix):
                    issues.append({
                        "type": "invalid_proof_of_work",
                        "severity": "critical",
                        "message": "Block does not meet proof of work requirement"
                    })
                
                if block.evidence_hash != evidence.evidence_hash if evidence else False:
                    issues.append({
                        "type": "blockchain_mismatch",
                        "severity": "critical",
                        "message": "Block evidence hash does not match"
                    })
            else:
                issues.append({
                    "type": "block_not_found",
                    "severity": "critical",
                    "message": f"Block {record.block_number} not found in chain"
                })
            
            # Check confirmations
            if record.confirmations < 3:
                warnings.append({
                    "type": "low_confirmations",
                    "severity": "info",
                    "message": f"Only {record.confirmations} confirmations (recommended: 3+)"
                })
        
        return {
            "session_id": session_id,
            "verified": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "timestamp": datetime.utcnow().isoformat(),
            "re_verification": True
        }
    
    async def run_full_verification(
        self,
        session_id: str,
        re_verification: bool = False
    ) -> VerificationResult:
        """
        Run complete verification pipeline
        
        Args:
            session_id: Verification session ID
            re_verification: If True, indicates re-verification
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
        if re_verification:
            status = "re_verified" if not tamper_detected else "tamper_detected"
        else:
            status = "verified" if not tamper_detected else "tamper_detected"
        
        session.status = status
        
        # Create audit log
        audit = AuditLog(
            timestamp=datetime.utcnow(),
            session_id=session_id,
            action="verification_completed",
            details={
                "match_found": session.match_found,
                "confidence": session.match_confidence,
                "tamper_detected": tamper_detected,
                "re_verification": re_verification
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
    
    async def re_verify_session(self, session_id: str) -> Dict[str, Any]:
        """
        Re-verify an existing session
        Used for auditing and compliance
        """
        # Run recalculation
        recalc_result = await self.recalculate_chain(session_id)
        
        # Run tamper detection
        tamper_result = await self.tamper_detection(session_id)
        
        # Run full verification
        verification_result = await self.run_full_verification(session_id, re_verification=True)
        
        return {
            "session_id": session_id,
            "re_verification": True,
            "timestamp": datetime.utcnow().isoformat(),
            "recalculation": recalc_result,
            "tamper_detection": tamper_result,
            "verification": {
                "status": verification_result.status,
                "is_verified": verification_result.is_verified,
                "tamper_detected": verification_result.tamper_detected,
                "confidence_score": verification_result.confidence_score
            }
        }


# Helper imports
import uuid
from sqlalchemy import select