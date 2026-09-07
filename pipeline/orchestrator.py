"""
VeriFace Chain - Central Pipeline Orchestrator
Coordinates the complete end-to-end execution across all 10 stages:
Input -> Vision Intelligence -> Visual Search -> Candidate Intelligence ->
Confidence Scoring -> Evidence Provenance -> Canonical Hashing -> Blockchain Registration
"""
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

from config.settings import settings
from core.exceptions import (
    InputValidationError,
    InvalidImageError,
    FaceDetectionError,
    FaceQualityError,
    VeriFaceError
)
from core.logging import get_logger
from core.result_types import StageStatus, MatchDecision
from pipeline.stages import PipelineStage, StageExecution

from intelligence.face_detector import FaceDetector
from intelligence.face_quality import FaceQualityAnalyzer
from intelligence.face_encoder import FaceEncoder
from intelligence.face_matcher import FaceMatcher
from intelligence.candidate_ranker import CandidateRanker

from discovery.candidate_fetcher import CandidateFetcher
from discovery.search_service import SearchService, SearchAuditEntry

from evidence.evidence_models import (
    FaceDetection,
    FaceQualityResult,
    CandidateResult,
    CandidateAnalysis,
    EvidenceRecord,
    BlockchainReceipt
)
from evidence.evidence_builder import EvidenceBuilder
from evidence.hasher import hash_evidence, hash_bytes
from evidence.provenance import ProvenanceChain

from trust.blockchain_client import BlockchainClient
from trust.audit_service import AuditService

logger = get_logger("pipeline.orchestrator")


class PipelineDiscoveryResult(BaseModel):
    """Unified output contract produced by the complete verification pipeline."""
    pipeline_id: str
    status: str
    input_image_path: str
    input_image_sha256: str
    face_detection: Optional[FaceDetection] = None
    all_faces_detected: int = 0
    face_quality: Optional[FaceQualityResult] = None
    search_audit: Optional[SearchAuditEntry] = None
    candidates: List[CandidateAnalysis] = Field(default_factory=list)
    best_match: Optional[CandidateAnalysis] = None
    evidence_record: Optional[EvidenceRecord] = None
    evidence_hash: Optional[str] = None
    blockchain_receipt: Optional[BlockchainReceipt] = None
    provenance: Optional[ProvenanceChain] = None
    stages: Dict[str, StageExecution] = Field(default_factory=dict)
    total_duration_ms: float = 0.0


class PipelineOrchestrator:
    """Autonomous coordinator for the VeriFace Chain pipeline."""
    
    def __init__(
        self,
        search_provider: str = None,
        blockchain_client: BlockchainClient = None,
        audit_service: AuditService = None
    ):
        self.search_provider = search_provider or settings.DEFAULT_SEARCH_PROVIDER
        self.detector = FaceDetector()
        self.quality_analyzer = FaceQualityAnalyzer()
        self.encoder = FaceEncoder()
        self.matcher = FaceMatcher()
        self.search_service = SearchService()
        self.fetcher = CandidateFetcher()
        self.blockchain_client = blockchain_client or BlockchainClient()
        self.audit_service = audit_service or AuditService()

    def run_pipeline(
        self,
        image_path: Union[Path, str],
        register_on_chain: bool = True,
        search_mode: str = "web_discovery",
        provider_name: Optional[str] = None
    ) -> PipelineDiscoveryResult:
        """
        Executes the complete VeriFace Chain pipeline from input acquisition to blockchain notarization.
        
        Args:
            image_path: Path to the consented input face image.
            register_on_chain: If True, registers evidence hash onto the blockchain.
            search_mode: "web_discovery" (simulate finding public web appearances) or "private_unindexed" (no match test).
            provider_name: Specific visual search provider (e.g. "mock", "serpapi", "authorized_api").
        """
        pipeline_id = f"vfc_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        
        stages: Dict[str, StageExecution] = {
            stage.value: StageExecution(stage=stage, status=StageStatus.PENDING)
            for stage in PipelineStage
        }
        provenance = ProvenanceChain(pipeline_id=pipeline_id)
        
        logger.info(f"=== Starting VeriFace Chain Pipeline [{pipeline_id}] ===")
        
        input_path = Path(image_path)
        input_sha256 = ""
        face_detection = None
        face_quality = None
        analyzed_candidates: List[CandidateAnalysis] = []
        best_match = None
        evidence_record = None
        evidence_hash = None
        blockchain_receipt = None

        # =========================================================================
        # STAGE 1: INPUT VALIDATION
        # =========================================================================
        stage_key = PipelineStage.INPUT_VALIDATION.value
        stages[stage_key].status = StageStatus.RUNNING
        s_start = time.time()
        try:
            if not input_path.exists():
                raise InputValidationError(f"Input file does not exist: {input_path}")
                
            raw_bytes = input_path.read_bytes()
            if len(raw_bytes) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise InputValidationError(f"Input file exceeds {settings.MAX_FILE_SIZE_MB}MB size limit")
                
            input_sha256 = hash_bytes(raw_bytes)
            image_rgb = self.detector.load_image(raw_bytes)
            
            h, w = image_rgb.shape[:2]
            if w < settings.MIN_IMAGE_WIDTH or h < settings.MIN_IMAGE_HEIGHT:
                raise InputValidationError(
                    f"Image resolution ({w}x{h}) is below minimum requirement ({settings.MIN_IMAGE_WIDTH}x{settings.MIN_IMAGE_HEIGHT})"
                )
                
            stages[stage_key].status = StageStatus.SUCCESS
            stages[stage_key].duration_ms = round((time.time() - s_start) * 1000, 2)
            stages[stage_key].details = {"dimensions": f"{w}x{h}", "sha256": input_sha256}
            provenance.add_step("INPUT_VALIDATION", None, input_sha256, {"dimensions": f"{w}x{h}"})
            logger.info(f"[Stage 1] Input validated: SHA-256={input_sha256[:12]}...")
        except Exception as e:
            stages[stage_key].status = StageStatus.FAILED
            stages[stage_key].error = str(e)
            return self._finalize_result(pipeline_id, "FAILED", input_path, input_sha256, stages, provenance, start_time)

        # =========================================================================
        # STAGE 2: FACE DETECTION
        # =========================================================================
        stage_key = PipelineStage.FACE_DETECTION.value
        stages[stage_key].status = StageStatus.RUNNING
        s_start = time.time()
        try:
            detections = self.detector.detect(image_rgb)
            if not detections:
                raise FaceDetectionError("No faces detected in the provided input image")
                
            # Sort faces by area to isolate the primary foreground subject
            detections_sorted = sorted(
                detections,
                key=lambda d: (d.bounding_box[2] - d.bounding_box[0]) * (d.bounding_box[1] - d.bounding_box[3]),
                reverse=True
            )
            face_detection = detections_sorted[0]
            
            stages[stage_key].status = StageStatus.SUCCESS
            stages[stage_key].duration_ms = round((time.time() - s_start) * 1000, 2)
            stages[stage_key].details = {"faces_detected": len(detections), "confidence": face_detection.confidence}
            provenance.add_step("FACE_DETECTION", input_sha256, None, {"faces_count": len(detections)})
            logger.info(f"[Stage 2] Face detected: Total={len(detections)}, Primary Confidence={face_detection.confidence}")
        except Exception as e:
            stages[stage_key].status = StageStatus.FAILED
            stages[stage_key].error = str(e)
            return self._finalize_result(pipeline_id, "FAILED", input_path, input_sha256, stages, provenance, start_time)

        # =========================================================================
        # STAGE 3: FACE QUALITY ANALYSIS
        # =========================================================================
        stage_key = PipelineStage.FACE_QUALITY.value
        stages[stage_key].status = StageStatus.RUNNING
        s_start = time.time()
        try:
            face_quality = self.quality_analyzer.evaluate(image_rgb, face_detection)
            stages[stage_key].status = StageStatus.SUCCESS
            stages[stage_key].duration_ms = round((time.time() - s_start) * 1000, 2)
            stages[stage_key].details = {
                "quality_score": face_quality.quality_score,
                "accepted": face_quality.accepted,
                "issues": face_quality.issues
            }
            provenance.add_step("FACE_QUALITY", None, None, {"quality_score": face_quality.quality_score, "accepted": face_quality.accepted})
            logger.info(f"[Stage 3] Face quality: Score={face_quality.quality_score}, Accepted={face_quality.accepted}")
        except Exception as e:
            stages[stage_key].status = StageStatus.FAILED
            stages[stage_key].error = str(e)
            return self._finalize_result(pipeline_id, "FAILED", input_path, input_sha256, stages, provenance, start_time)

        # =========================================================================
        # STAGE 4: BIOMETRIC FACE EMBEDDING
        # =========================================================================
        stage_key = PipelineStage.FACE_EMBEDDING.value
        stages[stage_key].status = StageStatus.RUNNING
        s_start = time.time()
        try:
            # Query embedding vector - strictly transient in volatile memory!
            query_embedding = self.encoder.encode(image_rgb, face_detection)
            stages[stage_key].status = StageStatus.SUCCESS
            stages[stage_key].duration_ms = round((time.time() - s_start) * 1000, 2)
            stages[stage_key].details = {"dimension": query_embedding.dimension, "normalized": query_embedding.normalized}
            provenance.add_step("FACE_EMBEDDING", None, None, {"dimension": query_embedding.dimension})
            logger.info(f"[Stage 4] Generated {query_embedding.dimension}-d normalized biometric embedding in-memory")
        except Exception as e:
            stages[stage_key].status = StageStatus.FAILED
            stages[stage_key].error = str(e)
            return self._finalize_result(pipeline_id, "FAILED", input_path, input_sha256, stages, provenance, start_time)

        # =========================================================================
        # STAGE 5: VISUAL SEARCH DISCOVERY
        # =========================================================================
        stage_key = PipelineStage.VISUAL_SEARCH.value
        stages[stage_key].status = StageStatus.RUNNING
        s_start = time.time()
        provider_key = provider_name or self.search_provider
        try:
            raw_candidates = self.search_service.search(input_path, provider_key, search_mode=search_mode)
            stages[stage_key].status = StageStatus.SUCCESS
            stages[stage_key].duration_ms = round((time.time() - s_start) * 1000, 2)
            stages[stage_key].details = {"candidates_found": len(raw_candidates), "provider": provider_key}
            provenance.add_step("VISUAL_SEARCH", None, None, {"provider": provider_key, "count": len(raw_candidates)})
            logger.info(f"[Stage 5] Visual search returned {len(raw_candidates)} candidate match targets via '{provider_key}'")

        except Exception as e:
            stages[stage_key].status = StageStatus.FAILED
            stages[stage_key].error = str(e)
            return self._finalize_result(pipeline_id, "FAILED", input_path, input_sha256, stages, provenance, start_time)

        # =========================================================================
        # STAGE 6: CANDIDATE ANALYSIS & MULTI-FACE COMPARISON
        # =========================================================================
        stage_key = PipelineStage.CANDIDATE_ANALYSIS.value
        stages[stage_key].status = StageStatus.RUNNING
        s_start = time.time()
        try:
            for idx, cand in enumerate(raw_candidates):
                cand_id = f"cand_{idx+1}"
                try:
                    # 1. Fetch & hash candidate image
                    cand_bytes, cand_sha256 = self.fetcher.fetch(cand.image_url)
                    cand_rgb = self.detector.load_image(cand_bytes)
                    
                    # 2. Detect all faces in candidate image
                    cand_faces = self.detector.detect(cand_rgb)
                    
                    if not cand_faces:
                        analyzed_candidates.append(
                            CandidateAnalysis(
                                candidate_id=cand_id,
                                source=cand.source,
                                page_url=cand.page_url,
                                image_url=cand.image_url,
                                image_sha256=cand_sha256,
                                faces_detected=0,
                                highest_similarity=0.0,
                                match_status=MatchDecision.NO_MATCH,
                                error="No face detected in candidate image"
                            )
                        )
                        continue
                        
                    # 3. Generate embeddings for all detected faces in candidate
                    cand_embeddings = [
                        self.encoder.encode(cand_rgb, f).embedding
                        for f in cand_faces
                    ]
                    
                    # 4. Compare all candidate faces against query embedding
                    sim, best_idx, decision = self.matcher.compare_one_to_many(
                        query_embedding.embedding,
                        cand_embeddings
                    )
                    
                    analyzed_candidates.append(
                        CandidateAnalysis(
                            candidate_id=cand_id,
                            source=cand.source,
                            page_url=cand.page_url,
                            image_url=cand.image_url,
                            image_sha256=cand_sha256,
                            faces_detected=len(cand_faces),
                            highest_similarity=sim,
                            best_face_index=best_idx,
                            match_status=decision
                        )
                    )
                except Exception as cand_err:
                    logger.warning(f"Error analyzing candidate {cand.image_url}: {cand_err}")
                    analyzed_candidates.append(
                        CandidateAnalysis(
                            candidate_id=cand_id,
                            source=cand.source,
                            page_url=cand.page_url,
                            image_url=cand.image_url,
                            match_status=MatchDecision.NO_MATCH,
                            error=str(cand_err)
                        )
                    )
                    
            stages[stage_key].status = StageStatus.SUCCESS
            stages[stage_key].duration_ms = round((time.time() - s_start) * 1000, 2)
            stages[stage_key].details = {"analyzed_count": len(analyzed_candidates)}
            provenance.add_step("CANDIDATE_ANALYSIS", None, None, {"analyzed_count": len(analyzed_candidates)})
            logger.info(f"[Stage 6] Completed multi-face candidate intelligence on {len(analyzed_candidates)} images")
        except Exception as e:
            stages[stage_key].status = StageStatus.FAILED
            stages[stage_key].error = str(e)
            return self._finalize_result(pipeline_id, "FAILED", input_path, input_sha256, stages, provenance, start_time)

        # =========================================================================
        # STAGE 7: CONFIDENCE CLASSIFICATION & RANKING
        # =========================================================================
        stage_key = PipelineStage.MATCH_DECISION.value
        stages[stage_key].status = StageStatus.RUNNING
        s_start = time.time()
        try:
            ranked_candidates = CandidateRanker.rank(analyzed_candidates)
            best_match = ranked_candidates[0] if ranked_candidates else None
            
            stages[stage_key].status = StageStatus.SUCCESS
            stages[stage_key].duration_ms = round((time.time() - s_start) * 1000, 2)
            stages[stage_key].details = {
                "top_similarity": best_match.highest_similarity if best_match else 0.0,
                "top_decision": best_match.match_status.value if best_match else "NO_MATCH"
            }
            provenance.add_step(
                "MATCH_DECISION",
                None,
                None,
                {"top_similarity": best_match.highest_similarity if best_match else 0.0}
            )
            logger.info(f"[Stage 7] Match decision: Top Sim={best_match.highest_similarity if best_match else 0.0} ({best_match.match_status.value if best_match else 'NONE'})")
        except Exception as e:
            stages[stage_key].status = StageStatus.FAILED
            stages[stage_key].error = str(e)
            return self._finalize_result(pipeline_id, "FAILED", input_path, input_sha256, stages, provenance, start_time)

        # =========================================================================
        # STAGE 8: EVIDENCE PROVENANCE ASSEMBLY
        # =========================================================================
        stage_key = PipelineStage.EVIDENCE_GENERATION.value
        stages[stage_key].status = StageStatus.RUNNING
        s_start = time.time()
        try:
            if best_match and best_match.highest_similarity >= settings.FACE_MATCH_THRESHOLD:
                evidence_record = EvidenceBuilder.build_evidence(
                    candidate=best_match,
                    input_image_sha256=input_sha256,
                    threshold=settings.FACE_MATCH_THRESHOLD,
                    model_name=settings.FACE_RECOGNITION_MODEL
                )
                stages[stage_key].status = StageStatus.SUCCESS
                stages[stage_key].duration_ms = round((time.time() - s_start) * 1000, 2)
                stages[stage_key].details = {"evidence_id": evidence_record.evidence_id}
                provenance.add_step("EVIDENCE_GENERATION", None, evidence_record.evidence_id, {"source": evidence_record.source})
                logger.info(f"[Stage 8] Evidence provenance assembled: ID={evidence_record.evidence_id}")
            else:
                stages[stage_key].status = StageStatus.SKIPPED
                stages[stage_key].details = {"reason": "No candidate exceeded confidence threshold"}
                logger.info("[Stage 8] Evidence generation skipped (similarity below match threshold)")
        except Exception as e:
            stages[stage_key].status = StageStatus.FAILED
            stages[stage_key].error = str(e)
            return self._finalize_result(pipeline_id, "FAILED", input_path, input_sha256, stages, provenance, start_time)

        # =========================================================================
        # STAGE 9: CANONICALIZATION & SHA-256 HASHING
        # =========================================================================
        stage_key = PipelineStage.CANONICAL_HASHING.value
        if evidence_record is not None:
            stages[stage_key].status = StageStatus.RUNNING
            s_start = time.time()
            try:
                evidence_hash = hash_evidence(evidence_record)
                stages[stage_key].status = StageStatus.SUCCESS
                stages[stage_key].duration_ms = round((time.time() - s_start) * 1000, 2)
                stages[stage_key].details = {"sha256": evidence_hash}
                provenance.add_step("CANONICAL_HASHING", None, evidence_hash)
                logger.info(f"[Stage 9] Canonical SHA-256 evidence fingerprint: {evidence_hash}")
            except Exception as e:
                stages[stage_key].status = StageStatus.FAILED
                stages[stage_key].error = str(e)
                return self._finalize_result(pipeline_id, "FAILED", input_path, input_sha256, stages, provenance, start_time)
        else:
            stages[stage_key].status = StageStatus.SKIPPED

        # =========================================================================
        # STAGE 10: BLOCKCHAIN REGISTRATION
        # =========================================================================
        stage_key = PipelineStage.BLOCKCHAIN_REGISTRATION.value
        if evidence_hash is not None and register_on_chain:
            stages[stage_key].status = StageStatus.RUNNING
            s_start = time.time()
            try:
                blockchain_receipt = self.blockchain_client.register_evidence(evidence_hash)
                stages[stage_key].status = StageStatus.SUCCESS
                stages[stage_key].duration_ms = round((time.time() - s_start) * 1000, 2)
                stages[stage_key].details = {
                    "tx_hash": blockchain_receipt.transaction_hash,
                    "block_number": blockchain_receipt.block_number
                }
                provenance.add_step("BLOCKCHAIN_REGISTRATION", evidence_hash, blockchain_receipt.transaction_hash)
                
                # Record to audit log
                self.audit_service.log_event(
                    operation="EVIDENCE_NOTARIZED",
                    pipeline_id=pipeline_id,
                    evidence_id=evidence_record.evidence_id if evidence_record else None,
                    blockchain_tx=blockchain_receipt.transaction_hash,
                    verification_status="NOTARIZED",
                    metadata={"evidence_hash": evidence_hash}
                )
                logger.info(f"[Stage 10] Blockchain registration complete: Tx={blockchain_receipt.transaction_hash}")
            except Exception as e:
                stages[stage_key].status = StageStatus.FAILED
                stages[stage_key].error = str(e)
                logger.error(f"[Stage 10] Blockchain registration failed: {e}")
        else:
            stages[stage_key].status = StageStatus.SKIPPED

        # Finalize successful result
        return self._finalize_result(
            pipeline_id=pipeline_id,
            status="SUCCESS",
            input_path=input_path,
            input_sha256=input_sha256,
            stages=stages,
            provenance=provenance,
            start_time=start_time,
            face_detection=face_detection,
            all_faces_count=len(detections) if 'detections' in locals() else 0,
            face_quality=face_quality,
            search_audit=self.search_service.last_audit,
            candidates=analyzed_candidates,
            best_match=best_match,
            evidence_record=evidence_record,
            evidence_hash=evidence_hash,
            blockchain_receipt=blockchain_receipt
        )

    def _finalize_result(
        self,
        pipeline_id: str,
        status: str,
        input_path: Path,
        input_sha256: str,
        stages: Dict[str, StageExecution],
        provenance: ProvenanceChain,
        start_time: float,
        face_detection: Optional[FaceDetection] = None,
        all_faces_count: int = 0,
        face_quality: Optional[FaceQualityResult] = None,
        search_audit: Optional[SearchAuditEntry] = None,
        candidates: List[CandidateAnalysis] = None,
        best_match: Optional[CandidateAnalysis] = None,
        evidence_record: Optional[EvidenceRecord] = None,
        evidence_hash: Optional[str] = None,
        blockchain_receipt: Optional[BlockchainReceipt] = None
    ) -> PipelineDiscoveryResult:
        """Packages pipeline execution results."""
        duration_ms = round((time.time() - start_time) * 1000, 2)
        return PipelineDiscoveryResult(
            pipeline_id=pipeline_id,
            status=status,
            input_image_path=str(input_path),
            input_image_sha256=input_sha256,
            face_detection=face_detection,
            all_faces_detected=all_faces_count,
            face_quality=face_quality,
            search_audit=search_audit,
            candidates=candidates or [],
            best_match=best_match,
            evidence_record=evidence_record,
            evidence_hash=evidence_hash,
            blockchain_receipt=blockchain_receipt,
            provenance=provenance,
            stages=stages,
            total_duration_ms=duration_ms
        )


# Helper type
Union_Path = Union_type = Path | str
