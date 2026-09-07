"""
VERIFACE CHAIN Verification Router
Main API endpoints for the verification pipeline
"""
import io
import time
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from veriface.models.database import get_db
from veriface.models.schemas import VerificationSession
from veriface.models.schemas_api import (
    VerificationChainResponse, VerificationStatus, VerificationResult,
    HealthResponse, ErrorResponse, AuditLogResponse
)
from veriface.services import (
    InputAcquisitionService,
    BiometricProcessingService,
    DiscoveryEngineService,
    CandidateIntelligenceService,
    ConfidenceEngineService,
    EvidenceVaultService,
    CryptoEngineService,
    BlockchainService,
    VerificationEngineService,
    VerificationOrchestrator
)

router = APIRouter(prefix="/api/v1/verify", tags=["Verification Pipeline"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database="connected",
        services={
            "input_acquisition": "ready",
            "biometric_processing": "ready",
            "discovery_engine": "ready",
            "candidate_intelligence": "ready",
            "confidence_engine": "ready",
            "evidence_vault": "ready",
            "crypto_engine": "ready",
            "blockchain": "ready",
            "verification": "ready"
        }
    )


@router.post("/sessions", response_model=dict)
async def create_session(db: AsyncSession = Depends(get_db)):
    """Create a new verification session"""
    session = VerificationSession(status="pending")
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    return {
        "session_id": session.id,
        "status": session.status,
        "created_at": session.created_at.isoformat()
    }


@router.post("/{session_id}/upload")
async def upload_image(
    session_id: str,
    file: UploadFile = File(...),
    auto_process: bool = Query(True),
    db: AsyncSession = Depends(get_db)
):
    """
    STAGE 1: INPUT ACQUISITION
    Upload face image for verification
    """
    # Validate session exists
    session = await db.get(VerificationSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Read file content
    content = await file.read()
    
    # Process upload
    service = InputAcquisitionService(db)
    result = await service.process_upload(
        filename=file.filename,
        content=content,
        mime_type=file.content_type or "image/jpeg"
    )
    
    # Auto-process if requested
    if auto_process:
        orchestrator = VerificationOrchestrator(db)
        await orchestrator.run_full_pipeline(session_id)
    
    return {
        "session_id": session_id,
        **result.model_dump()
    }


@router.post("/{session_id}/biometric")
async def process_biometric(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    STAGE 2: BIOMETRIC PROCESSING
    Detect faces, check quality, generate embeddings
    """
    # Get input image
    from veriface.models.schemas import InputImage
    result = await db.execute(
        select(InputImage).where(InputImage.session_id == session_id)
    )
    input_image = result.scalar_one_or_none()
    
    if not input_image:
        raise HTTPException(status_code=404, detail="No input image found")
    
    service = BiometricProcessingService(db)
    result = await service.process_biometrics(session_id, input_image.file_path)
    
    return result.model_dump()


@router.post("/{session_id}/discover")
async def discover_candidates(
    session_id: str,
    search_providers: list = Query(None),
    max_candidates: int = Query(50),
    db: AsyncSession = Depends(get_db)
):
    """
    STAGE 3: DISCOVERY ENGINE
    Search multiple providers for candidate matches
    """
    # Get biometric result
    from veriface.models.schemas import BiometricResult
    result = await db.execute(
        select(BiometricResult).where(BiometricResult.session_id == session_id)
    )
    biometric = result.scalar_one_or_none()
    
    if not biometric or not biometric.face_embedding:
        raise HTTPException(status_code=400, detail="No biometric data available")
    
    import numpy as np
    embedding = np.frombuffer(biometric.face_embedding, dtype=np.float64)
    image_hash = ""
    
    from veriface.models.schemas import InputImage
    img_result = await db.execute(
        select(InputImage).where(InputImage.session_id == session_id)
    )
    input_image = img_result.scalar_one_or_none()
    if input_image:
        image_hash = input_image.image_hash
    
    from veriface.models.schemas_api import DiscoveryRequest
    request = DiscoveryRequest(
        search_providers=search_providers,
        max_candidates=max_candidates
    )
    
    service = DiscoveryEngineService(db)
    result = await service.discover_candidates(
        session_id, embedding.tolist(), image_hash, request
    )
    
    return result.model_dump()


@router.post("/{session_id}/analyze-candidates")
async def analyze_candidates(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    STAGE 4: CANDIDATE INTELLIGENCE
    Download and analyze candidate images
    """
    # Get biometric embedding
    from veriface.models.schemas import BiometricResult
    result = await db.execute(
        select(BiometricResult).where(BiometricResult.session_id == session_id)
    )
    biometric = result.scalar_one_or_none()
    
    if not biometric or not biometric.face_embedding:
        raise HTTPException(status_code=400, detail="No biometric data available")
    
    import numpy as np
    embedding = np.frombuffer(biometric.face_embedding, dtype=np.float64)
    
    service = CandidateIntelligenceService(db)
    result = await service.analyze_all_candidates(session_id, embedding)
    
    return result.model_dump()


@router.post("/{session_id}/confidence")
async def compute_confidence(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    STAGE 5: CONFIDENCE ENGINE
    Calculate similarity scores and make match decision
    """
    service = ConfidenceEngineService(db)
    result = await service.compute_confidence(session_id)
    
    return result.model_dump()


@router.post("/{session_id}/evidence")
async def create_evidence(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    STAGE 6: EVIDENCE VAULT
    Create evidence record
    """
    service = EvidenceVaultService(db)
    result = await service.create_evidence(session_id)
    
    return result.model_dump()


@router.post("/{session_id}/crypto")
async def process_crypto(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    STAGE 7: CRYPTO ENGINE
    Calculate hashes and Merkle proofs
    """
    # Get evidence
    from veriface.models.schemas import Evidence
    result = await db.execute(
        select(Evidence).where(Evidence.session_id == session_id)
    )
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="No evidence found")
    
    service = CryptoEngineService()
    crypto_result = await service.process_evidence(
        evidence.evidence_json,
        evidence.canonical_image_hash
    )
    
    return {
        "session_id": session_id,
        "sha256_hash": crypto_result["sha256_hash"],
        "merkle_root": crypto_result["merkle_root"],
        "is_canonical": crypto_result["is_canonical"],
        "processing_time_ms": crypto_result["processing_time_ms"]
    }


@router.post("/{session_id}/blockchain")
async def record_blockchain(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    STAGE 8: BLOCKCHAIN
    Record evidence on blockchain
    """
    # Get evidence and crypto results
    from veriface.models.schemas import Evidence
    result = await db.execute(
        select(Evidence).where(Evidence.session_id == session_id)
    )
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="No evidence found")
    
    service = BlockchainService(db)
    result = await service.record_evidence(
        session_id,
        evidence.evidence_hash,
        evidence.evidence_hash  # Use as merkle root placeholder
    )
    
    return result.model_dump()


@router.get("/{session_id}/verify", response_model=VerificationResult)
async def run_verification(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    STAGE 9: VERIFICATION
    Run complete verification chain
    """
    service = VerificationEngineService(db)
    result = await service.run_full_verification(session_id)
    
    return result


@router.post("/{session_id}/pipeline")
async def run_full_pipeline(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Run complete verification pipeline (all 9 stages)
    """
    orchestrator = VerificationOrchestrator(db)
    result = await orchestrator.run_full_pipeline(session_id)
    
    return result


@router.get("/{session_id}/status")
async def get_status(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get current verification status"""
    orchestrator = VerificationOrchestrator(db)
    status = await orchestrator.get_pipeline_status(session_id)
    
    return status


@router.get("/{session_id}/evidence")
async def get_evidence(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get evidence record"""
    from veriface.models.schemas import Evidence
    result = await db.execute(
        select(Evidence).where(Evidence.session_id == session_id)
    )
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="No evidence found")
    
    return {
        "session_id": session_id,
        "evidence_id": evidence.id,
        "evidence_hash": evidence.evidence_hash,
        "quality_score": evidence.overall_quality_score,
        "created_at": evidence.created_at.isoformat(),
        "evidence_json": evidence.evidence_json
    }


@router.get("/{session_id}/blockchain")
async def get_blockchain_record(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get blockchain record"""
    from veriface.models.schemas import BlockchainRecord
    result = await db.execute(
        select(BlockchainRecord).where(BlockchainRecord.session_id == session_id)
    )
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(status_code=404, detail="No blockchain record found")
    
    return {
        "session_id": session_id,
        "transaction_hash": record.transaction_hash,
        "block_number": record.block_number,
        "block_hash": record.block_hash,
        "evidence_hash": record.evidence_hash,
        "timestamp": record.timestamp.isoformat(),
        "confirmations": record.confirmations
    }


@router.get("/{session_id}/tamper-check")
async def check_tampering(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Check for tampering"""
    service = VerificationEngineService(db)
    result = await service.tamper_detection(session_id)
    
    return result


@router.post("/{session_id}/recalculate")
async def recalculate_chain(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Recalculate verification chain"""
    service = VerificationEngineService(db)
    result = await service.recalculate_chain(session_id)
    
    return result


# Helper imports
from sqlalchemy import select