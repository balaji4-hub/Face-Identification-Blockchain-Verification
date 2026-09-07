"""
VeriFace Chain - Pipeline: Stage Definitions
Defines pipeline stages, transitions, and observable stage execution models.
"""
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from core.result_types import StageStatus


class PipelineStage(str, Enum):
    """Enumeration of all discrete pipeline execution stages."""
    INPUT_VALIDATION = "input_validation"
    FACE_DETECTION = "face_detection"
    FACE_QUALITY = "face_quality"
    FACE_EMBEDDING = "face_embedding"
    VISUAL_SEARCH = "visual_search"
    CANDIDATE_ANALYSIS = "candidate_analysis"
    MATCH_DECISION = "match_decision"
    EVIDENCE_GENERATION = "evidence_generation"
    CANONICAL_HASHING = "canonical_hashing"
    BLOCKCHAIN_REGISTRATION = "blockchain_registration"


class StageExecution(BaseModel):
    """Tracks the observable state and duration of an individual stage."""
    stage: PipelineStage
    status: StageStatus = StageStatus.PENDING
    duration_ms: float = 0.0
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
