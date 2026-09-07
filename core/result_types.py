"""
VeriFace Chain - Common Result Types and Enums
Defines standardized enumerations and data types across the pipeline.
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class StageStatus(str, Enum):
    """Lifecycle states for pipeline execution stages."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class MatchDecision(str, Enum):
    """Confidence-based decision states."""
    HIGH_CONFIDENCE_MATCH = "HIGH_CONFIDENCE_MATCH"
    POSSIBLE_MATCH = "POSSIBLE_MATCH"
    NO_MATCH = "NO_MATCH"


class VerificationState(str, Enum):
    """Integrity verification states from blockchain lookup."""
    VERIFIED = "VERIFIED"
    NOT_REGISTERED = "NOT_REGISTERED"
    TAMPERED_OR_DIFFERENT = "TAMPERED_OR_DIFFERENT"
    BLOCKCHAIN_UNAVAILABLE = "BLOCKCHAIN_UNAVAILABLE"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"


class StageResult(BaseModel):
    """Result of a single pipeline stage execution."""
    stage_name: str
    status: StageStatus
    duration_ms: float = 0.0
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
