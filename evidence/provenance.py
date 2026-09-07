"""
VeriFace Chain - Provenance Chain Tracker
Tracks step-by-step audit provenance across the entire lifecycle without leaking biometric data.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ProvenanceStep(BaseModel):
    """Immutable audit record for a single stage in the verification pipeline."""
    stage: str = Field(..., description="Stage name")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    input_fingerprint: Optional[str] = Field(default=None, description="SHA-256 of stage input")
    output_fingerprint: Optional[str] = Field(default=None, description="SHA-256 of stage output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sanitized execution parameters")


class ProvenanceChain(BaseModel):
    """Chronological chain of provenance steps."""
    pipeline_id: str = Field(..., description="Pipeline execution run ID")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    steps: List[ProvenanceStep] = Field(default_factory=list)
    
    def add_step(
        self,
        stage: str,
        input_fingerprint: Optional[str] = None,
        output_fingerprint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Appends a new verified provenance step."""
        clean_meta = {}
        if metadata:
            for k, v in metadata.items():
                if "embedding" not in k.lower() and "key" not in k.lower():
                    clean_meta[k] = v
                    
        self.steps.append(
            ProvenanceStep(
                stage=stage,
                input_fingerprint=input_fingerprint,
                output_fingerprint=output_fingerprint,
                metadata=clean_meta
            )
        )
