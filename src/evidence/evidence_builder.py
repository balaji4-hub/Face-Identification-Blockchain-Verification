"""
evidence_builder.py

Defines a structured, privacy-conscious EvidenceRecord for a single
match result, plus a builder/helper class for constructing and
exporting these records as JSON.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class MatchDecision(str, Enum):
    """Coarse-grained decision bucket derived from a similarity score."""
    HIGH_CONFIDENCE_MATCH = "HIGH_CONFIDENCE_MATCH"
    LIKELY_MATCH = "LIKELY_MATCH"
    UNCERTAIN = "UNCERTAIN"
    NO_MATCH = "NO_MATCH"


class EvidenceSource(str, Enum):
    """Where the candidate image/page was discovered."""
    PUBLIC_WEB = "Public Web"
    INTERNAL_DATASET = "Internal Dataset"
    USER_SUBMITTED = "User Submitted"
    THIRD_PARTY_API = "Third Party API"


# --------------------------------------------------------------------------
# Core model
# --------------------------------------------------------------------------

class EvidenceRecord(BaseModel):
    """
    A structured, hashable record describing a single candidate match.
    Deliberately excludes raw biometrics and secrets.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    evidence_id: str = Field(
        default_factory=lambda: f"EV-{uuid.uuid4().hex[:12].upper()}",
        description="Unique identifier for this evidence record.",
    )
    pipeline_version: str = Field(
        ...,
        description="Version tag of the pipeline that produced this record, e.g. '7.0.0'.",
    )
    source: EvidenceSource = Field(
        ..., description="Where the candidate was discovered."
    )
    page_url: str = Field(
        ..., description="URL of the page on which the candidate image was found."
    )
    image_url: str = Field(
        ..., description="Direct URL of the candidate image."
    )
    image_sha256: str = Field(
        ...,
        description="SHA-256 hex digest of the image bytes.",
    )
    similarity_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Similarity score in [-1, 1]."
    )
    match_decision: MatchDecision = Field(
        ..., description="Decision bucket derived from similarity_score / policy thresholds."
    )
    recognition_model: str = Field(
        ..., min_length=1, description="Name/version of the recognition model used."
    )
    discovered_at: datetime = Field(
        ..., description="UTC timestamp when the candidate was discovered/crawled."
    )
    analysis_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this evidence record was analyzed/built.",
    )

    # ---- validators -----------------------------------------------------

    @field_validator("image_sha256")
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        v = str(v).lower().strip()
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("image_sha256 must be a 64-character hex SHA-256 digest")
        return v

    @field_validator("discovered_at", "analysis_timestamp")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @field_validator("pipeline_version")
    @classmethod
    def validate_pipeline_version(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("pipeline_version must not be empty")
        return v.strip()

    # ---- convenience ------------------------------------------------------

    def canonical_json(self) -> str:
        """Deterministic JSON serialization suitable for hashing and blockchain anchoring."""
        return json.dumps(
            json.loads(self.model_dump_json()),
            sort_keys=True,
            separators=(",", ":"),
        )

    def record_hash(self) -> str:
        """SHA-256 hex digest of the canonical JSON — anchored on-chain."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------

class EvidenceBuilder:
    """Helper for constructing EvidenceRecord instances and exporting to JSON."""

    def __init__(self, pipeline_version: str, recognition_model: str):
        self.pipeline_version = pipeline_version
        self.recognition_model = recognition_model

    @staticmethod
    def hash_image_bytes(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def decide(similarity_score: float,
               high_threshold: float = 0.65,
               likely_threshold: float = 0.45,
               uncertain_threshold: float = 0.30) -> MatchDecision:
        if similarity_score >= high_threshold:
            return MatchDecision.HIGH_CONFIDENCE_MATCH
        if similarity_score >= likely_threshold:
            return MatchDecision.LIKELY_MATCH
        if similarity_score >= uncertain_threshold:
            return MatchDecision.UNCERTAIN
        return MatchDecision.NO_MATCH

    def build(
        self,
        *,
        source: EvidenceSource,
        page_url: str,
        image_url: str,
        image_sha256: str,
        similarity_score: float,
        discovered_at: Optional[datetime] = None,
        match_decision: Optional[MatchDecision] = None,
        evidence_id: Optional[str] = None,
    ) -> EvidenceRecord:
        kwargs = dict(
            pipeline_version=self.pipeline_version,
            source=source,
            page_url=page_url,
            image_url=image_url,
            image_sha256=image_sha256,
            similarity_score=similarity_score,
            match_decision=match_decision or self.decide(similarity_score),
            recognition_model=self.recognition_model,
            discovered_at=discovered_at or datetime.now(timezone.utc),
        )
        if evidence_id:
            kwargs["evidence_id"] = evidence_id
        return EvidenceRecord(**kwargs)

    @staticmethod
    def export_json(record: EvidenceRecord, path: str, *, indent: int = 2) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(record.model_dump_json(indent=indent))
        return path

    @staticmethod
    def export_json_batch(records: list[EvidenceRecord], path: str, *, indent: int = 2) -> str:
        payload = [json.loads(r.model_dump_json()) for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
        return path


if __name__ == "__main__":
    builder = EvidenceBuilder(pipeline_version="7.0.0", recognition_model="InsightFace-r100")
    fake_image_bytes = b"not-a-real-image-just-for-demo"
    digest = builder.hash_image_bytes(fake_image_bytes)

    record = builder.build(
        source=EvidenceSource.PUBLIC_WEB,
        page_url="https://example.com/profile/123",
        image_url="https://example.com/images/abc.jpg",
        image_sha256=digest,
        similarity_score=0.91,
    )

    print("Canonical JSON:", record.canonical_json())
    print("Record Hash  :", record.record_hash())
