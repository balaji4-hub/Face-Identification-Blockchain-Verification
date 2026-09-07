"""
VERIFACE CHAIN Database Models
SQLAlchemy models for evidence, candidates, and blockchain
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


class VerificationSession(Base):
    """Main verification session tracking"""
    __tablename__ = "verification_sessions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    match_found = Column(Boolean, nullable=True)
    match_confidence = Column(Float, nullable=True)
    
    # Relationships
    input_image = relationship("InputImage", back_populates="session", uselist=False)
    biometric_result = relationship("BiometricResult", back_populates="session", uselist=False)
    candidates = relationship("Candidate", back_populates="session")
    evidence = relationship("Evidence", back_populates="session")
    blockchain_record = relationship("BlockchainRecord", back_populates="session", uselist=False)


class InputImage(Base):
    """Stores input face image/scan data"""
    __tablename__ = "input_images"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("verification_sessions.id"))
    
    filename = Column(String(255))
    file_path = Column(String(512))
    file_size = Column(Integer)
    mime_type = Column(String(50))
    
    image_hash = Column(String(64), unique=True)
    width = Column(Integer)
    height = Column(Integer)
    channels = Column(Integer)
    
    captured_at = Column(DateTime, default=datetime.utcnow)
    extra_metadata = Column(JSON, nullable=True)
    
    # Relationship
    session = relationship("VerificationSession", back_populates="input_image")


class BiometricResult(Base):
    """Stores face detection, quality check, and embedding results"""
    __tablename__ = "biometric_results"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("verification_sessions.id"))
    
    face_detected = Column(Boolean)
    detection_confidence = Column(Float)
    
    quality_score = Column(Float)
    quality_check_passed = Column(Boolean)
    
    face_embedding = Column(LargeBinary)  # 128-d vector from dlib
    embedding_model = Column(String(50))
    embedding_size = Column(Integer)
    
    face_locations = Column(JSON, nullable=True)  # Bounding boxes
    landmarks = Column(JSON, nullable=True)  # Facial landmarks
    
    processing_time_ms = Column(Float)
    error_message = Column(Text, nullable=True)
    
    # Relationship
    session = relationship("VerificationSession", back_populates="biometric_result")


class Candidate(Base):
    """Stores candidate matches from discovery engine"""
    __tablename__ = "candidates"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("verification_sessions.id"))
    
    provider_name = Column(String(100))  # Source of the candidate
    external_id = Column(String(255), nullable=True)  # ID from external provider
    
    candidate_image_url = Column(String(512), nullable=True)
    candidate_image_path = Column(String(512), nullable=True)
    
    # Face analysis results
    face_detected = Column(Boolean)
    detection_confidence = Column(Float)
    face_count = Column(Integer, default=0)
    
    # Comparison results
    embedding = Column(LargeBinary, nullable=True)
    similarity_score = Column(Float)
    distance = Column(Float, nullable=True)
    
    # Ranking
    rank = Column(Integer)
    is_top_match = Column(Boolean, default=False)
    
    # Additional metadata
    extra_metadata = Column(JSON, nullable=True)
    downloaded_at = Column(DateTime)
    
    # Relationship
    session = relationship("VerificationSession", back_populates="candidates")


class Evidence(Base):
    """Stores evidence vault data"""
    __tablename__ = "evidence"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("verification_sessions.id"))
    
    # Evidence JSON
    evidence_json = Column(JSON)
    evidence_hash = Column(String(64), unique=True)
    
    # Image hash
    canonical_image_hash = Column(String(64))
    
    # Provenance
    provenance_chain = Column(JSON)
    source_metadata = Column(JSON, nullable=True)
    
    # Quality metrics
    overall_quality_score = Column(Float)
    biometric_quality_score = Column(Float)
    comparison_quality_score = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    verified_at = Column(DateTime, nullable=True)
    
    # Relationship
    session = relationship("VerificationSession", back_populates="evidence")


class BlockchainRecord(Base):
    """Stores blockchain transaction records"""
    __tablename__ = "blockchain_records"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("verification_sessions.id"))
    
    # Block information
    block_number = Column(Integer)
    block_hash = Column(String(64))
    transaction_hash = Column(String(64))
    
    # Evidence
    evidence_hash = Column(String(64))
    previous_hash = Column(String(64), nullable=True)
    
    # Merkle proof
    merkle_root = Column(String(64))
    merkle_proof = Column(JSON, nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow)
    confirmed = Column(Boolean, default=False)
    confirmations = Column(Integer, default=0)
    
    # Additional data
    extra_data = Column(JSON, nullable=True)
    
    # Relationship
    session = relationship("VerificationSession", back_populates="blockchain_record")


class BlockchainBlock(Base):
    """Simplified blockchain block storage"""
    __tablename__ = "blocks"
    
    index = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    evidence_hash = Column(String(64))
    previous_hash = Column(String(64))
    hash = Column(String(64))
    nonce = Column(Integer, default=0)
    transactions_count = Column(Integer, default=0)


class AuditLog(Base):
    """Audit log for compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String(36), nullable=True)
    action = Column(String(100))
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)