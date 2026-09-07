"""VERIFACE CHAIN - Facial Recognition Verification System

A comprehensive facial recognition verification system with blockchain-based provenance.

This package provides:
- Input acquisition for face images
- Biometric processing with face detection and embedding
- Multi-provider discovery engine
- Candidate intelligence and comparison
- Confidence-based decision engine
- Evidence vault with provenance tracking
- Cryptographic hashing and Merkle trees
- Blockchain recording
- Verification and tamper detection
"""

__version__ = "1.0.0"
__author__ = "VERIFACE Team"

from veriface.main import app
from veriface.models.database import init_db

__all__ = ["app", "init_db"]