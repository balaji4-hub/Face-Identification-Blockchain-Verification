"""
VeriFace Chain - Cryptographic Fingerprinting Engine
Computes SHA-256 cryptographic digests and Solidity bytes32 representations.
"""
import hashlib
from typing import Union, Dict, Any
from pydantic import BaseModel
from evidence.canonicalizer import canonicalize_evidence


def hash_evidence(evidence: Union[BaseModel, Dict[str, Any]]) -> str:
    """
    Computes deterministic SHA-256 hexadecimal digest from canonical evidence.
    
    Args:
        evidence: Evidence record or dictionary.
        
    Returns:
        64-character lowercase hexadecimal hash string.
    """
    canonical_bytes = canonicalize_evidence(evidence)
    return hashlib.sha256(canonical_bytes).hexdigest().lower()


def hash_bytes(data: bytes) -> str:
    """Computes SHA-256 hexadecimal digest of raw bytes (e.g., image files)."""
    return hashlib.sha256(data).hexdigest().lower()


def hash_to_bytes32(hex_hash: str) -> bytes:
    """
    Converts a 64-character hex hash (with or without '0x' prefix) to 32 raw bytes
    compatible with Solidity bytes32 arguments.
    """
    clean_hex = hex_hash.strip().lower()
    if clean_hex.startswith("0x"):
        clean_hex = clean_hex[2:]
        
    if len(clean_hex) != 64:
        raise ValueError(f"Invalid SHA-256 hex string length ({len(clean_hex)}). Must be 64 characters.")
        
    return bytes.fromhex(clean_hex)


def bytes32_to_hex(b32: bytes) -> str:
    """Converts 32 raw bytes into a 0x-prefixed 64-character hex string."""
    if len(b32) != 32:
        raise ValueError(f"Expected 32 bytes, got {len(b32)}")
    return "0x" + b32.hex()


def validate_hash(evidence: Union[BaseModel, Dict[str, Any]], expected_hash: str) -> bool:
    """
    Validates whether canonical evidence recalculates to the expected SHA-256 hash.
    
    Args:
        evidence: Evidence record to evaluate.
        expected_hash: Hash string to verify against.
        
    Returns:
        True if recalculation matches expected hash exactly.
    """
    recalculated = hash_evidence(evidence)
    clean_expected = expected_hash.strip().lower()
    if clean_expected.startswith("0x"):
        clean_expected = clean_expected[2:]
    return recalculated == clean_expected
