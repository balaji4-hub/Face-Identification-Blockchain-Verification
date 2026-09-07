"""
VeriFace Chain - Deterministic Canonicalization Engine
The single authoritative canonicalization module for the entire repository.
Guarantees identical JSON representation across any environment, machine, or language.
"""
import json
from typing import Dict, Any, Union
from pydantic import BaseModel
from core.exceptions import CanonicalizationError


def canonicalize_evidence(evidence: Union[BaseModel, Dict[str, Any]]) -> bytes:
    """
    Transforms an evidence object or dictionary into a deterministic UTF-8 byte stream.
    
    Rules enforced:
    1. Keys recursively sorted in lexicographical order (sort_keys=True).
    2. Zero extraneous whitespace in separators (separators=(',', ':')).
    3. ASCII character escaping for unicode safety (ensure_ascii=True).
    4. Deterministic encoding to UTF-8 bytes.
    
    Args:
        evidence: Pydantic model or python dictionary representing evidence.
        
    Returns:
        Deterministic bytes sequence suitable for cryptographic hashing.
    """
    try:
        if isinstance(evidence, BaseModel):
            evidence_dict = evidence.model_dump(mode="json")
        elif isinstance(evidence, dict):
            # Ensure any nested pydantic models or special types are serializable
            evidence_dict = json.loads(json.dumps(evidence, default=str))
        else:
            raise CanonicalizationError(f"Unsupported evidence type: {type(evidence)}")
            
        canonical_str = json.dumps(
            evidence_dict,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True
        )
        return canonical_str.encode("utf-8")
    except Exception as e:
        raise CanonicalizationError(f"Canonicalization failed: {str(e)}") from e


def canonical_json_str(evidence: Union[BaseModel, Dict[str, Any]]) -> str:
    """Returns the canonical JSON string representation."""
    return canonicalize_evidence(evidence).decode("utf-8")
