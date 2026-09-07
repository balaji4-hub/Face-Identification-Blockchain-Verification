"""
STAGE 7: CRYPTO ENGINE
Handles SHA-256 hashing and Merkle tree generation
"""
import hashlib
import json
import time
import math
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from functools import reduce

from veriface.core.config import settings


@dataclass
class MerkleNode:
    """Node in Merkle tree"""
    hash: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    is_leaf: bool = False


@dataclass
class MerkleProof:
    """Proof of inclusion in Merkle tree"""
    leaf_index: int
    leaf_hash: str
    merkle_root: str
    proof: List[Dict[str, Any]]
    hash_algorithm: str = "sha256"


class CryptoEngineService:
    """Service for cryptographic operations"""
    
    def __init__(self):
        self.hash_algorithm = settings.hash_algorithm
        self.merkle_leaf_count = settings.merkle_leaf_count
    
    def sha256_hash(self, data: Any) -> str:
        """
        Calculate SHA-256 hash of data
        
        Args:
            data: Any serializable data
            
        Returns:
            Hex string of hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        elif not isinstance(data, bytes):
            data = json.dumps(data, sort_keys=True, default=str).encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    def double_sha256(self, data: bytes) -> str:
        """Double SHA-256 (like Bitcoin)"""
        return hashlib.sha256(hashlib.sha256(data).digest()).hexdigest()
    
    def hash_concat(self, hash1: str, hash2: str) -> str:
        """Concatenate and hash two hashes"""
        return self.sha256(hash1 + hash2)
    
    def calculate_canonical_hash(self, evidence_json: Dict[str, Any]) -> str:
        """
        Calculate canonical hash for evidence
        
        Canonicalization ensures consistent hash regardless of
        key order in JSON
        """
        # Sort keys for deterministic output
        canonical_json = json.dumps(evidence_json, sort_keys=True, default=str)
        return self.sha256_hash(canonical_json)
    
    def calculate_image_hash(self, image_bytes: bytes) -> str:
        """Calculate hash of image content"""
        return self.sha256_hash(image_bytes)
    
    def build_merkle_leaf(self, data: Any, index: int) -> MerkleNode:
        """Build a leaf node for Merkle tree"""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True, default=str)
        
        hash_value = self.sha256_hash(str(index) + str(data))
        
        return MerkleNode(hash=hash_value, is_leaf=True)
    
    def build_merkle_tree(self, leaves: List[MerkleNode]) -> MerkleNode:
        """
        Build Merkle tree from leaves
        
        Args:
            leaves: List of leaf nodes
            
        Returns:
            Root node of Merkle tree
        """
        if len(leaves) == 0:
            return MerkleNode(hash=self.sha256(""), is_leaf=False)
        
        if len(leaves) == 1:
            return leaves[0]
        
        # Pad to power of 2
        n = len(leaves)
        next_power = 2 ** math.ceil(math.log2(n))
        
        # Create padding nodes
        padding = [MerkleNode(hash=self.sha256_hash(""), is_leaf=True) 
                   for _ in range(next_power - n)]
        all_leaves = leaves + padding
        
        # Build tree bottom-up
        current_level = all_leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                
                parent_hash = self.sha256_hash(left.hash + right.hash)
                next_level.append(MerkleNode(
                    hash=parent_hash,
                    left=left,
                    right=right,
                    is_leaf=False
                ))
            
            current_level = next_level
        
        return current_level[0]
    
    def get_merkle_proof(self, root: MerkleNode, leaf_hash: str, index: int) -> Optional[MerkleProof]:
        """
        Generate Merkle proof for a leaf
        
        Args:
            root: Root of Merkle tree
            leaf_hash: Hash of the leaf to prove
            index: Index of the leaf
            
        Returns:
            MerkleProof or None if not found
        """
        def find_proof(node: MerkleNode, target_hash: str, index: int, depth: int = 0) -> List[Dict[str, Any]]:
            if node.is_leaf:
                if node.hash == target_hash:
                    return []
                return None
            
            # Check left subtree
            proof = []
            left_proof = None
            right_proof = None
            
            if node.left:
                left_result = find_proof(node.left, target_hash, index * 2, depth + 1)
                if left_result is not None:
                    left_proof = left_result
            
            # Check right subtree
            if node.right:
                right_result = find_proof(node.right, target_hash, index * 2 + 1, depth + 1)
                if right_result is not None:
                    right_proof = right_result
            
            # Determine which branch contains our target
            if left_proof is not None:
                # Target is in left, add right hash to proof
                if node.right:
                    proof.append({
                        "direction": "right",
                        "hash": node.right.hash,
                        "index": index * 2 + 1
                    })
                return left_proof + proof
            elif right_proof is not None:
                # Target is in right, add left hash to proof
                if node.left:
                    proof.append({
                        "direction": "left",
                        "hash": node.left.hash,
                        "index": index * 2
                    })
                return right_proof + proof
            
            return None
        
        proof_hashes = find_proof(root, leaf_hash, index)
        
        if proof_hashes is None:
            return None
        
        return MerkleProof(
            leaf_index=index,
            leaf_hash=leaf_hash,
            merkle_root=root.hash,
            proof=proof_hashes,
            hash_algorithm=self.hash_algorithm
        )
    
    def verify_merkle_proof(
        self,
        proof: MerkleProof,
        leaf_data: Any
    ) -> bool:
        """
        Verify a Merkle proof
        
        Args:
            proof: MerkleProof to verify
            leaf_data: Original data of the leaf
            
        Returns:
            True if proof is valid
        """
        # Recalculate leaf hash — must match build_merkle_leaf serialization
        _leaf_data = leaf_data
        if isinstance(_leaf_data, dict):
            _leaf_data = json.dumps(_leaf_data, sort_keys=True, default=str)
        leaf_hash = self.sha256_hash(str(proof.leaf_index) + str(_leaf_data))
        
        if leaf_hash != proof.leaf_hash:
            return False
        
        # Reconstruct path to root
        current_hash = leaf_hash
        current_index = proof.leaf_index
        
        for step in proof.proof:
            if step["direction"] == "left":
                # Our hash is right, combine with left sibling
                current_hash = self.sha256_hash(step["hash"] + current_hash)
            else:
                # Our hash is left, combine with right sibling
                current_hash = self.sha256_hash(current_hash + step["hash"])
            current_index = current_index // 2
        
        return current_hash == proof.merkle_root
    
    async def process_evidence(
        self,
        evidence_json: Dict[str, Any],
        canonical_image_hash: str
    ) -> Dict[str, Any]:
        """
        Process evidence through crypto engine
        
        Args:
            evidence_json: Evidence data
            canonical_image_hash: Hash of canonical image
            
        Returns:
            Dict with hash and merkle data
        """
        start_time = time.time()
        
        # Calculate canonical hash
        canonical_hash = self.calculate_canonical_hash(evidence_json)
        
        # Build Merkle tree from evidence components
        merkle_leaves = [
            self.build_merkle_leaf({"type": "input", "data": evidence_json.get("input", {})}, 0),
            self.build_merkle_leaf({"type": "biometric", "data": evidence_json.get("biometric", {})}, 1),
            self.build_merkle_leaf({"type": "candidates", "data": evidence_json.get("candidates", [])}, 2),
            self.build_merkle_leaf({"type": "decision", "data": evidence_json.get("decision", {})}, 3),
            self.build_merkle_leaf({"type": "image_hash", "data": canonical_image_hash}, 4),
        ]
        
        # Build Merkle tree
        merkle_root = self.build_merkle_tree(merkle_leaves)
        
        # Generate proof for canonical hash leaf
        merkle_proof = self.get_merkle_proof(merkle_root, canonical_image_hash, 4)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "sha256_hash": canonical_hash,
            "merkle_root": merkle_root.hash if merkle_root else None,
            "merkle_proof": merkle_proof,
            "is_canonical": True,
            "processing_time_ms": processing_time
        }
    
    def verify_chain_integrity(
        self,
        evidence_hash: str,
        blockchain_record: Dict[str, Any]
    ) -> bool:
        """Verify integrity against blockchain record"""
        stored_hash = blockchain_record.get("evidence_hash")
        return stored_hash == evidence_hash


# Singleton instance
crypto_engine = CryptoEngineService()