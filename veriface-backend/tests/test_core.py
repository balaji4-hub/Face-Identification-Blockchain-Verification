"""
VERIFACE CHAIN Unit Tests - Core
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    """Tests for configuration"""
    
    def test_settings_loaded(self):
        """Test that settings load correctly"""
        from veriface.core.config import settings
        assert settings.app_name == "VERIFACE CHAIN"
        assert settings.app_version == "1.0.0"
    
    def test_face_processing_settings(self):
        """Test face processing configuration"""
        from veriface.core.config import settings
        assert settings.face_detection_confidence == 0.6
        assert settings.quality_threshold == 0.7
        assert settings.min_face_size == 100
    
    def test_confidence_settings(self):
        """Test confidence engine configuration"""
        from veriface.core.config import settings
        assert settings.similarity_threshold == 0.6
        assert settings.ranking_top_k == 10
    
    def test_crypto_settings(self):
        """Test crypto configuration"""
        from veriface.core.config import settings
        assert settings.hash_algorithm == "sha256"
        assert settings.merkle_leaf_count == 16
    
    def test_blockchain_settings(self):
        """Test blockchain configuration"""
        from veriface.core.config import settings
        assert settings.difficulty == 2
        assert settings.block_reward == 1.0


class TestCryptoEngine:
    """Tests for crypto engine"""
    
    def test_sha256_hash(self):
        """Test SHA-256 hashing"""
        from veriface.services.crypto_engine import CryptoEngineService
        
        crypto = CryptoEngineService()
        result = crypto.sha256_hash("test data")
        
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 produces 64 hex chars
    
    def test_sha256_consistency(self):
        """Test that same input produces same hash"""
        from veriface.services.crypto_engine import CryptoEngineService
        
        crypto = CryptoEngineService()
        hash1 = crypto.sha256_hash("consistent data")
        hash2 = crypto.sha256_hash("consistent data")
        
        assert hash1 == hash2
    
    def test_sha256_different_inputs(self):
        """Test that different inputs produce different hashes"""
        from veriface.services.crypto_engine import CryptoEngineService
        
        crypto = CryptoEngineService()
        hash1 = crypto.sha256_hash("data1")
        hash2 = crypto.sha256_hash("data2")
        
        assert hash1 != hash2
    
    def test_calculate_canonical_hash(self):
        """Test canonical hash calculation"""
        from veriface.services.crypto_engine import CryptoEngineService
        
        crypto = CryptoEngineService()
        
        data = {"name": "test", "value": 1}
        hash1 = crypto.calculate_canonical_hash(data)
        hash2 = crypto.calculate_canonical_hash(data)
        
        # Same data should produce same hash
        assert hash1 == hash2
        
        # Different order should produce same hash
        data2 = {"value": 1, "name": "test"}
        hash3 = crypto.calculate_canonical_hash(data2)
        
        assert hash1 == hash3
    
    def test_merkle_tree_construction(self):
        """Test Merkle tree construction"""
        from veriface.services.crypto_engine import CryptoEngineService
        
        crypto = CryptoEngineService()
        
        # Create leaf nodes
        leaves = []
        for i in range(4):
            leaf = crypto.build_merkle_leaf({"data": i}, i)
            leaves.append(leaf)
        
        # Build tree
        root = crypto.build_merkle_tree(leaves)
        
        assert root is not None
        assert isinstance(root.hash, str)
        assert len(root.hash) == 64
    
    def test_merkle_proof(self):
        """Test Merkle proof generation"""
        from veriface.services.crypto_engine import CryptoEngineService
        
        crypto = CryptoEngineService()
        
        # Create leaves
        leaves = []
        for i in range(4):
            leaf = crypto.build_merkle_leaf({"data": i}, i)
            leaves.append(leaf)
        
        # Build tree
        root = crypto.build_merkle_tree(leaves)
        
        # Get proof for leaf 2
        leaf_hash = leaves[2].hash
        proof = crypto.get_merkle_proof(root, leaf_hash, 2)
        
        assert proof is not None
        assert proof.merkle_root == root.hash
        assert len(proof.proof) > 0
    
    def test_merkle_proof_verification(self):
        """Test Merkle proof verification"""
        from veriface.services.crypto_engine import CryptoEngineService
        
        crypto = CryptoEngineService()
        
        # Create leaves
        leaves = []
        data_list = [{"data": i} for i in range(4)]
        for i, data in enumerate(data_list):
            leaf = crypto.build_merkle_leaf(data, i)
            leaves.append(leaf)
        
        # Build tree
        root = crypto.build_merkle_tree(leaves)
        
        # Get proof and verify
        proof = crypto.get_merkle_proof(root, leaves[2].hash, 2)
        verified = crypto.verify_merkle_proof(proof, {"data": 2})
        
        assert verified is True


class TestModels:
    """Tests for database models"""
    
    def test_verification_session_model(self):
        """Test VerificationSession model structure"""
        from veriface.models.schemas import VerificationSession
        
        session = VerificationSession()
        assert hasattr(session, 'id')
        assert hasattr(session, 'status')
        assert hasattr(session, 'created_at')
        assert hasattr(session, 'match_found')
    
    def test_input_image_model(self):
        """Test InputImage model structure"""
        from veriface.models.schemas import InputImage
        
        image = InputImage()
        assert hasattr(image, 'id')
        assert hasattr(image, 'session_id')
        assert hasattr(image, 'filename')
        assert hasattr(image, 'image_hash')
    
    def test_biometric_result_model(self):
        """Test BiometricResult model structure"""
        from veriface.models.schemas import BiometricResult
        
        result = BiometricResult()
        assert hasattr(result, 'face_detected')
        assert hasattr(result, 'quality_score')
        assert hasattr(result, 'face_embedding')
    
    def test_candidate_model(self):
        """Test Candidate model structure"""
        from veriface.models.schemas import Candidate
        
        candidate = Candidate()
        assert hasattr(candidate, 'provider_name')
        assert hasattr(candidate, 'similarity_score')
        assert hasattr(candidate, 'rank')
    
    def test_evidence_model(self):
        """Test Evidence model structure"""
        from veriface.models.schemas import Evidence
        
        evidence = Evidence()
        assert hasattr(evidence, 'evidence_json')
        assert hasattr(evidence, 'evidence_hash')
        assert hasattr(evidence, 'provenance_chain')
    
    def test_blockchain_record_model(self):
        """Test BlockchainRecord model structure"""
        from veriface.models.schemas import BlockchainRecord
        
        record = BlockchainRecord()
        assert hasattr(record, 'transaction_hash')
        assert hasattr(record, 'evidence_hash')
        assert hasattr(record, 'block_number')


class TestAPISchemas:
    """Tests for API schemas"""
    
    def test_health_response(self):
        """Test HealthResponse schema"""
        from veriface.models.schemas_api import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            database="connected",
            services={}
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
    
    def test_verification_status_enum(self):
        """Test VerificationStatus enum"""
        from veriface.models.schemas_api import VerificationStatus
        
        assert VerificationStatus.PENDING == "pending"
        assert VerificationStatus.PROCESSING == "processing"
        assert VerificationStatus.COMPLETED == "completed"
        assert VerificationStatus.FAILED == "failed"
    
    def test_face_location_schema(self):
        """Test FaceLocation schema"""
        from veriface.models.schemas_api import FaceLocation
        
        location = FaceLocation(top=100, right=200, bottom=300, left=50)
        
        assert location.top == 100
        assert location.right == 200
        assert location.bottom == 300
        assert location.left == 50
    
    def test_candidate_info_schema(self):
        """Test CandidateInfo schema"""
        from veriface.models.schemas_api import CandidateInfo
        
        info = CandidateInfo(
            candidate_id="test-123",
            provider_name="internal",
            similarity_score=0.85,
            rank=1,
            is_top_match=True
        )
        
        assert info.candidate_id == "test-123"
        assert info.similarity_score == 0.85
        assert info.rank == 1
        assert info.is_top_match is True
    
    def test_blockchain_transaction_schema(self):
        """Test BlockchainTransaction schema"""
        from veriface.models.schemas_api import BlockchainTransaction
        from datetime import datetime
        
        tx = BlockchainTransaction(
            transaction_hash="abc123",
            block_number=1,
            block_hash="def456",
            timestamp=datetime.utcnow(),
            evidence_hash="hash123"
        )
        
        assert tx.transaction_hash == "abc123"
        assert tx.block_number == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])