"""
VERIFACE CHAIN Unit Tests - Services
"""
import pytest
import sys
import os
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestInputAcquisitionService:
    """Tests for InputAcquisition service"""
    
    @pytest.mark.asyncio
    async def test_calculate_image_hash(self):
        """Test image hash calculation"""
        from veriface.services.input_acquisition import InputAcquisitionService
        
        service = InputAcquisitionService(Mock())
        content = b"test image content"
        
        hash1 = service.calculate_image_hash(content)
        hash2 = service.calculate_image_hash(content)
        
        # Same content should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64
    
    def test_get_image_dimensions(self):
        """Test image dimension extraction"""
        from veriface.services.input_acquisition import InputAcquisitionService
        
        service = InputAcquisitionService(Mock())
        
        # Create test image
        from PIL import Image
        import io
        
        img = Image.new('RGB', (640, 480), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        
        width, height, channels = service.get_image_dimensions(buffer.getvalue())
        
        assert width == 640
        assert height == 480
        assert channels == 3


class TestBiometricProcessingService:
    """Tests for BiometricProcessing service"""
    
    @pytest.mark.asyncio
    async def test_check_min_face_size(self):
        """Test minimum face size check"""
        from veriface.services.biometric_processing import BiometricProcessingService
        
        service = BiometricProcessingService(Mock())
        
        # Face larger than minimum
        large_face = [0, 200, 200, 0]
        assert service.check_min_face_size(large_face, 100) is True
        
        # Face smaller than minimum
        small_face = [0, 50, 50, 0]
        assert service.check_min_face_size(small_face, 100) is False
    
    @pytest.mark.asyncio
    async def test_calculate_similarity(self):
        """Test similarity calculation"""
        from veriface.services.biometric_processing import BiometricProcessingService
        import numpy as np
        
        service = BiometricProcessingService(Mock())
        
        # Create identical embeddings
        embedding1 = np.zeros(128)
        embedding2 = np.zeros(128)
        
        similarity, distance = service.calculate_similarity(embedding1, embedding2)
        
        assert similarity == 1.0
        assert distance == 0.0
        
        # Create different embeddings
        embedding1 = np.ones(128)
        embedding2 = np.zeros(128)
        
        similarity, distance = service.calculate_similarity(embedding1, embedding2)
        
        assert similarity < 1.0
        assert distance > 0.0


class TestConfidenceEngineService:
    """Tests for ConfidenceEngine service"""
    
    def test_calculate_ensemble_score(self):
        """Test ensemble score calculation"""
        from veriface.services.confidence_engine import ConfidenceEngineService
        
        service = ConfidenceEngineService(Mock())
        
        # Perfect scores
        score = service.calculate_ensemble_score(
            similarity_score=1.0,
            detection_confidence=1.0,
            face_count=1,
            provider_trust=1.0
        )
        
        assert score <= 1.0
        assert score > 0.8
        
        # Poor scores
        score = service.calculate_ensemble_score(
            similarity_score=0.2,
            detection_confidence=0.3,
            face_count=3,
            provider_trust=0.5
        )
        
        assert score < 0.5
    
    def test_calculate_rank_score(self):
        """Test rank score calculation"""
        from veriface.services.confidence_engine import ConfidenceEngineService
        
        service = ConfidenceEngineService(Mock())
        
        # Top rank
        top_score = service.calculate_rank_score(1, 100)
        assert top_score > 0.9
        
        # Lower rank
        lower_score = service.calculate_rank_score(10, 100)
        assert lower_score < top_score
        
        # Empty list
        empty_score = service.calculate_rank_score(0, 0)
        assert empty_score == 0.0
    
    def test_detect_tie(self):
        """Test tie detection"""
        from veriface.services.confidence_engine import ConfidenceEngineService
        
        service = ConfidenceEngineService(Mock())
        
        # No tie
        scores = [0.8, 0.5, 0.3]
        assert service.detect_tie(scores) is False
        
        # Tie
        scores = [0.8, 0.79, 0.3]
        assert service.detect_tie(scores) is True
        
        # Single item
        scores = [0.8]
        assert service.detect_tie(scores) is False


class TestEvidenceVaultService:
    """Tests for EvidenceVault service"""
    
    @pytest.mark.asyncio
    async def test_calculate_provenance_score(self):
        """Test provenance score calculation"""
        from veriface.services.evidence_vault import EvidenceVaultService
        from veriface.models.schemas_api import EvidenceData
        
        service = EvidenceVaultService(Mock())
        
        # Complete evidence
        evidence = EvidenceData(
            session_id="test",
            timestamp="2024-01-01T00:00:00",
            input={"data": "test"},
            biometric={"face_detected": True, "quality_score": 0.8},
            candidates=[{"similarity_score": 0.7}],
            decision={"match_found": True},
            metadata={}
        )
        
        score = service.calculate_provenance_score(evidence)
        
        assert score > 0.0
        assert score <= 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_evidence_hash(self):
        """Test evidence hash calculation"""
        from veriface.services.evidence_vault import EvidenceVaultService
        from veriface.models.schemas_api import EvidenceData
        
        service = EvidenceVaultService(Mock())
        
        evidence = EvidenceData(
            session_id="test",
            timestamp="2024-01-01T00:00:00",
            input={"data": "test"},
            biometric={"face_detected": True},
            candidates=[],
            decision={},
            metadata={}
        )
        
        hash_value = service.calculate_evidence_hash(evidence)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64


class TestBlockchainService:
    """Tests for Blockchain service"""
    
    @pytest.mark.asyncio
    async def test_block_hash_calculation(self):
        """Test block hash calculation"""
        from veriface.services.blockchain import Block
        from datetime import datetime
        
        block = Block(
            index=1,
            timestamp=datetime.utcnow(),
            evidence_hash="abc123",
            previous_hash="prev123",
            hash="",
            nonce=0
        )
        
        block_hash = block.calculate_hash()
        
        assert isinstance(block_hash, str)
        assert len(block_hash) == 64
    
    @pytest.mark.asyncio
    async def test_proof_of_work(self):
        """Test proof of work"""
        from veriface.services.blockchain import Block, BlockchainService
        from datetime import datetime
        
        # Create block
        block = Block(
            index=1,
            timestamp=datetime.utcnow(),
            evidence_hash="abc123",
            previous_hash="0" * 64,
            hash="",
            nonce=0
        )
        
        service = BlockchainService(Mock())
        nonce, block_hash = service.proof_of_work(block)
        
        assert nonce >= 0
        assert block_hash.startswith("00")  # Difficulty 2


class TestVerificationEngineService:
    """Tests for VerificationEngine service"""
    
    @pytest.mark.asyncio
    async def test_extract_merkle_data(self):
        """Test Merkle data extraction"""
        from veriface.services.verification_engine import VerificationEngineService
        
        service = VerificationEngineService(Mock())
        
        evidence = {
            "input": {"test": "input"},
            "biometric": {"test": "biometric"},
            "candidates": [],
            "decision": {"test": "decision"}
        }
        
        data = service._extract_merkle_data(evidence)
        
        assert "input" in data
        assert "biometric" in data
        assert "candidates" in data
        assert "decision" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])