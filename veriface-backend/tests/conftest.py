"""
VERIFACE CHAIN Pytest Configuration
"""
import pytest
import asyncio
from typing import Generator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from veriface.models.schemas import Base


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_veriface.db"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> Generator[AsyncSession, None, None]:
    """Create test database session"""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing"""
    import numpy as np
    from PIL import Image
    import io
    
    # Create a simple test image (100x100 RGB)
    img = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def sample_face_image_bytes():
    """Create sample face image bytes"""
    # This is a minimal valid JPEG for testing
    # In production, use actual face images
    import numpy as np
    from PIL import Image
    import io
    
    # Create a simple image that will pass basic validation
    img = Image.new('RGB', (200, 200), color=(100, 150, 200))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def sample_embedding():
    """Create sample face embedding"""
    import numpy as np
    return np.random.randn(128).astype(np.float64)


@pytest.fixture
def sample_evidence_json():
    """Create sample evidence JSON"""
    return {
        "session_id": "test-session-123",
        "timestamp": "2024-01-01T00:00:00",
        "input": {
            "id": "input-123",
            "filename": "test.jpg",
            "image_hash": "abc123def456"
        },
        "biometric": {
            "face_detected": True,
            "quality_score": 0.85,
            "embedding_size": 128
        },
        "candidates": [],
        "decision": {
            "match_found": False,
            "confidence_score": 0.0
        },
        "metadata": {
            "version": "1.0.0"
        }
    }