"""
VeriFace Chain - System Configuration
Loads configuration parameters from environment variables with strong type validation.
"""
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for VeriFace Chain pipeline."""
    
    # Application Metadata
    APP_NAME: str = "VeriFace Chain"
    APP_VERSION: str = "1.0.0"
    PIPELINE_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    INPUT_DIR: Path = BASE_DIR / "input"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    FIXTURES_DIR: Path = BASE_DIR / "tests" / "fixtures"
    
    # Input Validation
    MAX_FILE_SIZE_MB: int = 15
    ALLOWED_IMAGE_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp"
    ]
    MIN_IMAGE_WIDTH: int = 80
    MIN_IMAGE_HEIGHT: int = 80
    
    # Computer Vision: Face Detection & Quality
    FACE_DETECTION_CONFIDENCE: float = 0.60
    MIN_FACE_SIZE: int = 50
    MIN_FACE_RATIO: float = 0.02
    BLUR_THRESHOLD: float = 50.0
    FACE_QUALITY_THRESHOLD: float = 0.50
    FACE_RECOGNITION_MODEL: str = "insightface_arcface_r100"
    EMBEDDING_DIMENSION: int = 512
    
    # Decision Thresholds
    FACE_MATCH_THRESHOLD: float = 0.70
    HIGH_CONFIDENCE_THRESHOLD: float = 0.75
    POSSIBLE_MATCH_THRESHOLD: float = 0.50
    MAX_CANDIDATES: int = 20
    
    # Discovery Engine
    DEFAULT_SEARCH_PROVIDER: str = "mock"  # "mock", "authorized_api", "serpapi", "google_vision", "tineye"
    SEARCH_TIMEOUT_SECONDS: int = 15
    AUTHORIZED_SEARCH_API_KEY: Optional[str] = None
    AUTHORIZED_SEARCH_ENDPOINT: Optional[str] = None
    
    # Live Reverse Image Search API Providers
    SERPAPI_API_KEY: Optional[str] = None
    GOOGLE_VISION_API_KEY: Optional[str] = None
    TINEYE_API_KEY: Optional[str] = None
    
    # Candidate Fetcher Security
    MAX_DOWNLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    DOWNLOAD_TIMEOUT_SECONDS: int = 10
    MAX_REDIRECTS: int = 5
    
    # Cryptography
    HASH_ALGORITHM: str = "sha256"
    
    # Blockchain Configuration
    BLOCKCHAIN_ENABLED: bool = True
    BLOCKCHAIN_RPC_URL: str = "http://127.0.0.1:8545"
    CONTRACT_ADDRESS: str = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
    BLOCKCHAIN_CHAIN_ID: int = 31337  # Hardhat default chain ID
    BLOCKCHAIN_PRIVATE_KEY: Optional[str] = None
    BLOCKCHAIN_GAS_LIMIT: int = 300000
    
    # Audit Logging
    AUDIT_LOG_FILE: Path = BASE_DIR / "output" / "audit.jsonl"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure critical runtime directories exist
settings.INPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
