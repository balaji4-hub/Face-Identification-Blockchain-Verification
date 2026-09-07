"""
VERIFACE CHAIN Configuration Settings
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = "VERIFACE CHAIN"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./veriface.db"
    
    # Upload settings
    max_file_size_mb: int = 50
    allowed_image_types: List[str] = ["image/jpeg", "image/png", "image/bmp", "image/webp"]
    upload_dir: str = "./uploads"
    
    # Face Processing
    face_detection_confidence: float = 0.6
    embedding_model: str = "dlib_cnn"
    min_face_size: int = 100
    quality_threshold: float = 0.7
    
    # Discovery Engine
    search_providers: List[str] = ["internal", "external"]
    max_candidates: int = 100
    download_timeout: int = 30
    
    # Confidence Engine
    similarity_threshold: float = 0.6
    ranking_top_k: int = 10
    
    # Crypto Engine
    hash_algorithm: str = "sha256"
    merkle_leaf_count: int = 16
    
    # Blockchain
    block_reward: float = 1.0
    difficulty: int = 2
    genesis_timestamp: int = 1704067200  # 2024-01-01
    
    # Evidence Vault
    evidence_retention_days: int = 365
    hash_verification: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()