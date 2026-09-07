"""
VERIFACE CHAIN - Main Application Entry Point

Facial Recognition Verification System with Blockchain Provenance

This module initializes the FastAPI application and configures all routes.
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from veriface.core.config import settings
from veriface.models.database import init_db
from veriface.api.routers import verify_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print("Initializing database...")
    await init_db()
    print("Database initialized successfully")
    print("All services ready")
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
## VERIFACE CHAIN - Facial Recognition Verification System

This API provides a complete verification pipeline for facial recognition:

### Pipeline Stages:
1. **Input Acquisition** - Upload and validate face images
2. **Biometric Processing** - Face detection, quality checks, embedding generation
3. **Discovery Engine** - Multi-provider visual search
4. **Candidate Intelligence** - Candidate analysis and comparison
5. **Confidence Engine** - Similarity scoring and match decision
6. **Evidence Vault** - Structured evidence storage
7. **Crypto Engine** - SHA-256 hashing, Merkle tree generation
8. **Blockchain** - Immutable timestamped records
9. **Verification** - Chain validation and tamper detection

### Features:
- Face detection and quality assessment
- Multi-provider search capabilities
- Confidence-based match decisions
- Blockchain-based evidence provenance
- Tamper detection and verification
    """,
    version=settings.app_version,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount upload directory
import os
os.makedirs(settings.upload_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

# Include routers
app.include_router(verify_router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "documentation": "/docs",
        "health": "/api/v1/verify/health"
    }


@app.get("/api/v1/chain", tags=["Chain"])
async def get_chain_info():
    """Get blockchain chain information"""
    return {
        "name": "VERIFACE CHAIN",
        "version": settings.app_version,
        "consensus": "Proof of Work",
        "difficulty": settings.difficulty,
        "genesis_timestamp": settings.genesis_timestamp
    }


def main():
    """Run the application using uvicorn"""
    import uvicorn
    uvicorn.run(
        "veriface.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )


if __name__ == "__main__":
    main()