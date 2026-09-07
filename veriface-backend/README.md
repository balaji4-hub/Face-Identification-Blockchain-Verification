# VERIFACE CHAIN

A comprehensive facial recognition verification system with blockchain-based provenance.

## Overview

VERIFACE CHAIN provides a complete pipeline for facial recognition verification, from image input to blockchain-backed evidence recording. The system implements 9 distinct stages:

```
┌─────────────────────────┐
│    INPUT ACQUISITION    │  → Face Image / Face Scan
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ BIOMETRIC PROCESSING    │  → Face Detection, Quality Check, Embedding
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ DISCOVERY ENGINE        │  → Dynamic Visual Search, Multi-provider
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ CANDIDATE INTELLIGENCE  │  → Download Candidates, Face Detection, Embedding
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ CONFIDENCE ENGINE       │  → Similarity Scoring, Ranking, Decision
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ EVIDENCE VAULT          │  → Evidence JSON, Image Hash, Provenance
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ CRYPTO ENGINE           │  → SHA-256, Merkle Trees
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ BLOCKCHAIN              │  → Evidence Hash, Timestamp, Transaction
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ VERIFICATION            │  → Recalculate, Query, Tamper Detect
└─────────────────────────┘
```

## Features

- **Face Detection & Quality Assessment**: Uses dlib/face_recognition for accurate face detection and quality metrics
- **Multi-provider Search**: Extensible discovery engine for searching multiple face databases
- **Confidence-based Matching**: Ensemble scoring with configurable thresholds
- **Evidence Vault**: Structured evidence storage with provenance tracking
- **Cryptographic Security**: SHA-256 hashing and Merkle tree proofs
- **Blockchain Recording**: Immutable proof of verification with timestamps
- **Tamper Detection**: Chain verification and integrity checks

## Installation

```bash
# Clone the repository
cd veriface-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file based on `.env.example`:

```env
# Application
DEBUG=True
APP_NAME="VERIFACE CHAIN"
APP_VERSION="1.0.0"

# Database
DATABASE_URL="sqlite+aiosqlite:///./veriface.db"

# Upload Settings
MAX_FILE_SIZE_MB=50
UPLOAD_DIR="./uploads"

# Face Processing
FACE_DETECTION_CONFIDENCE=0.6
QUALITY_THRESHOLD=0.7
MIN_FACE_SIZE=100

# Confidence Engine
SIMILARITY_THRESHOLD=0.6
RANKING_TOP_K=10

# Blockchain
DIFFICULTY=2
BLOCK_REWARD=1.0
```

## Running the Application

```bash
# Start the server
python -m veriface.main

# Or with uvicorn directly
uvicorn veriface.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```
GET /api/v1/verify/health
```

### Session Management
```
POST /api/v1/verify/sessions          # Create new session
GET  /api/v1/verify/{id}/status       # Get session status
```

### Pipeline Stages
```
POST /api/v1/verify/{id}/upload       # Stage 1: Input Acquisition
POST /api/v1/verify/{id}/biometric    # Stage 2: Biometric Processing
POST /api/v1/verify/{id}/discover     # Stage 3: Discovery Engine
POST /api/v1/verify/{id}/analyze-candidates  # Stage 4: Candidate Intelligence
POST /api/v1/verify/{id}/confidence   # Stage 5: Confidence Engine
POST /api/v1/verify/{id}/evidence     # Stage 6: Evidence Vault
POST /api/v1/verify/{id}/crypto       # Stage 7: Crypto Engine
POST /api/v1/verify/{id}/blockchain   # Stage 8: Blockchain
```

### Full Pipeline
```
POST /api/v1/verify/{id}/pipeline     # Run all 9 stages
GET  /api/v1/verify/{id}/verify       # Run verification
```

### Verification Queries
```
GET  /api/v1/verify/{id}/evidence     # Get evidence
GET  /api/v1/verify/{id}/blockchain   # Get blockchain record
POST /api/v1/verify/{id}/tamper-check # Check tampering
POST /api/v1/verify/{id}/recalculate  # Recalculate chain
```

## Usage Example

```python
import httpx

# 1. Create session
response = requests.post("http://localhost:8000/api/v1/verify/sessions")
session_id = response.json()["session_id"]

# 2. Upload image
with open("face.jpg", "rb") as f:
    files = {"file": ("face.jpg", f, "image/jpeg")}
    response = requests.post(
        f"http://localhost:8000/api/v1/verify/{session_id}/upload",
        files=files
    )

# 3. Run full pipeline
response = requests.post(
    f"http://localhost:8000/api/v1/verify/{session_id}/pipeline"
)
result = response.json()

# 4. Get verification result
response = requests.get(
    f"http://localhost:8000/api/v1/verify/{session_id}/verify"
)
verification = response.json()
```

## Project Structure

```
veriface-backend/
├── veriface/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py              # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py            # Database connection
│   │   ├── schemas.py             # SQLAlchemy models
│   │   └── schemas_api.py         # Pydantic schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── input_acquisition.py   # Stage 1
│   │   ├── biometric_processing.py # Stage 2
│   │   ├── discovery_engine.py    # Stage 3
│   │   ├── candidate_intelligence.py # Stage 4
│   │   ├── confidence_engine.py   # Stage 5
│   │   ├── evidence_vault.py      # Stage 6
│   │   ├── crypto_engine.py       # Stage 7
│   │   ├── blockchain.py          # Stage 8
│   │   ├── verification_engine.py # Stage 9
│   │   └── orchestrator.py        # Pipeline orchestration
│   └── api/
│       ├── __init__.py
│       └── routers/
│           ├── __init__.py
│           └── verify.py          # API endpoints
├── uploads/                        # Uploaded images
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core.py
│   └── test_services.py
├── requirements.txt
├── README.md
└── .env.example
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=veriface
```

## Requirements

- Python 3.10+
- FastAPI
- SQLAlchemy with async support
- face-recognition / dlib
- OpenCV
- NumPy
- Pydantic

## License

MIT License