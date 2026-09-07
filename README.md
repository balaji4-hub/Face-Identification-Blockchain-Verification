# VERIFACE CHAIN

A comprehensive facial recognition verification system with blockchain-based provenance.

## ✅ Features

### ✅ Face Recognition
- Uses `face-recognition` library with OpenCV fallback
- 128-dimensional face embeddings
- Face detection, quality assessment, and landmark detection

### ✅ Genuine Web/Social Search
Multi-provider search engine:
- **PimEyes** - Facial recognition search (API key required)
- **Google Vision API** - Cloud-based face detection
- **Bing Visual Search** - Reverse image search
- **Social Media** - Instagram, TikTok, Facebook, Twitter
- **Web Search** - General reverse image search

### ✅ Blockchain Record
- Proof of Work (PoW) consensus algorithm
- SHA-256 evidence hashing
- Timestamped transactions
- Merkle tree proofs
- Chain integrity verification

### ✅ Re-verification
- Chain recalculation for tamper detection
- Tamper detection and reporting
- Audit trail with timestamps
- Support for compliance verification

## Pipeline Stages

```
┌─────────────────────────┐
│    INPUT ACQUISITION    │  → Face Image / Face Scan
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ BIOMETRIC PROCESSING    │  → Face Detection, Quality Check, Embedding
│ ✅ Face Recognition     │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ DISCOVERY ENGINE        │  → ✅ Genuine Web/Social Search
│ ✅ Multi-provider       │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ CANDIDATE INTELLIGENCE  │  → Download Candidates, Face Analysis
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ CONFIDENCE ENGINE       │  → Similarity Scoring, Ranking
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ EVIDENCE VAULT          │  → Evidence JSON, Provenance
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ CRYPTO ENGINE           │  → SHA-256, Merkle Trees
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ BLOCKCHAIN              │  → ✅ PoW, Timestamp, ✅ Blockchain Record
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ VERIFICATION            │  → ✅ Re-verification, Tamper Detection
└─────────────────────────┘
```

## Installation

```bash
cd veriface-backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** `face-recognition` requires CMake and dlib. On Windows, you may need:
1. Visual Studio Build Tools
2. CMake installed separately

For fallback mode without compilation, remove `face-recognition` from requirements.txt.

## Running the Server

```bash
python start_server.py
```

Server runs at: `http://127.0.0.1:8000`

API Documentation: `http://127.0.0.1:8000/docs`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/verify/health` | GET | Health check |
| `/api/v1/verify/sessions` | POST | Create session |
| `/api/v1/verify/{id}/upload` | POST | Upload face image |
| `/api/v1/verify/{id}/pipeline` | POST | Run full pipeline |
| `/api/v1/verify/{id}/discover` | POST | ✅ Web/social search |
| `/api/v1/verify/{id}/blockchain` | POST | ✅ Record on blockchain |
| `/api/v1/verify/{id}/verify` | GET | ✅ Get verification result |
| `/api/v1/verify/{id}/re-verify` | POST | ✅ Re-verify session |
| `/api/v1/verify/{id}/chain` | GET | Get full chain details |

## Usage Example

```python
import httpx

BASE_URL = "http://127.0.0.1:8000"

# 1. Create session
resp = httpx.post(f"{BASE_URL}/api/v1/verify/sessions")
session_id = resp.json()["session_id"]

# 2. Upload face image
with open("face.jpg", "rb") as f:
    resp = httpx.post(
        f"{BASE_URL}/api/v1/verify/{session_id}/upload",
        files={"file": ("face.jpg", f, "image/jpeg")}
    )

# 3. Run full pipeline (all 9 stages)
resp = httpx.post(f"{BASE_URL}/api/v1/verify/{session_id}/pipeline")

# 4. Get verification result
resp = httpx.get(f"{BASE_URL}/api/v1/verify/{session_id}/verify")
print(resp.json())

# 5. Re-verify (audit/compliance)
resp = httpx.post(f"{BASE_URL}/api/v1/verify/{session_id}/re-verify")

# 6. Get blockchain record
resp = httpx.get(f"{BASE_URL}/api/v1/verify/{session_id}/blockchain")
print(resp.json())
```

## Project Structure

```
veriface-backend/
├── veriface/
│   ├── main.py
│   ├── core/config.py
│   ├── models/
│   │   ├── database.py
│   │   ├── schemas.py (8 models)
│   │   └── schemas_api.py
│   ├── services/
│   │   ├── input_acquisition.py
│   │   ├── biometric_processing.py  ✅ Face Recognition
│   │   ├── discovery_engine.py      ✅ Web/Social Search
│   │   ├── candidate_intelligence.py
│   │   ├── confidence_engine.py
│   │   ├── evidence_vault.py
│   │   ├── crypto_engine.py         ✅ SHA-256/Merkle
│   │   ├── blockchain.py            ✅ PoW Blockchain
│   │   ├── verification_engine.py   ✅ Re-verification
│   │   └── orchestrator.py
│   └── api/routers/verify.py
├── tests/
├── requirements.txt
└── README.md
```

## Configuration

Edit `.env` file:

```env
# Blockchain
DIFFICULTY=2
BLOCK_REWARD=1.0

# Search Providers (set API keys to enable)
PIMEYES_API_KEY=
GOOGLE_VISION_KEY=
BING_API_KEY=
```

## Testing

```bash
pytest tests/ -v
```