# PROJECT SPECIFICATION: VeriFace Chain

**Title:** Privacy-Aware Face Discovery and Tamper-Evident Digital Evidence Verification System  
**Version:** 1.0.0 (Academic & Production Grade Specification)  
**Author:** Lead Architecture & Engineering Team  
**Date:** September 2026  

---

## 1. Executive Summary & Core Mission

**VeriFace Chain** is an end-to-end, privacy-preserving digital forensics and evidence notarization system. The system enables lawful, consented discovery of public face appearances across online media sources, generates an immutable cryptographic provenance record, and notarizes that digital evidence onto an Ethereum-compatible blockchain registry.

### Core Problem Statement
Digital media authentication faces two compounding crises:
1. **Biometric Privacy Violation:** Traditional facial recognition databases persist raw, sensitive facial feature embeddings, exposing individuals to permanent identity compromise if breached.
2. **Evidence Fragility & Tampering:** Digital discovery findings (e.g. matching social media posts, URLs, similarity scores, capture timestamps) can easily be manipulated, backdated, or fabricated without leaving a cryptographic audit trail.

### VeriFace Chain Solution
VeriFace Chain enforces a **Zero-Biometric-Persistence Architecture**:
- Face recognition and embedding extraction happen strictly in volatile memory ($L_2$-normalized 512-dimensional feature representations).
- Embeddings are destroyed immediately after candidate similarity ranking.
- Only non-biometric evidence metadata (discovered URL, candidate image SHA-256 fingerprint, model architecture, timestamp, and match decision) is serialized into a canonical JSON representation.
- The canonical representation is fingerprinted via SHA-256 and notarized as a `bytes32` digest onto a decentralized smart contract (`EvidenceRegistry.sol`).
- Any third-party verifier can independently recompute the canonical fingerprint and verify provenance against immutable block timestamps without needing access to private biometric databases.

---

## 2. System Pipeline Architecture

```
                  ┌─────────────────────────┐
                  │    INPUT ACQUISITION    │  → Validated JPEG/PNG/WebP
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │  BIOMETRIC PROCESSING   │  → Multi-Face Detection (OpenCV/DNN)
                  │                         │  → Face Quality Scoring (Laplacian Blur & Ratio)
                  │                         │  → Transient Feature Embedding (512-d)
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │    DISCOVERY ENGINE     │  → Dynamic Visual Search
                  │                         │  → Provider Abstraction (Authorized API & Mock)
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ CANDIDATE INTELLIGENCE  │  → Safe Download (Size limit, MIME, SHA-256)
                  │                         │  → Multi-Face Detection on Candidates
                  │                         │  → Cosine Similarity Vector Comparison
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │    CONFIDENCE ENGINE    │  → Similarity Ranking & Decision Classifier
                  │                         │    [HIGH_CONFIDENCE, POSSIBLE, NO_MATCH]
                  └────────────┬────────────┘
                               │
                        Match Found?
                       /            \
                     NO              YES
                     │                │
                     ▼                ▼
                Audit Log     ┌─────────────────┐
                              │ EVIDENCE VAULT  │  → EvidenceRecord Model (Zero Biometrics)
                              │                 │  → Provenance Step Tracking
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  CRYPTO ENGINE  │  → Deterministic Canonical JSON
                              │                 │  → SHA-256 Cryptographic Digest
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   BLOCKCHAIN    │  → EvidenceRegistry.sol (EVM)
                              │                 │  → Timestamp, Uploader, Tx Receipt
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  VERIFICATION   │  → Independent Hash Recomputation
                              │                 │  → On-Chain Query & Tamper Detection
                              └─────────────────┘
```

---

## 3. Detailed Subsystem Specifications

### 3.1 Input Acquisition & Validation Subsystem
- **Supported Formats:** JPEG, PNG, WebP.
- **Constraints:**
  - File Size: Maximum 50 MB (`MAX_FILE_SIZE_MB`).
  - Resolution: Minimum $100 \times 100$ pixels (`MIN_IMAGE_WIDTH`, `MIN_IMAGE_HEIGHT`).
- **Integrity Validation:** SHA-256 fingerprint generated upon arrival; corrupt and malformed headers rejected prior to computer vision pipeline entry.

### 3.2 Face Intelligence Subsystem (`intelligence/`)
- **Face Detection (`face_detector.py`):**
  - OpenCV Deep Neural Network (DNN) SSD face detector with fallback to Haar cascade and PIL.
  - Multi-face support returning `List[FaceDetection]` with bounding boxes $[top, right, bottom, left]$, confidence scores, and facial landmark coordinates.
- **Face Quality Engine (`face_quality.py`):**
  - **Laplacian Variance Blur Estimation:** Focus metric $\sigma^2 = \text{Var}(\nabla^2 I_{gray})$.
  - **Geometric Ratio:** Ratio of face area relative to total image area.
  - **Resolution Filter:** Minimum $100 \times 100$ px face crop.
  - **Outcome:** `FaceQualityResult` (`quality_score`, `accepted: bool`, `issues: List[str]`).
- **Face Encoder & Transient Biometrics (`face_encoder.py`):**
  - Computes 512-dimensional unit $L_2$-normalized biometric vector ($\|v\|_2 = 1.0$).
  - **Strict Privacy Rule:** In-memory execution only. Zero persistence to disk, database, or network logs.
- **Face Matcher & Decision Classifier (`face_matcher.py`, `candidate_ranker.py`):**
  - Cosine similarity: $S(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v$.
  - Decision state configuration:
    - $S \ge 0.75 \implies$ `HIGH_CONFIDENCE_MATCH`
    - $0.50 \le S < 0.75 \implies$ `POSSIBLE_MATCH`
    - $S < 0.50 \implies$ `NO_MATCH`

### 3.3 Dynamic Discovery Subsystem (`discovery/`)
- **Provider-Agnostic Interface (`provider_base.py`):**
  - Abstract base class `VisualSearchProvider` exposing `search(image_path: Path) -> List[CandidateResult]`.
- **Implementations (`provider_registry.py`):**
  - `MockVisualSearchProvider`: Offline fixture-based testing provider.
  - `AuthorizedSearchProvider`: Authenticated external reverse-image API endpoint (`AUTHORIZED_SEARCH_ENDPOINT`, `AUTHORIZED_SEARCH_API_KEY`).
  - **Ethics standard:** Strict adherence to terms of service. Never bypasses CAPTCHA, authentication, or private account permissions.
- **Safe Candidate Fetcher (`candidate_fetcher.py`):**
  - Configurable download timeouts (10s), maximum content size (15 MB), MIME type validation, and streaming SHA-256 image fingerprinting.

### 3.4 Evidence Provenance & Cryptography (`evidence/`)
- **Evidence Model (`evidence_models.py`):**
  - `EvidenceRecord`: `evidence_id`, `source`, `page_url`, `image_url`, `image_sha256`, `input_image_sha256`, `similarity_score`, `match_threshold`, `match_decision`, `model_name`, `discovered_at`, `pipeline_version`.
  - **Zero Biometric Fields:** No embedding vectors or biometric raw points are recorded.
- **Single Source of Truth Canonicalization (`canonicalizer.py`):**
  - Deterministic serialization:
    ```python
    json.dumps(evidence_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ```
- **Cryptographic Hashing (`hasher.py`):**
  - Standard SHA-256 digest outputting 64-character lowercase hex string and Solidity-compatible `bytes32`.

### 3.5 Blockchain & Trust Subsystem (`blockchain/`, `trust/`)
- **Smart Contract (`blockchain/contracts/EvidenceRegistry.sol`):**
  - Implements immutable on-chain mapping: `mapping(bytes32 => Record) private _records;`.
  - Methods:
    - `registerEvidence(bytes32 evidenceHash)`: Prevents zero-hash and duplicate registrations; emits `EvidenceRegistered`.
    - `verifyEvidence(bytes32 evidenceHash)`: View function returning `(bool exists, uint256 timestamp, address uploader)`.
    - `getEvidence(bytes32 evidenceHash)`: View function returning full struct or reverting with `EvidenceNotFound`.
- **Blockchain Client (`trust/blockchain_client.py`):**
  - Web3.py client with automatic failover to deterministic local EVM simulation when external RPC is offline.
- **Independent Verifier & Tamper Detection (`trust/verifier.py`):**
  - Canonicalizes input evidence, recalculates SHA-256 digest, and queries smart contract state.
  - States: `VERIFIED`, `NOT_REGISTERED`, `TAMPERED_OR_DIFFERENT`, `BLOCKCHAIN_UNAVAILABLE`, `INVALID_EVIDENCE`.

---

## 4. Operational Requirements & CLI Interface

The system is fully operable via standard command line:
1. `py -m cli.main run --image input/test_face.jpg`: Full end-to-end pipeline run with evidence generation and on-chain registration.
2. `py -m cli.main analyze --image input/test_face.jpg`: Computer vision and discovery intelligence execution only.
3. `py -m cli.main register --evidence output/evidence.json`: Notarizes an existing evidence JSON on the blockchain.
4. `py -m cli.main verify --evidence output/evidence.json`: Independently verifies an evidence record against the blockchain.
5. `py -m cli.main demo-tamper`: Live interactive demonstration showing authentic evidence verification followed by detection of a single-field modification.
