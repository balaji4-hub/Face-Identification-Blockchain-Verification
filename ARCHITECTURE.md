# ARCHITECTURE DOCUMENTATION: VeriFace Chain

**System Architecture & Threat Model**  
**Document Version:** 1.0.0  
**Target Audience:** Software Architects, Cryptographers, Academic Evaluators  

---

## 1. High-Level Architectural Blueprint

VeriFace Chain is split into two primary operational nodes:
1. **The Discovery & Intelligence Node** (`intelligence/`, `discovery/`, `evidence/`): Handles computer vision, quality gating, visual search discovery, and candidate intelligence.
2. **The Trust & Verification Node** (`blockchain/`, `trust/`): Handles deterministic canonicalization, cryptographic hashing, smart contract notarization, and independent verification.

```
+---------------------------------------------------------------------------------------+
|                             DISCOVERY & INTELLIGENCE NODE                             |
|                                                                                       |
|   [Input Image]                                                                       |
|         │                                                                             |
|         ▼                                                                             |
|   ┌───────────────┐     ┌──────────────────────┐     ┌────────────────────────────┐   |
|   │ FaceDetector  │ ──> │ FaceQualityAnalyzer  │ ──> │ FaceEncoder (In-Memory 512)│   |
|   └───────────────┘     └──────────────────────┘     └─────────────┬──────────────┘   |
|                                                                    │                  |
|                                                                    ▼                  |
|   ┌────────────────────────┐      ┌─────────────────┐       ┌──────────────┐          |
|   │ CandidateFetcher (Safe)│ <─── │ SearchService   │ <──── │ VisualSearch │          |
|   └───────────┬────────────┘      └─────────────────┘       │ Provider ABC │          |
|               │                                             └──────────────┘          |
|               ▼                                                                       |
|   ┌────────────────────────┐      ┌─────────────────┐                                 |
|   │ Multi-Face Candidate   │ ───> │ CandidateRanker │                                 |
|   │ Intelligence Analyzer  │      │ & FaceMatcher   │                                 |
|   └────────────────────────┘      └────────┬────────┘                                 |
|                                            │                                          |
|                                            ▼                                          |
|                              ┌───────────────────────────┐                            |
|                              │ EvidenceBuilder           │                            |
|                              │ (Zero Biometric Vectors!) │                            |
|                              └─────────────┬─────────────┘                            |
+--------------------------------------------│------------------------------------------+
                                             │ [EvidenceRecord]
+--------------------------------------------│------------------------------------------+
|                        TRUST & VERIFICATION NODE                                      |
|                                            ▼                                          |
|                              ┌───────────────────────────┐                            |
|                              │ Deterministic             │                            |
|                              │ Canonicalizer Engine      │                            |
|                              └─────────────┬─────────────┘                            |
|                                            │ (UTF-8 Canonical Bytes)                  |
|                                            ▼                                          |
|                              ┌───────────────────────────┐                            |
|                              │ SHA-256 Hasher Engine     │                            |
|                              └─────────────┬─────────────┘                            |
|                                            │ (bytes32 Digest)                         |
|                                            ▼                                          |
|                              ┌───────────────────────────┐                            |
|                              │ BlockchainClient (Web3.py)│                            |
|                              └─────────────┬─────────────┘                            |
|                                            │                                          |
|                                            ▼                                          |
|                              ┌───────────────────────────┐                            |
|                              │ EvidenceRegistry.sol      │                            |
|                              │ [Immutable EVM Ledger]    │                            |
|                              └───────────────────────────┘                            |
+---------------------------------------------------------------------------------------+
```

---

## 2. Core Threat Model & Cryptographic Guarantees

### Threat Matrix & Countermeasures

| Threat Scenario | Threat Description | Architectural Mitigation |
| :--- | :--- | :--- |
| **Biometric Database Leak** | An adversary compromises the backend server or reads storage volumes. | **Zero Biometric Persistence:** Raw vectors exist only during execution in volatile RAM. No database or disk files contain embeddings. |
| **Evidence Metadata Tampering** | An attacker alters the similarity score or URL in an evidence JSON. | **Deterministic Canonicalization + SHA-256:** Any modification alters the canonical digest, triggering `TAMPERED_OR_DIFFERENT`. |
| **Key Ordering Attacks** | An adversary reorders JSON dictionary keys to generate hash collisions. | **Sorted Canonicalization:** `canonicalize_evidence()` enforces recursive alphabetical key ordering (`sort_keys=True`). |
| **Backdating Fraud** | An entity claims an evidence match was discovered earlier than it actually was. | **Blockchain Timestamping:** The smart contract assigns block timestamp `block.timestamp` upon registration. |
| **Duplicate Registration** | An adversary attempts to front-run or overwrite an existing registration. | **Smart Contract Revert:** `EvidenceRegistry.sol` reverts with `EvidenceAlreadyRegistered` on collision. |
| **DoS via Massive Candidate Downloads** | A malicious search result returns a 10 GB file or slow stream. | **CandidateFetcher Guardrails:** Strict 15 MB size ceiling, MIME validation, and 10-second connection timeouts. |

---

## 3. The 10-Stage Pipeline Lifecycle

Each stage within `pipeline/orchestrator.py` transition through strict observable states:
`PENDING` $\to$ `RUNNING` $\to$ (`SUCCESS` | `FAILED` | `SKIPPED`).

1. **Stage 1: Input Validation** (`PipelineStage.INPUT_VALIDATION`): Validates file existence, format, dimensions ($\ge 100\times 100$), and generates input image SHA-256.
2. **Stage 2: Face Detection** (`PipelineStage.FACE_DETECTION`): Locates all faces and isolates the primary subject.
3. **Stage 3: Face Quality Analysis** (`PipelineStage.FACE_QUALITY`): Computes Laplacian variance focus score ($\sigma^2$), bounding box ratio, and flags blur or low resolution.
4. **Stage 4: Biometric Face Embedding** (`PipelineStage.FACE_EMBEDDING`): Extracts 512-dimensional $L_2$-normalized vector into volatile RAM.
5. **Stage 5: Visual Search Discovery** (`PipelineStage.VISUAL_SEARCH`): Queries authorized search endpoint or mock fixture provider.
6. **Stage 6: Candidate Intelligence** (`PipelineStage.CANDIDATE_ANALYSIS`): Safely fetches candidate images, runs multi-face detection, extracts candidate embeddings, and performs cosine vector matching against the query embedding.
7. **Stage 7: Confidence Decision** (`PipelineStage.MATCH_DECISION`): Ranks candidates and classifies the match as `HIGH_CONFIDENCE_MATCH` ($\ge 0.75$), `POSSIBLE_MATCH` ($\ge 0.50$), or `NO_MATCH`.
8. **Stage 8: Evidence Generation** (`PipelineStage.EVIDENCE_GENERATION`): Assembles an `EvidenceRecord` without any biometric vectors.
9. **Stage 9: Canonical Hashing** (`PipelineStage.CANONICAL_HASHING`): Canonicalizes JSON structure and calculates SHA-256 fingerprint.
10. **Stage 10: Blockchain Registration** (`PipelineStage.BLOCKCHAIN_REGISTRATION`): Notarizes `bytes32` digest onto `EvidenceRegistry.sol` and writes receipt.

---

## 4. Blockchain Smart Contract Design

The `EvidenceRegistry.sol` contract enforces strict immutability:
```solidity
struct Record {
    bytes32 evidenceHash;
    uint256 timestamp;
    address uploader;
}
mapping(bytes32 => Record) private _records;
```
- **Gas Efficiency:** O(1) storage lookup and insertion.
- **Zero Extraneous Complexity:** No ERC-20 tokens, no NFTs, no governance tokens, and no upgradeable proxies. The contract functions purely as an access-independent integrity registry.
