# VeriFace Chain - Web Application Server
import os
import time
import uuid
import json
import base64
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from pipeline.orchestrator import PipelineOrchestrator, PipelineDiscoveryResult
from trust.verifier import EvidenceVerifier
from trust.blockchain_client import BlockchainClient
from evidence.canonicalizer import canonicalize_evidence
from evidence.hasher import hash_evidence
from core.logging import get_logger

logger = get_logger("webapp.app")

app = FastAPI(
    title="VeriFace Chain",
    description="Decentralized Facial Discovery and Tamper-Proof Digital Evidence Verification",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = settings.OUTPUT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_HISTORY: List[Dict[str, Any]] = []
blockchain_client = BlockchainClient()
verifier = EvidenceVerifier(blockchain_client=blockchain_client)
orchestrator = PipelineOrchestrator(blockchain_client=blockchain_client)


def sanitize_candidate_for_web(candidate_dict: Dict[str, Any]) -> Dict[str, Any]:
    c = dict(candidate_dict)
    img_url = c.get("image_url", "")
    if img_url and not (img_url.startswith("http://") or img_url.startswith("https://") or img_url.startswith("data:")):
        p = Path(img_url)
        v = int(p.stat().st_mtime * 1000) if p.exists() else int(time.time() * 1000)
        c["display_url"] = f"/api/media?path={img_url}&v={v}"
    else:
        c["display_url"] = img_url
    return c


@app.get("/api/health")
async def health_check():
    from trust.blockchain_client import _simulated_evm
    block_num = _simulated_evm.block_number
    total_records = len(_simulated_evm._records)
    return {
        "status": "healthy",
        "pipeline": "VeriFace Chain v2.0",
        "detector_model": "OpenCV Haar Cascade + DNN SSD + Contrast-Region Fallback",
        "embedding_dimensions": 512,
        "embedding_features": "DCT + Sobel + LBP + Color Distribution",
        "search_mode": "Simulated Reverse-Image Search (Demo Mode)",
        "blockchain_provider": "Simulated EVM / Web3 Gateway",
        "chain_id": 1337,
        "block_number": block_num,
        "total_registered_hashes": total_records
    }


@app.get("/api/media")
async def get_media(path: str = Query(..., description="Local path to image")):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    
    mime_type, _ = mimetypes.guess_type(str(p))
    if not mime_type:
        mime_type = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else "application/octet-stream"
    
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return FileResponse(str(p), media_type=mime_type, headers=headers)


class Base64RunRequest(BaseModel):
    image_base64: str
    filename: Optional[str] = "pasted_image.png"
    register_on_chain: Optional[bool] = True
    search_provider: Optional[str] = "mock"


@app.post("/api/pipeline/run-base64")
async def run_pipeline_base64(payload: Base64RunRequest):
    try:
        data = payload.image_base64
        if "," in data:
            data = data.split(",", 1)[1]
        raw_bytes = base64.b64decode(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    ext = ".png"
    if payload.filename and "." in payload.filename:
        ext = Path(payload.filename).suffix
    uid = uuid.uuid4().hex[:8]
    filename = f"{uid}_pasted{ext}"
    target_path = UPLOAD_DIR / filename
    target_path.write_bytes(raw_bytes)

    return await execute_pipeline_on_file(target_path, payload.register_on_chain, provider_name=payload.search_provider)


@app.post("/api/pipeline/run")
async def run_pipeline_upload(
    file: UploadFile = File(...),
    register_on_chain: bool = Form(True),
    search_provider: Optional[str] = Form("mock")
):
    safe_suffix = Path(file.filename).suffix or ".jpg"
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).stem}{safe_suffix}"
    target_path = UPLOAD_DIR / safe_name
    content = await file.read()
    target_path.write_bytes(content)

    return await execute_pipeline_on_file(target_path, register_on_chain, provider_name=search_provider)


@app.post("/api/pipeline/run-sample")
async def run_pipeline_sample(sample_name: str = Query("test_face"), search_provider: str = Query("mock")):
    sample_paths = {
        "test_face": Path("input/test_face.jpg"),
        "candidate_match": settings.FIXTURES_DIR / "candidate_match.jpg"
    }
    
    target = sample_paths.get(sample_name)
    if not target or not target.exists():
        target = Path("input/test_face.jpg")
    
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Sample face not found at {target}")
        
    return await execute_pipeline_on_file(target, True, provider_name=search_provider)


async def execute_pipeline_on_file(file_path: Path, register_on_chain: bool, provider_name: Optional[str] = None):
    try:
        result: PipelineDiscoveryResult = orchestrator.run_pipeline(
            image_path=file_path,
            register_on_chain=register_on_chain,
            search_mode="web_discovery",
            provider_name=provider_name or "mock"
        )
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


    res_dict = result.model_dump()

    candidates_enriched = []
    for c in res_dict.get("candidates", []):
        candidates_enriched.append(sanitize_candidate_for_web(c))
    res_dict["candidates"] = candidates_enriched

    if res_dict.get("best_match"):
        res_dict["best_match"] = sanitize_candidate_for_web(res_dict["best_match"])

    resolved_input_path = str(file_path.resolve())
    res_dict["input_image_display_url"] = f"/api/media?path={resolved_input_path}"

    # Add to history
    PIPELINE_HISTORY.insert(0, {
        "pipeline_id": res_dict.get("pipeline_id"),
        "status": res_dict.get("status"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": file_path.name,
        "similarity": res_dict.get("best_match", {}).get("highest_similarity") if res_dict.get("best_match") else 0.0,
        "evidence_hash": res_dict.get("evidence_hash"),
        "tx_hash": res_dict.get("blockchain_receipt", {}).get("transaction_hash") if res_dict.get("blockchain_receipt") else None,
        "block_number": res_dict.get("blockchain_receipt", {}).get("block_number") if res_dict.get("blockchain_receipt") else None,
        "data": res_dict
    })
    if len(PIPELINE_HISTORY) > 30:
        PIPELINE_HISTORY.pop()

    return res_dict


class VerifyRequest(BaseModel):
    evidence: Optional[Dict[str, Any]] = None
    hash: Optional[str] = None
    expected_hash: Optional[str] = None


@app.post("/api/verify")
async def verify_evidence_endpoint(payload: VerifyRequest):
    if payload.evidence:
        try:
            res = verifier.verify(payload.evidence, expected_hash=payload.expected_hash)
            return {
                "verified": res.verified,
                "evidence_hash": res.evidence_hash,
                "on_chain": res.on_chain,
                "blockchain_timestamp": res.blockchain_timestamp,
                "uploader": res.uploader,
                "verification_state": res.verification_state.value if hasattr(res.verification_state, "value") else str(res.verification_state),
                "reason": res.reason
            }
        except Exception as e:
            return {
                "verified": False,
                "evidence_hash": "",
                "on_chain": False,
                "verification_state": "ERROR",
                "reason": str(e)
            }
    elif payload.hash:
        clean_hash = payload.hash.strip().lower()
        if clean_hash.startswith("0x"):
            clean_hash = clean_hash[2:]
        try:
            exists, ts, uploader = blockchain_client.verify_evidence(clean_hash)
            return {
                "verified": exists,
                "evidence_hash": clean_hash,
                "on_chain": exists,
                "blockchain_timestamp": ts if exists else None,
                "uploader": uploader if exists else None,
                "verification_state": "VERIFIED" if exists else "NOT_REGISTERED",
                "reason": "Cryptographic hash matches immutable blockchain record exactly." if exists else "Hash not found in blockchain smart contract registry."
            }
        except Exception as e:
            return {
                "verified": False,
                "evidence_hash": clean_hash,
                "on_chain": False,
                "verification_state": "ERROR",
                "reason": str(e)
            }
    else:
        raise HTTPException(status_code=400, detail="Provide either evidence object or hash")


@app.post("/api/verify/tamper-demo")
async def tamper_demo_endpoint(payload: Optional[Dict[str, Any]] = None):
    base_evidence = None
    if payload and payload.get("evidence"):
        base_evidence = dict(payload["evidence"])
    elif PIPELINE_HISTORY and PIPELINE_HISTORY[0].get("data", {}).get("evidence_record"):
        base_evidence = dict(PIPELINE_HISTORY[0]["data"]["evidence_record"])
    else:
        base_evidence = {
            "evidence_id": "ev_tamper_demo_001",
            "source": "mock_test_provider",
            "page_url": "https://public-web-archive.org/posts/verified_identity_profile",
            "image_url": "candidate_match.jpg",
            "image_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "input_image_sha256": "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
            "similarity_score": 1.0,
            "match_threshold": 0.75,
            "match_decision": "HIGH_CONFIDENCE_MATCH",
            "model_name": "OpenCV-FaceNet-Sim-512",
            "discovered_at": "2026-09-06T12:00:00Z",
            "pipeline_version": "2.0.0"
        }

    orig_canonical = canonicalize_evidence(base_evidence)
    orig_hash = hash_evidence(base_evidence)

    tampered_evidence = dict(base_evidence)
    tampered_evidence["similarity_score"] = 0.825
    tampered_evidence["tamper_note"] = "Malicious attacker altered score to forge match confidence"

    tampered_canonical = canonicalize_evidence(tampered_evidence)
    tampered_hash = hash_evidence(tampered_evidence)

    orig_check, orig_ts, orig_up = blockchain_client.verify_evidence(orig_hash)
    tamp_check, tamp_ts, tamp_up = blockchain_client.verify_evidence(tampered_hash)

    return {
        "original": {
            "evidence": base_evidence,
            "canonical_json": orig_canonical,
            "sha256_hash": orig_hash,
            "on_chain": orig_check,
            "timestamp": orig_ts
        },
        "tampered": {
            "evidence": tampered_evidence,
            "canonical_json": tampered_canonical,
            "sha256_hash": tampered_hash,
            "on_chain": tamp_check,
            "tamper_explanation": "Field 'similarity_score' modified from 1.0 -> 0.825. SHA-256 hash completely changed due to avalanche effect."
        },
        "verification_result": {
            "tamper_detected": True,
            "hash_mismatch": orig_hash != tampered_hash,
            "tampered_registered_on_chain": tamp_check,
            "status": "TAMPERING_CONFIRMED_BLOCKCHAIN_REJECTED"
        }
    }


@app.get("/api/history")
async def get_history():
    return PIPELINE_HISTORY


@app.get("/api/blockchain/records")
async def get_blockchain_records():
    """Returns all evidence hashes persisted in the blockchain EVM store (survives server restarts)."""
    from trust.blockchain_client import _simulated_evm
    _simulated_evm._load_state()  # Always reload from disk for freshest state
    records = []
    for hash_key, rec in _simulated_evm._records.items():
        import datetime
        ts = rec.get("timestamp", 0)
        try:
            registered_at = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            registered_at = str(ts)
        records.append({
            "evidence_hash": hash_key.replace("0x", ""),
            "hash_0x": hash_key,
            "tx_hash": rec.get("tx_hash", ""),
            "block_number": rec.get("block_number", 0),
            "uploader": rec.get("uploader", ""),
            "registered_at": registered_at,
            "timestamp": ts
        })
    # Sort newest first
    records.sort(key=lambda r: r["timestamp"], reverse=True)
    return {"total": len(records), "records": records}


STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def index():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file), media_type="text/html")
    return HTMLResponse("<h1>VeriFace Chain Backend Running</h1><p>Static UI not yet built.</p>")
