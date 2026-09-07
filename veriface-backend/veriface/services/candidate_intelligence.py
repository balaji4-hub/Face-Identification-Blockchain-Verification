"""
STAGE 4: CANDIDATE INTELLIGENCE
Handles downloading candidates, face detection, multi-face analysis, and embedding comparison
"""
import io
import time
import hashlib
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import aiofiles

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PIL import Image

from veriface.core.config import settings
from veriface.models.database import AsyncSession
from veriface.models.schemas import VerificationSession, Candidate
from veriface.models.schemas_api import CandidateAnalysisResponse, CandidateInfo


@dataclass
class CandidateAnalysis:
    """Analysis result for a candidate image"""
    candidate_id: str
    face_detected: bool
    face_count: int
    detection_confidence: float
    embedding: Optional[np.ndarray]
    similarity_score: float
    distance: float
    error: Optional[str] = None


class CandidateIntelligenceService:
    """Service for analyzing and comparing candidate matches"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.upload_dir = Path(settings.upload_dir)
        self.processing_cache: Dict[str, bytes] = {}
    
    async def download_candidate_image(self, candidate: Candidate) -> Optional[bytes]:
        """
        Download candidate image from URL or local path
        
        Returns:
            Image bytes or None if download failed
        """
        image_data = None
        
        # Try to download from URL first
        if candidate.candidate_image_url:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=settings.download_timeout) as client:
                    response = await client.get(candidate.candidate_image_url)
                    if response.status_code == 200:
                        image_data = response.content
            except Exception as e:
                print(f"Download URL error: {str(e)}")
        
        # Try local file path
        if image_data is None and candidate.candidate_image_path:
            try:
                path = Path(candidate.candidate_image_path)
                if path.exists():
                    async with aiofiles.open(path, 'rb') as f:
                        image_data = await f.read()
            except Exception as e:
                print(f"Read local file error: {str(e)}")
        
        return image_data
    
    async def detect_faces_in_candidate(self, image_data: bytes) -> Tuple[bool, int, float, List[List[int]]]:
        """
        Detect faces in candidate image
        
        Returns:
            Tuple of (face_detected, face_count, confidence, face_locations)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return False, 0, 0.0, []
        
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            # Convert to RGB if needed
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGB)
            elif image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(image_array)
            face_count = len(face_locations)
            face_detected = face_count > 0
            
            # Confidence is based on face size and quality
            confidence = 0.0
            if face_count > 0:
                avg_area = np.mean([
                    (loc[2] - loc[0]) * (loc[1] - loc[3])
                    for loc in face_locations
                ])
                # Normalize confidence based on typical face size
                confidence = min(1.0, avg_area / 10000)
            
            return face_detected, face_count, confidence, face_locations
            
        except Exception as e:
            print(f"Face detection error: {str(e)}")
            return False, 0, 0.0, []
    
    def get_face_embedding(self, image_array: np.ndarray, face_location: List[int]) -> Optional[np.ndarray]:
        """Get face embedding for candidate image"""
        if not FACE_RECOGNITION_AVAILABLE:
            return None
        
        try:
            encodings = face_recognition.face_encodings(image_array, [face_location])
            if len(encodings) > 0:
                return encodings[0]
        except Exception as e:
            print(f"Embedding error: {str(e)}")
        
        return None
    
    def calculate_embedding_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate similarity between two embeddings
        
        Returns:
            Tuple of (similarity_score, euclidean_distance)
        """
        # Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)
        
        # Convert to similarity (dlib uses 0.6 as threshold)
        similarity = 1.0 / (1.0 + distance)
        
        return float(similarity), float(distance)
    
    async def analyze_candidate(
        self,
        candidate: Candidate,
        query_embedding: np.ndarray
    ) -> CandidateAnalysis:
        """
        Analyze a single candidate image
        
        Args:
            candidate: Candidate database record
            query_embedding: Embedding from query face
            
        Returns:
            CandidateAnalysis result
        """
        start_time = time.time()
        
        try:
            # Download image
            image_data = await self.download_candidate_image(candidate)
            
            if image_data is None:
                return CandidateAnalysis(
                    candidate_id=candidate.id,
                    face_detected=False,
                    face_count=0,
                    detection_confidence=0.0,
                    embedding=None,
                    similarity_score=0.0,
                    distance=1.0,
                    error="Could not download candidate image"
                )
            
            # Detect faces
            face_detected, face_count, detection_confidence, face_locations = \
                await self.detect_faces_in_candidate(image_data)
            
            if not face_detected or face_count == 0:
                return CandidateAnalysis(
                    candidate_id=candidate.id,
                    face_detected=False,
                    face_count=0,
                    detection_confidence=0.0,
                    embedding=None,
                    similarity_score=0.0,
                    distance=1.0,
                    error="No face detected in candidate image"
                )
            
            # Get largest face embedding
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            # Convert color format
            if len(image_array.shape) == 3:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_array
            
            # Find largest face
            largest_face = max(face_locations, key=lambda loc: 
                (loc[2] - loc[0]) * (loc[1] - loc[3]))
            
            # Get embedding
            candidate_embedding = self.get_face_embedding(image_rgb, largest_face)
            
            if candidate_embedding is None:
                return CandidateAnalysis(
                    candidate_id=candidate.id,
                    face_detected=True,
                    face_count=face_count,
                    detection_confidence=detection_confidence,
                    embedding=None,
                    similarity_score=0.0,
                    distance=1.0,
                    error="Could not generate embedding"
                )
            
            # Calculate similarity
            similarity, distance = self.calculate_embedding_similarity(
                query_embedding, candidate_embedding
            )
            
            return CandidateAnalysis(
                candidate_id=candidate.id,
                face_detected=True,
                face_count=face_count,
                detection_confidence=detection_confidence,
                embedding=candidate_embedding,
                similarity_score=similarity,
                distance=distance
            )
            
        except Exception as e:
            return CandidateAnalysis(
                candidate_id=candidate.id,
                face_detected=False,
                face_count=0,
                detection_confidence=0.0,
                embedding=None,
                similarity_score=0.0,
                distance=1.0,
                error=str(e)
            )
    
    async def analyze_all_candidates(
        self,
        session_id: str,
        query_embedding: np.ndarray
    ) -> CandidateAnalysisResponse:
        """
        Analyze all candidates for a session and update rankings
        
        Args:
            session_id: Verification session ID
            query_embedding: Embedding from query face
            
        Returns:
            CandidateAnalysisResponse with analysis results
        """
        start_time = time.time()
        
        # Get all candidates for session
        result = await self.session.execute(
            select(Candidate).where(Candidate.session_id == session_id)
        )
        candidates = result.scalars().all()
        
        if not candidates:
            return CandidateAnalysisResponse(
                session_id=session_id,
                total_candidates=0,
                analyzed_candidates=0,
                candidates=[],
                analysis_time_ms=(time.time() - start_time) * 1000
            )
        
        # Analyze each candidate
        analyses: List[CandidateAnalysis] = []
        
        for candidate in candidates:
            analysis = await self.analyze_candidate(candidate, query_embedding)
            analyses.append(analysis)
        
        # Sort by similarity
        analyses.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update candidate records and get best match
        best_match = None
        candidate_infos = []
        
        for rank, analysis in enumerate(analyses):
            # Update database record
            candidate = next(c for c in candidates if c.id == analysis.candidate_id)
            candidate.face_detected = analysis.face_detected
            candidate.face_count = analysis.face_count
            candidate.detection_confidence = analysis.detection_confidence
            candidate.similarity_score = analysis.similarity_score
            candidate.distance = analysis.distance
            candidate.rank = rank + 1
            candidate.is_top_match = rank == 0
            
            if analysis.embedding is not None:
                candidate.embedding = analysis.embedding.tobytes()
            
            # Create response info
            info = CandidateInfo(
                candidate_id=analysis.candidate_id,
                provider_name=candidate.provider_name,
                external_id=candidate.external_id,
                image_url=candidate.candidate_image_url,
                face_detected=analysis.face_detected,
                face_count=analysis.face_count,
                detection_confidence=analysis.detection_confidence,
                similarity_score=analysis.similarity_score,
                distance=analysis.distance,
                rank=rank + 1,
                is_top_match=rank == 0,
                metadata=candidate.extra_metadata
            )
            candidate_infos.append(info)
            
            if rank == 0 and analysis.face_detected:
                best_match = info
        
        # Update session status
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one()
        session.status = "candidates_analyzed"
        
        await self.session.commit()
        
        analysis_time = (time.time() - start_time) * 1000
        
        return CandidateAnalysisResponse(
            session_id=session_id,
            total_candidates=len(candidates),
            analyzed_candidates=len([a for a in analyses if a.face_detected]),
            candidates=candidate_infos,
            best_match=best_match,
            analysis_time_ms=analysis_time
        )
    
    async def get_candidates(self, session_id: str) -> List[Candidate]:
        """Get all candidates for a session"""
        result = await self.session.execute(
            select(Candidate).where(Candidate.session_id == session_id)
            .order_by(Candidate.rank)
        )
        return result.scalars().all()


# Helper imports
from sqlalchemy import select