"""
STAGE 2: BIOMETRIC PROCESSING
Handles face detection, quality check, and embedding generation
Fallback implementation using OpenCV for face detection
"""
import io
import time
import hashlib
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PIL import Image

from veriface.core.config import settings
from veriface.models.database import AsyncSession
from veriface.models.schemas import VerificationSession, BiometricResult
from veriface.models.schemas_api import BiometricResultResponse, FaceLocation, FaceLandmark


@dataclass
class FaceQualityMetrics:
    """Quality metrics for face detection"""
    brightness: float = 0.0
    sharpness: float = 0.0
    contrast: float = 0.0
    symmetry_score: float = 0.0
    overall_score: float = 0.0


class BiometricProcessingService:
    """Service for biometric face processing - OpenCV fallback implementation"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.detection_confidence = settings.face_detection_confidence
        self.quality_threshold = settings.quality_threshold
        self._face_cascade = None
        self._load_face_detector()
    
    def _load_face_detector(self):
        """Load OpenCV face detector"""
        if CV2_AVAILABLE:
            try:
                # Try to load the default face cascade
                self._face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except Exception:
                self._face_cascade = None
        else:
            self._face_cascade = None
    
    def detect_faces_opencv(self, image_array: np.ndarray) -> Tuple[bool, List[List[int]], int]:
        """
        Detect faces using OpenCV
        
        Returns:
            Tuple of (face_detected, face_locations, face_count)
        """
        if not CV2_AVAILABLE or self._face_cascade is None:
            return False, [], 0
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            # Equalize histogram for better detection
            gray = cv2.equalizeHist(gray)
            
            # Detect faces
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int(settings.min_face_size), int(settings.min_face_size)),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append([y, x + w, y + h, x])  # top, right, bottom, left
            
            return len(faces) > 0, face_locations, len(faces)
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return False, [], 0
    
    def calculate_quality_metrics(self, image_array: np.ndarray, face_location: List[int]) -> FaceQualityMetrics:
        """
        Calculate quality metrics for a detected face
        """
        metrics = FaceQualityMetrics()
        
        # Extract face region
        top, right, bottom, left = face_location
        face_region = image_array[top:bottom, left:right]
        
        if face_region.size == 0:
            return metrics
        
        # Convert to grayscale for analysis
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_region
        
        # Calculate brightness (mean intensity)
        metrics.brightness = np.mean(gray) / 255.0
        
        # Calculate sharpness (Laplacian variance)
        try:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            metrics.sharpness = np.var(laplacian) / 1000.0
            metrics.sharpness = min(1.0, metrics.sharpness)
        except Exception:
            metrics.sharpness = 0.5
        
        # Calculate contrast
        contrast = np.std(gray) / 128.0
        metrics.contrast = min(1.0, contrast)
        
        # Overall quality score (weighted average)
        metrics.overall_score = (
            0.3 * metrics.brightness +
            0.3 * metrics.sharpness +
            0.2 * metrics.contrast +
            0.2 * 0.5  # symmetry placeholder
        )
        
        return metrics
    
    def check_min_face_size(self, face_location: List[int], min_size: int = None) -> bool:
        """Check if detected face meets minimum size requirement"""
        if min_size is None:
            min_size = settings.min_face_size
        
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        
        return face_width >= min_size and face_height >= min_size
    
    def generate_mock_embedding(self, face_location: List[int], image_hash: str) -> np.ndarray:
        """
        Generate a pseudo-embedding based on face features and image hash
        Note: In production, replace with actual face recognition model
        """
        # Create a deterministic but unique embedding from face location
        hash_bytes = hashlib.md5(f"{image_hash}:{face_location}".encode()).digest()
        hash_numbers = [b for b in hash_bytes]
        
        # Expand to 128 dimensions
        embedding = np.array(hash_numbers * 2 + hash_numbers[:64], dtype=np.float64)
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding[:128]
    
    def get_face_embedding(self, image_array: np.ndarray, face_location: List[int], image_hash: str) -> np.ndarray:
        """
        Generate face embedding vector
        
        In production, replace with actual face recognition model
        """
        return self.generate_mock_embedding(face_location, image_hash)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate similarity and distance between two embeddings
        
        Returns:
            Tuple of (similarity_score, euclidean_distance)
        """
        # Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)
        
        # Convert distance to similarity (0-1 range)
        similarity = 1.0 / (1.0 + distance)
        
        return float(similarity), float(distance)
    
    async def process_biometrics(
        self, 
        session_id: str, 
        image_path: str,
        image_hash: str = ""
    ) -> BiometricResultResponse:
        """
        Process biometric data for a verification session
        """
        start_time = time.time()
        error_message = None
        
        try:
            # Load image
            if not CV2_AVAILABLE:
                # Use PIL as fallback
                img = Image.open(image_path)
                image_array = np.array(img)
            else:
                image_array = cv2.imread(image_path)
            
            if image_array is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Detect faces using OpenCV
            face_detected, face_locations, face_count = self.detect_faces_opencv(image_array)
            
            if not face_detected:
                error_message = "No face detected in image"
            
            # Get the largest face if multiple detected
            best_face = None
            
            if face_detected and face_locations:
                # Select largest face
                max_area = 0
                for location in face_locations:
                    area = (location[2] - location[0]) * (location[1] - location[3])
                    if area > max_area:
                        max_area = area
                        best_face = location
                
                # Check minimum size
                if best_face and not self.check_min_face_size(best_face):
                    face_detected = False
                    error_message = f"Face too small (minimum {settings.min_face_size}px required)"
            
            # Calculate quality metrics
            quality_metrics = FaceQualityMetrics()
            if face_detected and best_face:
                quality_metrics = self.calculate_quality_metrics(image_array, best_face)
            
            quality_check_passed = quality_metrics.overall_score >= self.quality_threshold
            
            # Generate embedding
            embedding = None
            embedding_size = 0
            
            if face_detected and best_face:
                embedding = self.get_face_embedding(image_array, best_face, image_hash)
                if embedding is not None:
                    embedding_size = len(embedding)
                else:
                    error_message = "Failed to generate face embedding"
            
            # Convert embedding to binary for storage
            embedding_bytes = None
            if embedding is not None:
                embedding_bytes = embedding.tobytes()
            
            # Create biometric result record
            biometric_id = str(uuid.uuid4())
            processing_time = (time.time() - start_time) * 1000
            
            biometric_result = BiometricResult(
                id=biometric_id,
                session_id=session_id,
                face_detected=face_detected,
                detection_confidence=1.0 if face_detected else 0.0,
                quality_score=quality_metrics.overall_score,
                quality_check_passed=quality_check_passed,
                face_embedding=embedding_bytes,
                embedding_model="opencv_fallback",
                embedding_size=embedding_size,
                face_locations=face_locations,
                landmarks=None,
                processing_time_ms=processing_time,
                error_message=error_message
            )
            
            self.session.add(biometric_result)
            
            # Update session status
            session = (await self.session.execute(
                select(VerificationSession).where(VerificationSession.id == session_id)
            )).scalar_one()
            session.status = "biometric_processed"
            
            await self.session.commit()
            
            # Format face locations for response
            formatted_locations = None
            if face_locations:
                formatted_locations = [
                    FaceLocation(
                        top=loc[0], right=loc[1], bottom=loc[2], left=loc[3]
                    ) for loc in face_locations
                ]
            
            return BiometricResultResponse(
                biometric_id=biometric_id,
                face_detected=face_detected,
                detection_confidence=1.0 if face_detected else 0.0,
                quality_score=quality_metrics.overall_score,
                quality_check_passed=quality_check_passed,
                embedding_size=embedding_size,
                face_count=face_count,
                face_locations=formatted_locations,
                landmarks=None,
                processing_time_ms=processing_time,
                error_message=error_message
            )
            
        except Exception as e:
            await self.session.rollback()
            raise Exception(f"Biometric processing failed: {str(e)}")
    
    async def get_biometric_result(self, session_id: str) -> Optional[BiometricResult]:
        """Get biometric result for a session"""
        result = await self.session.execute(
            select(BiometricResult).where(BiometricResult.session_id == session_id)
        )
        return result.scalar_one_or_none()


# Helper imports
import uuid
from sqlalchemy import select