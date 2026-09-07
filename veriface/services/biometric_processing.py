"""
STAGE 2: BIOMETRIC PROCESSING
Handles face detection, quality check, and embedding generation
Uses face-recognition library for accurate face processing
"""
import io
import time
import hashlib
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

# Try face-recognition library first, fallback to OpenCV
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
    """Service for biometric face processing"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.detection_confidence = settings.face_detection_confidence
        self.quality_threshold = settings.quality_threshold
        self._face_cascade = None
        
        if CV2_AVAILABLE:
            self._load_opencv_fallback()
    
    def _load_opencv_fallback(self):
        """Load OpenCV face detector as fallback"""
        try:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception:
            self._face_cascade = None
    
    def detect_faces_face_recognition(self, image_array: np.ndarray) -> Tuple[bool, List[List[int]], List]:
        """
        Detect faces using face-recognition library
        
        Returns:
            Tuple of (face_detected, face_locations, landmarks)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return False, [], []
        
        # Convert to RGB
        if len(image_array.shape) == 2:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGB)
        elif image_array.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_array
        
        # Detect face locations
        face_locations = face_recognition.face_locations(image_rgb)
        face_detected = len(face_locations) > 0
        
        # Get landmarks if faces detected
        face_landmarks = []
        if face_detected:
            all_landmarks = face_recognition.face_landmarks(image_rgb, face_locations)
            face_landmarks = [list(lmarks.values()) for lmarks in all_landmarks]
        
        return face_detected, face_locations, face_landmarks
    
    def detect_faces_opencv(self, image_array: np.ndarray) -> Tuple[bool, List[List[int]], int]:
        """Fallback face detection using OpenCV"""
        if not CV2_AVAILABLE or self._face_cascade is None:
            return False, [], 0
        
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int(settings.min_face_size), int(settings.min_face_size)),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append([y, x + w, y + h, x])
            
            return len(faces) > 0, face_locations, len(faces)
        except Exception as e:
            print(f"OpenCV face detection error: {e}")
            return False, [], 0
    
    def detect_faces(self, image_array: np.ndarray) -> Tuple[bool, List[List[int]], List]:
        """Detect faces - use face-recognition library if available"""
        if FACE_RECOGNITION_AVAILABLE:
            return self.detect_faces_face_recognition(image_array)
        elif CV2_AVAILABLE:
            face_detected, face_locations, face_count = self.detect_faces_opencv(image_array)
            return face_detected, face_locations, []
        else:
            return False, [], []
    
    def calculate_quality_metrics(self, image_array: np.ndarray, face_location: List[int]) -> FaceQualityMetrics:
        """Calculate quality metrics for a detected face"""
        metrics = FaceQualityMetrics()
        
        top, right, bottom, left = face_location
        face_region = image_array[top:bottom, left:right]
        
        if face_region.size == 0:
            return metrics
        
        # Convert to grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_region
        
        # Brightness
        metrics.brightness = np.mean(gray) / 255.0
        
        # Sharpness
        try:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            metrics.sharpness = min(1.0, np.var(laplacian) / 1000.0)
        except Exception:
            metrics.sharpness = 0.5
        
        # Contrast
        metrics.contrast = min(1.0, np.std(gray) / 128.0)
        
        # Overall score
        metrics.overall_score = (
            0.3 * metrics.brightness +
            0.3 * metrics.sharpness +
            0.2 * metrics.contrast +
            0.2 * 0.5
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
    
    def get_face_embedding(self, image_array: np.ndarray, face_location: List[int]) -> Optional[np.ndarray]:
        """
        Generate 128-d face embedding using face-recognition library
        Falls back to hash-based embedding if library not available
        """
        if FACE_RECOGNITION_AVAILABLE:
            try:
                # Convert to RGB
                if len(image_array.shape) == 3:
                    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image_array
                
                encodings = face_recognition.face_encodings(image_rgb, [face_location])
                if len(encodings) > 0:
                    return encodings[0]
            except Exception as e:
                print(f"Face encoding error: {e}")
        
        # Fallback: generate deterministic embedding from image hash and location
        return self._generate_fallback_embedding(image_array, face_location)
    
    def _generate_fallback_embedding(self, image_array: np.ndarray, face_location: List[int]) -> np.ndarray:
        """Generate pseudo-embedding from face region"""
        # Extract face region
        top, right, bottom, left = face_location
        face_region = image_array[top:bottom, left:right]
        
        if face_region.size == 0:
            return np.random.randn(128).astype(np.float64)
        
        # Resize to standard size and flatten
        try:
            face_resized = cv2.resize(face_region, (16, 8))
            face_flat = face_resized.flatten()
            
            # Pad or truncate to 128
            if len(face_flat) < 128:
                face_flat = np.pad(face_flat, (0, 128 - len(face_flat)))
            else:
                face_flat = face_flat[:128]
            
            # Normalize
            norm = np.linalg.norm(face_flat)
            if norm > 0:
                face_flat = face_flat / norm
            
            return face_flat.astype(np.float64)
        except Exception:
            return np.random.randn(128).astype(np.float64)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[float, float]:
        """Calculate similarity and distance between two embeddings"""
        distance = np.linalg.norm(embedding1 - embedding2)
        similarity = 1.0 / (1.0 + distance)
        return float(similarity), float(distance)
    
    async def process_biometrics(
        self, 
        session_id: str, 
        image_path: str,
        image_hash: str = ""
    ) -> BiometricResultResponse:
        """Process biometric data for a verification session"""
        start_time = time.time()
        error_message = None
        
        try:
            # Load image
            if CV2_AVAILABLE:
                image_array = cv2.imread(image_path)
            else:
                img = Image.open(image_path)
                image_array = np.array(img)
            
            if image_array is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Detect faces
            face_detected, face_locations, face_landmarks = self.detect_faces(image_array)
            
            if not face_detected:
                error_message = "No face detected in image"
            
            # Get largest face
            best_face = None
            best_landmarks = None
            
            if face_detected and face_locations:
                max_area = 0
                for i, location in enumerate(face_locations):
                    area = (location[2] - location[0]) * (location[1] - location[3])
                    if area > max_area:
                        max_area = area
                        best_face = location
                        if i < len(face_landmarks):
                            best_landmarks = face_landmarks[i]
                
                if best_face and not self.check_min_face_size(best_face):
                    face_detected = False
                    error_message = f"Face too small (minimum {settings.min_face_size}px required)"
            
            # Calculate quality
            quality_metrics = FaceQualityMetrics()
            if face_detected and best_face:
                quality_metrics = self.calculate_quality_metrics(image_array, best_face)
            
            quality_check_passed = quality_metrics.overall_score >= self.quality_threshold
            
            # Generate embedding
            embedding = None
            embedding_size = 0
            
            if face_detected and best_face:
                embedding = self.get_face_embedding(image_array, best_face)
                if embedding is not None:
                    embedding_size = len(embedding)
                else:
                    error_message = "Failed to generate face embedding"
            
            embedding_bytes = None
            if embedding is not None:
                embedding_bytes = embedding.tobytes()
            
            # Create result
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
                embedding_model="face-recognition" if FACE_RECOGNITION_AVAILABLE else "opencv_fallback",
                embedding_size=embedding_size,
                face_locations=face_locations,
                landmarks=best_landmarks if best_landmarks else None,
                processing_time_ms=processing_time,
                error_message=error_message
            )
            
            self.session.add(biometric_result)
            
            # Update session
            session = (await self.session.execute(
                select(VerificationSession).where(VerificationSession.id == session_id)
            )).scalar_one()
            session.status = "biometric_processed"
            
            await self.session.commit()
            
            # Format response
            formatted_locations = None
            if face_locations:
                formatted_locations = [
                    FaceLocation(top=loc[0], right=loc[1], bottom=loc[2], left=loc[3])
                    for loc in face_locations
                ]
            
            return BiometricResultResponse(
                biometric_id=biometric_id,
                face_detected=face_detected,
                detection_confidence=1.0 if face_detected else 0.0,
                quality_score=quality_metrics.overall_score,
                quality_check_passed=quality_check_passed,
                embedding_size=embedding_size,
                face_count=len(face_locations) if face_detected else 0,
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