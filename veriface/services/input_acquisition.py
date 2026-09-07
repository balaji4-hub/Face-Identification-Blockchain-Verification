"""
STAGE 1: INPUT ACQUISITION
Handles face image/scan upload, validation, and preprocessing
"""
import os
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import aiofiles
from PIL import Image

from veriface.core.config import settings
from veriface.models.database import AsyncSession
from veriface.models.schemas import VerificationSession, InputImage
from veriface.models.schemas_api import InputUploadResponse, InputUploadRequest


class InputAcquisitionService:
    """Service for handling face image input acquisition"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def validate_image(self, file_content: bytes, mime_type: str) -> Tuple[bool, str]:
        """
        Validate uploaded image file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if len(file_content) > settings.max_file_size_mb * 1024 * 1024:
            return False, f"File size exceeds maximum of {settings.max_file_size_mb}MB"
        
        # Check mime type
        if mime_type not in settings.allowed_image_types:
            return False, f"File type {mime_type} not allowed. Allowed: {settings.allowed_image_types}"
        
        # Validate image content with PIL
        try:
            img = Image.open(io.BytesIO(file_content))
            img.verify()
            return True, ""
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    def calculate_image_hash(self, file_content: bytes) -> str:
        """Calculate SHA-256 hash of image content"""
        return hashlib.sha256(file_content).hexdigest()
    
    def get_image_dimensions(self, file_content: bytes) -> Tuple[int, int, int]:
        """Get image dimensions and channels"""
        try:
            img = Image.open(io.BytesIO(file_content))
            width, height = img.size
            channels = len(img.getbands()) if img.mode == 'RGB' else 1
            return width, height, channels
        except Exception:
            return 0, 0, 0
    
    async def save_image(self, session_id: str, filename: str, content: bytes) -> str:
        """Save uploaded image to disk"""
        # Create session-specific directory
        session_dir = self.upload_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        file_path = session_dir / safe_filename
        
        # Write file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        return str(file_path)
    
    async def process_upload(
        self, 
        filename: str, 
        content: bytes, 
        mime_type: str,
        request: Optional[InputUploadRequest] = None
    ) -> InputUploadResponse:
        """
        Process face image upload and create verification session
        
        Args:
            filename: Original filename
            content: File content bytes
            mime_type: MIME type of file
            request: Optional request with additional configuration
            
        Returns:
            InputUploadResponse with session and image details
        """
        import io
        
        # Validate image
        is_valid, error_msg = await self.validate_image(content, mime_type)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Create verification session
        session_id = str(uuid.uuid4())
        session = VerificationSession(
            id=session_id,
            status="input_acquired"
        )
        self.session.add(session)
        
        # Calculate hash and dimensions
        image_hash = self.calculate_image_hash(content)
        width, height, channels = self.get_image_dimensions(content)
        
        # Save image to disk
        file_path = await self.save_image(session_id, filename, content)
        file_size = len(content)
        
        # Create input image record
        input_image = InputImage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=mime_type,
            image_hash=image_hash,
            width=width,
            height=height,
            channels=channels,
            metadata=request.metadata if request else None
        )
        self.session.add(input_image)
        
        await self.session.commit()
        
        return InputUploadResponse(
            session_id=session_id,
            input_image_id=input_image.id,
            filename=filename,
            file_size=file_size,
            mime_type=mime_type,
            image_hash=image_hash,
            dimensions={"width": width, "height": height, "channels": channels},
            status="success",
            message="Image uploaded and validated successfully"
        )
    
    async def get_session_input(self, session_id: str) -> Optional[InputImage]:
        """Get input image for a session"""
        result = await self.session.execute(
            select(InputImage).where(InputImage.session_id == session_id)
        )
        return result.scalar_one_or_none()
    
    async def cleanup_session_files(self, session_id: str):
        """Remove uploaded files for a session"""
        session_dir = self.upload_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)


# Helper imports
import uuid
import io
from sqlalchemy import select