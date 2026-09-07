"""
VeriFace Chain - Discovery: Secure Candidate Image Fetcher
Safely downloads candidate images with size limits, timeout guards, MIME checks, and SHA-256 hashing.
"""
import io
import hashlib
from pathlib import Path
from typing import Tuple, Optional
import requests
from PIL import Image

from config.settings import settings
from core.exceptions import CandidateFetchError
from core.logging import get_logger

logger = get_logger("discovery.candidate_fetcher")


class CandidateFetcher:
    """Safely retrieves, validates, and fingerprints candidate image files."""
    
    def __init__(
        self,
        max_size_bytes: int = None,
        timeout_seconds: int = None,
        max_redirects: int = None
    ):
        self.max_size_bytes = max_size_bytes or settings.MAX_DOWNLOAD_SIZE_BYTES
        self.timeout_seconds = timeout_seconds or settings.DOWNLOAD_TIMEOUT_SECONDS
        self.max_redirects = max_redirects or settings.MAX_REDIRECTS

    def fetch(self, url_or_path: str) -> Tuple[bytes, str]:
        """
        Retrieves image content from a URL or local file path, validates integrity,
        and computes the cryptographic SHA-256 fingerprint.
        
        Args:
            url_or_path: HTTP/HTTPS URL or local filesystem path.
            
        Returns:
            Tuple of (raw_image_bytes, sha256_hex_digest).
            
        Raises:
            CandidateFetchError: If download fails, exceeds size limit, or fails image verification.
        """
        # Handle local file paths (used in tests and local mock providers)
        if not url_or_path.startswith("http://") and not url_or_path.startswith("https://"):
            local_path = Path(url_or_path)
            if not local_path.exists():
                raise CandidateFetchError(f"Local candidate path does not exist: {url_or_path}")
            try:
                content = local_path.read_bytes()
            except Exception as e:
                raise CandidateFetchError(f"Failed to read local candidate file: {e}") from e
                
            return self._validate_and_hash(content)

        # Handle remote HTTP/HTTPS downloads
        try:
            session = requests.Session()
            session.max_redirects = self.max_redirects
            
            with session.get(
                url_or_path,
                timeout=self.timeout_seconds,
                stream=True,
                headers={"User-Agent": "VeriFaceChain-ResearchBot/1.0"}
            ) as response:
                response.raise_for_status()
                
                # Check Content-Type header if present
                content_type = response.headers.get("Content-Type", "")
                if content_type and not any(t in content_type.lower() for t in ["image/", "application/octet-stream"]):
                    raise CandidateFetchError(f"Invalid Content-Type for candidate image: {content_type}")
                    
                # Stream content up to max size limit
                content_chunks = []
                bytes_read = 0
                for chunk in response.iter_content(chunk_size=65536):
                    bytes_read += len(chunk)
                    if bytes_read > self.max_size_bytes:
                        raise CandidateFetchError(
                            f"Candidate download exceeded max size limit of {self.max_size_bytes} bytes"
                        )
                    content_chunks.append(chunk)
                    
                raw_bytes = b"".join(content_chunks)
                return self._validate_and_hash(raw_bytes)
                
        except CandidateFetchError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch candidate from {url_or_path}: {e}")
            raise CandidateFetchError(f"Network error fetching candidate image: {str(e)}") from e

    def _validate_and_hash(self, raw_bytes: bytes) -> Tuple[bytes, str]:
        """Validates that bytes represent a valid image and calculates SHA-256."""
        if not raw_bytes:
            raise CandidateFetchError("Fetched empty image payload")
            
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            img.verify()
        except Exception as e:
            raise CandidateFetchError(f"Candidate file failed image validation: {e}") from e
            
        image_sha256 = hashlib.sha256(raw_bytes).hexdigest().lower()
        return raw_bytes, image_sha256
