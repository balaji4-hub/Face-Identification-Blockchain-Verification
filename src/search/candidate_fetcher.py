"""
VeriFace Chain - Phase 5: Candidate Image Download & Validation
==================================================================

Pipeline for this module:
    Candidate URL (from CandidateResult produced by Phase 4)
        |
        v
    Download / Fetch Image (streamed for URLs, or direct copy for local test paths)
        |
        v
    Validate Image (Content-Type header + actual pixel decode)
        |
        v
    Local temporary file + SHA-256 fingerprint
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple

import requests

from src.search.search_provider import CandidateResult

DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_CHUNK_SIZE_BYTES = 64 * 1024  # 64 KB streaming chunks

DEFAULT_ALLOWED_CONTENT_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class DownloadError(Exception):
    """Raised when the image could not be downloaded (network, timeout, HTTP error)."""


class FileTooLargeError(Exception):
    """Raised when the remote file exceeds the configured maximum size."""


class InvalidContentTypeError(Exception):
    """Raised when the response's Content-Type header isn't an accepted image type."""


class InvalidImageContentError(Exception):
    """Raised when the downloaded bytes don't actually decode as a valid image."""


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------
@dataclass
class FetchedCandidate:
    """
    Result of successfully downloading and validating one candidate image.
    """
    candidate: CandidateResult
    local_path: str
    sha256: str
    size_bytes: int
    content_type: str


# ---------------------------------------------------------------------------
# Core fetcher class
# ---------------------------------------------------------------------------
class CandidateFetcher:
    """
    Downloads and validates a single candidate image referenced by a
    CandidateResult, saving it to a temporary local file for downstream processing.
    Seamlessly supports both remote HTTP(S) URLs and local file paths (for testing).
    """

    def __init__(
        self,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        allowed_content_types: Optional[Set[str]] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_file_size_bytes = max_file_size_bytes
        self.allowed_content_types = allowed_content_types or DEFAULT_ALLOWED_CONTENT_TYPES

    def fetch(self, candidate: CandidateResult) -> FetchedCandidate:
        url = candidate.image_url

        # Check if URL is actually a local file path (for local fixtures/testing)
        local_candidate_file = Path(url)
        if local_candidate_file.exists() and local_candidate_file.is_file():
            return self._fetch_local(candidate, local_candidate_file)

        temp_path: Optional[str] = None
        response = None
        try:
            response = self._open_stream(url)
            self._validate_content_type(response)
            temp_path, size_bytes, sha256 = self._stream_to_temp_file(response)
            self._validate_is_image(temp_path)

            return FetchedCandidate(
                candidate=candidate,
                local_path=temp_path,
                sha256=sha256,
                size_bytes=size_bytes,
                content_type=response.headers.get("Content-Type", "unknown"),
            )
        except Exception:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        finally:
            if response is not None:
                response.close()

    def _fetch_local(self, candidate: CandidateResult, file_path: Path) -> FetchedCandidate:
        """Helper for local file paths used in offline testing."""
        size_bytes = file_path.stat().st_size
        if size_bytes > self.max_file_size_bytes:
            raise FileTooLargeError(
                f"Image exceeds maximum allowed size of {self.max_file_size_bytes} bytes."
            )

        fd, temp_path = tempfile.mkstemp(suffix=".img", prefix="veriface_candidate_")
        os.close(fd)
        shutil.copyfile(str(file_path), temp_path)

        self._validate_is_image(temp_path)

        with open(temp_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        return FetchedCandidate(
            candidate=candidate,
            local_path=temp_path,
            sha256=sha256,
            size_bytes=size_bytes,
            content_type="image/jpeg",
        )

    def _open_stream(self, url: str) -> requests.Response:
        try:
            response = requests.get(url, stream=True, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise DownloadError(f"Failed to download image from {url}: {exc}") from exc

        if response.status_code != 200:
            response.close()
            raise DownloadError(
                f"Unexpected HTTP status {response.status_code} fetching {url}."
            )
        return response

    def _validate_content_type(self, response: requests.Response) -> None:
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type not in self.allowed_content_types:
            raise InvalidContentTypeError(
                f"Rejected content type '{content_type or 'unknown'}'. "
                f"Allowed types: {sorted(self.allowed_content_types)}"
            )

    def _stream_to_temp_file(self, response: requests.Response) -> Tuple[str, int, str]:
        hasher = hashlib.sha256()
        total_bytes = 0

        fd, temp_path = tempfile.mkstemp(suffix=".img", prefix="veriface_candidate_")
        try:
            with os.fdopen(fd, "wb") as temp_file:
                for chunk in response.iter_content(chunk_size=DEFAULT_CHUNK_SIZE_BYTES):
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > self.max_file_size_bytes:
                        raise FileTooLargeError(
                            f"Image exceeds maximum allowed size of {self.max_file_size_bytes} bytes."
                        )
                    hasher.update(chunk)
                    temp_file.write(chunk)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        return temp_path, total_bytes, hasher.hexdigest()

    def _validate_is_image(self, path: str) -> None:
        import cv2

        image = cv2.imread(path)
        if image is None:
            raise InvalidImageContentError(
                f"Downloaded content at {path} could not be decoded as a valid image."
            )


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------
def cleanup(fetched: FetchedCandidate) -> None:
    if fetched.local_path and os.path.exists(fetched.local_path):
        os.remove(fetched.local_path)


@contextmanager
def fetched_candidate_image(
    fetcher: CandidateFetcher, candidate: CandidateResult
) -> Iterator[FetchedCandidate]:
    fetched = fetcher.fetch(candidate)
    try:
        yield fetched
    finally:
        cleanup(fetched)


# ---------------------------------------------------------------------------
# Manual test entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.search.candidate_fetcher <image_url_or_path>")
        sys.exit(1)

    image_url = sys.argv[1]
    candidate = CandidateResult(
        source="manual_test",
        page_url=image_url,
        image_url=image_url,
        title="Manual test candidate",
    )

    fetcher = CandidateFetcher()

    try:
        with fetched_candidate_image(fetcher, candidate) as fetched:
            print(f"\nDownloaded candidate image")
            print("-" * 40)
            print(f"Local path   : {fetched.local_path}")
            print(f"SHA-256      : {fetched.sha256}")
            print(f"Size (bytes) : {fetched.size_bytes}")
            print(f"Content-Type : {fetched.content_type}")
            print("-" * 40)
            print("(temporary file will be deleted automatically after this block)")
    except DownloadError as e:
        print(f"[DOWNLOAD ERROR] {e}")
        sys.exit(2)
    except FileTooLargeError as e:
        print(f"[FILE TOO LARGE] {e}")
        sys.exit(3)
    except InvalidContentTypeError as e:
        print(f"[INVALID CONTENT TYPE] {e}")
        sys.exit(4)
    except InvalidImageContentError as e:
        print(f"[INVALID IMAGE CONTENT] {e}")
        sys.exit(5)


if __name__ == "__main__":
    main()
