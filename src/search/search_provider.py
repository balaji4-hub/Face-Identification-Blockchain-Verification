"""
VeriFace Chain - Phase 4: Visual / Reverse-Image Search Provider Abstraction
==============================================================================
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class InvalidImageError(Exception):
    """Raised when the given image path does not exist or isn't a file."""


class ProviderConfigurationError(Exception):
    """Raised when required provider configuration (e.g. API key) is missing."""


class SearchProviderError(Exception):
    """Raised when a search request fails (network error, auth failure, bad response)."""


# ---------------------------------------------------------------------------
# Standardized result model
# ---------------------------------------------------------------------------
@dataclass
class CandidateResult:
    """
    A single, standardized search result, regardless of which provider
    produced it.
    """
    source: str
    page_url: str
    image_url: str
    title: Optional[str] = None
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class VisualSearchProvider(ABC):
    """
    Abstract interface every visual search provider must implement.
    """

    @abstractmethod
    def search(self, image_path: str) -> List[CandidateResult]:
        raise NotImplementedError

    def _validate_image_path(self, image_path: str) -> None:
        """Shared validation so every provider rejects bad paths the same way."""
        if not image_path or not os.path.isfile(image_path):
            raise InvalidImageError(f"Image path does not exist: {image_path}")


# ---------------------------------------------------------------------------
# Real provider (template for an authorized third-party API)
# ---------------------------------------------------------------------------
class RealVisualSearchProvider(VisualSearchProvider):
    """
    Calls an authorized, legitimate visual/reverse-image search API.
    """

    def __init__(
        self,
        api_key_env_var: str = "VERIFACE_SEARCH_API_KEY",
        endpoint_env_var: str = "VERIFACE_SEARCH_ENDPOINT",
        timeout_seconds: int = 15,
    ) -> None:
        self.api_key = os.environ.get(api_key_env_var)
        self.endpoint = os.environ.get(endpoint_env_var)
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise ProviderConfigurationError(
                f"Missing API key. Set the '{api_key_env_var}' environment "
                "variable to your authorized search provider's API key."
            )
        if not self.endpoint:
            raise ProviderConfigurationError(
                f"Missing API endpoint. Set the '{endpoint_env_var}' environment "
                "variable to your authorized search provider's API URL."
            )

    def search(self, image_path: str) -> List[CandidateResult]:
        self._validate_image_path(image_path)

        try:
            with open(image_path, "rb") as image_file:
                response = requests.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"image": image_file},
                    timeout=self.timeout_seconds,
                )
        except requests.RequestException as exc:
            raise SearchProviderError(f"Visual search request failed: {exc}") from exc

        if response.status_code in (401, 403):
            raise SearchProviderError(
                f"Authentication with the search provider failed "
                f"(HTTP {response.status_code}). Check your API key and permissions."
            )
        if response.status_code != 200:
            raise SearchProviderError(
                f"Search provider returned an unexpected status: "
                f"HTTP {response.status_code} - {response.text[:200]}"
            )

        try:
            payload: Dict[str, Any] = response.json()
        except ValueError as exc:
            raise SearchProviderError(f"Search provider returned invalid JSON: {exc}") from exc

        return self._parse_results(payload)

    def _parse_results(self, payload: Dict[str, Any]) -> List[CandidateResult]:
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise SearchProviderError("Unexpected response shape: 'results' is not a list.")

        candidates: List[CandidateResult] = []
        for item in raw_results:
            try:
                candidates.append(
                    CandidateResult(
                        source="authorized_api",
                        page_url=item["page_url"],
                        image_url=item["image_url"],
                        title=item.get("title"),
                    )
                )
            except KeyError as exc:
                raise SearchProviderError(
                    f"Search result missing required field: {exc}"
                ) from exc

        return candidates


# ---------------------------------------------------------------------------
# Mock provider (testing only)
# ---------------------------------------------------------------------------
class MockVisualSearchProvider(VisualSearchProvider):
    """
    Deterministic, offline mock provider for automated tests and local development.
    """

    def __init__(self, canned_results: Optional[List[CandidateResult]] = None) -> None:
        self._canned_results = (
            canned_results if canned_results is not None else self._default_results()
        )

    @staticmethod
    def _default_results() -> List[CandidateResult]:
        from pathlib import Path
        fixture_dir = Path("tests/fixtures").resolve()
        c1 = str((fixture_dir / "candidate_match.jpg").resolve())
        c2 = str((fixture_dir / "candidate_nonmatch.jpg").resolve())
        return [
            CandidateResult(
                source="mock",
                page_url="https://example-test-site.local/post/1",
                image_url=c1,
                title="Mock candidate match (local fixture)",
            ),
            CandidateResult(
                source="mock",
                page_url="https://example-test-site.local/post/2",
                image_url=c2,
                title="Mock candidate non-match (local fixture)",
            ),
        ]

    def search(self, image_path: str) -> List[CandidateResult]:
        self._validate_image_path(image_path)
        return list(self._canned_results)


# ---------------------------------------------------------------------------
# Google Lens / SerpApi Live Search Provider
# ---------------------------------------------------------------------------
class GoogleLensSearchProvider(VisualSearchProvider):
    """
    Live Reverse Image Search Provider using Google Lens via SerpApi.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")

    def search(self, image_path: str) -> List[CandidateResult]:
        self._validate_image_path(image_path)
        if not self.api_key:
            raise ProviderConfigurationError(
                "Missing SerpApi API Key. Set SERPAPI_API_KEY environment variable to enable live Google Lens search."
            )

        try:
            params = {
                "engine": "google_lens",
                "api_key": self.api_key,
                "hl": "en"
            }
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                response = requests.post("https://serpapi.com/search.json", params=params, files=files, timeout=15)

            if response.status_code != 200:
                raise SearchProviderError(f"Google Lens API returned HTTP {response.status_code}")

            data = response.json()
            candidates: List[CandidateResult] = []
            for item in data.get("visual_matches", [])[:15]:
                img_url = item.get("thumbnail") or item.get("original")
                page_url = item.get("link") or item.get("source")
                if img_url and page_url:
                    candidates.append(CandidateResult(
                        source="google_lens_live",
                        page_url=page_url,
                        image_url=img_url,
                        title=item.get("title", "Google Lens Visual Match")
                    ))
            return candidates
        except Exception as exc:
            raise SearchProviderError(f"Google Lens reverse search failed: {exc}") from exc



# ---------------------------------------------------------------------------
# Manual test entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.search.search_provider <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    provider: VisualSearchProvider = MockVisualSearchProvider()

    try:
        results = provider.search(image_path)
    except InvalidImageError as e:
        print(f"[INVALID IMAGE ERROR] {e}")
        sys.exit(2)
    except SearchProviderError as e:
        print(f"[SEARCH ERROR] {e}")
        sys.exit(3)

    print(f"\nCandidate results: {len(results)}")
    print("-" * 40)
    for candidate in results:
        print(f"Source       : {candidate.source}")
        print(f"Page URL     : {candidate.page_url}")
        print(f"Image URL    : {candidate.image_url}")
        print(f"Title        : {candidate.title}")
        print(f"Discovered at: {candidate.discovered_at.isoformat()}")
        print("-" * 40)


if __name__ == "__main__":
    main()
