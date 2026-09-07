"""
VeriFace Chain - Discovery: Provider Registry & Implementations
Includes MockVisualSearchProvider (labeled DEMO/SIMULATION) and AuthorizedSearchProvider.
"""
from pathlib import Path
from typing import List, Dict, Type, Optional
import requests

from config.settings import settings
from core.exceptions import SearchProviderError
from core.logging import get_logger
from discovery.provider_base import VisualSearchProvider
from evidence.evidence_models import CandidateResult

logger = get_logger("discovery.provider_registry")


class MockVisualSearchProvider(VisualSearchProvider):
    """
    DEMO / SIMULATION MODE — Simulated reverse-image & social media search provider.
    
    In a production deployment this would be replaced by an authorized API call to
    a real reverse-image search engine (e.g. Google Vision API, AWS Rekognition, etc.)
    
    Simulation behaviour:
      - Candidate 1: A web-compressed crop of the uploaded image itself,
                     simulating finding the person's own photo online.
      - Candidate 2 & 3: Fixture images of different people (non-matches).
    """

    PROVIDER_LABEL = "Simulated Reverse-Image Search (Demo Mode)"

    @property
    def name(self) -> str:
        return "mock"

    def search(self, image_path: Path, search_mode: str = "web_discovery", **kwargs) -> List[CandidateResult]:
        logger.info(
            f"[SIMULATED-SEARCH] Executing demo reverse-image search — "
            f"mode: {search_mode} | image: {image_path.name}"
        )

        fixture_dir = settings.FIXTURES_DIR
        candidate_2_path = fixture_dir / "candidate_nonmatch.jpg"
        candidate_3_path = fixture_dir / "candidate_group.jpg"

        # ── Candidate 1: Derive from uploaded image ────────────────────────────
        # Simulates a reverse-image search engine finding this person's own
        # publicly-available web appearance (slightly compressed, margins trimmed).
        candidate_1_img: str
        if search_mode == "web_discovery" and image_path.exists():
            try:
                import cv2
                src = cv2.imread(str(image_path))
                if src is not None:
                    h, w = src.shape[:2]
                    # 1% margin trim — simulates web re-encoding with minor crop
                    crop = src[int(h * 0.01):int(h * 0.99), int(w * 0.01):int(w * 0.99)]
                    disc_name = f"discovered_web_{image_path.stem[:12]}.jpg"
                    disc_path = settings.OUTPUT_DIR / disc_name
                    # 88% JPEG quality — typical social/web image compression
                    cv2.imwrite(str(disc_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    candidate_1_img = str(disc_path)
                    logger.info(f"[SIMULATED-SEARCH] Created discovered web appearance: {disc_name}")
                else:
                    candidate_1_img = str(fixture_dir / "candidate_match.jpg")
            except Exception as ex:
                logger.warning(f"[SIMULATED-SEARCH] Could not create web crop: {ex}")
                candidate_1_img = str(fixture_dir / "candidate_match.jpg")
        else:
            candidate_1_img = str(fixture_dir / "candidate_match.jpg")

        results = [
            CandidateResult(
                source="simulated_web_archive",
                page_url="https://demo.veriface-chain.local/simulated-search/result/1",
                image_url=candidate_1_img,
                title="[SIMULATED] Web Appearance — Reverse-Image Match Found"
            ),
            CandidateResult(
                source="simulated_social_media",
                page_url="https://demo.veriface-chain.local/simulated-search/result/2",
                image_url=str(candidate_2_path) if candidate_2_path.exists() else candidate_1_img,
                title="[SIMULATED] Social Profile — No Match"
            ),
            CandidateResult(
                source="simulated_public_records",
                page_url="https://demo.veriface-chain.local/simulated-search/result/3",
                image_url=str(candidate_3_path) if candidate_3_path.exists() else candidate_1_img,
                title="[SIMULATED] Public Event Photo — Group Candidate"
            ),
        ]
        return results


class AuthorizedSearchProvider(VisualSearchProvider):
    """
    PRODUCTION / REAL: Authorized visual search engine provider.
    Communicates with legitimate visual search or reverse-image APIs using environment credentials.
    Does NOT bypass CAPTCHAs, scrape private accounts, or violate terms of service.
    """

    def __init__(self, api_key: str = None, endpoint: str = None):
        self.api_key = api_key or settings.AUTHORIZED_SEARCH_API_KEY
        self.endpoint = endpoint or settings.AUTHORIZED_SEARCH_ENDPOINT

    @property
    def name(self) -> str:
        return "authorized_api"

    def search(self, image_path: Path, search_mode: str = "web_discovery", **kwargs) -> List[CandidateResult]:
        if not self.endpoint:
            raise SearchProviderError("Authorized search endpoint is not configured in settings")

        logger.info(f"Executing authorized visual search query to: {self.endpoint}")
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with open(image_path, "rb") as img_file:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    files={"image": img_file},
                    params={"mode": search_mode},
                    timeout=settings.SEARCH_TIMEOUT_SECONDS
                )
            response.raise_for_status()
            data = response.json()

            candidates: List[CandidateResult] = []
            for item in data.get("candidates", []):
                candidates.append(CandidateResult(
                    source=item.get("source", "authorized_api"),
                    page_url=item.get("page_url", ""),
                    image_url=item.get("image_url", ""),
                    title=item.get("title")
                ))
            return candidates
        except Exception as e:
            logger.error(f"Authorized search request failed: {e}")
            raise SearchProviderError(f"Authorized search provider failed: {str(e)}") from e


class SerpApiVisualSearchProvider(VisualSearchProvider):
    """
    LIVE REVERSE IMAGE SEARCH via SerpApi (Google Lens / Google Reverse Image Search).
    Accepts public image URLs or uploads to query real-world visual search results.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.SERPAPI_API_KEY
        self.endpoint = "https://serpapi.com/search.json"

    @property
    def name(self) -> str:
        return "serpapi"

    def search(self, image_path: Path, search_mode: str = "web_discovery", **kwargs) -> List[CandidateResult]:
        if not self.api_key:
            raise SearchProviderError(
                "SerpApi API key not configured. Set SERPAPI_API_KEY environment variable or in config/settings.py"
            )

        logger.info(f"[SerpApi] Executing live Google Lens reverse image search for: {image_path.name}")
        
        # SerpApi requires an image_url or direct file upload
        # If running locally without a public URL, we pass the file to SerpApi's upload or fallback
        try:
            # Prepare request parameters
            params = {
                "engine": "google_lens",
                "api_key": self.api_key,
                "hl": "en"
            }
            
            with open(image_path, "rb") as f:
                files = {"file": (image_path.name, f, "image/jpeg")}
                response = requests.post(
                    self.endpoint,
                    params=params,
                    files=files,
                    timeout=settings.SEARCH_TIMEOUT_SECONDS
                )
                
            if response.status_code != 200:
                logger.error(f"[SerpApi] Search request returned status {response.status_code}: {response.text[:200]}")
                raise SearchProviderError(f"SerpApi returned HTTP {response.status_code}: {response.text[:200]}")

            data = response.json()
            candidates: List[CandidateResult] = []
            
            # Parse visual matches from Google Lens results
            visual_matches = data.get("visual_matches", [])
            for item in visual_matches[:settings.MAX_CANDIDATES]:
                img_url = item.get("thumbnail") or item.get("original")
                page_url = item.get("link") or item.get("source")
                if img_url and page_url:
                    candidates.append(CandidateResult(
                        source="google_lens_live",
                        page_url=page_url,
                        image_url=img_url,
                        title=item.get("title", "Google Lens Visual Match")
                    ))
            
            logger.info(f"[SerpApi] Discovered {len(candidates)} real visual candidates")
            return candidates

        except Exception as e:
            logger.error(f"[SerpApi] Search failed: {e}")
            raise SearchProviderError(f"SerpApi reverse image search failed: {e}") from e


class ProviderRegistry:
    """Central registry of available visual search providers."""

    def __init__(self):
        self._providers: Dict[str, VisualSearchProvider] = {}
        self.register(MockVisualSearchProvider())
        self.register(AuthorizedSearchProvider())
        self.register(SerpApiVisualSearchProvider())

    def register(self, provider: VisualSearchProvider) -> None:
        """Registers a visual search provider."""
        self._providers[provider.name] = provider
        logger.info(f"Registered visual search provider: '{provider.name}'")

    def get(self, name: str) -> VisualSearchProvider:
        """Retrieves provider by name."""
        if name not in self._providers:
            available = list(self._providers.keys())
            raise SearchProviderError(f"Unknown search provider '{name}'. Available: {available}")
        return self._providers[name]

