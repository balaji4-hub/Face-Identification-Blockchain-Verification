"""
STAGE 3: DISCOVERY ENGINE
Handles dynamic visual search and multi-provider search
"""
import asyncio
import time
import httpx
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from veriface.core.config import settings
from veriface.models.database import AsyncSession
from veriface.models.schemas import VerificationSession, Candidate
from veriface.models.schemas_api import DiscoveryRequest, DiscoveryResponse, SearchProviderConfig


@dataclass
class SearchResult:
    """Result from a search provider"""
    provider: str
    external_id: Optional[str] = None
    image_url: Optional[str] = None
    image_data: Optional[bytes] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SearchProvider(ABC):
    """Abstract base class for search providers"""
    
    def __init__(self, config: SearchProviderConfig):
        self.name = config.name
        self.enabled = config.enabled
        self.api_key = config.api_key
        self.api_endpoint = config.api_endpoint
        self.max_results = config.max_results
    
    @abstractmethod
    async def search(self, embedding: List[float], face_image_hash: str) -> List[SearchResult]:
        """Perform search using face embedding"""
        pass
    
    @abstractmethod
    async def download_candidate(self, result: SearchResult) -> bytes:
        """Download candidate image"""
        pass


class InternalSearchProvider(SearchProvider):
    """Internal database search provider"""
    
    def __init__(self, config: SearchProviderConfig, session: AsyncSession):
        super().__init__(config)
        self.session = session
    
    async def search(self, embedding: List[float], face_image_hash: str) -> List[SearchResult]:
        """Search internal database for similar embeddings"""
        results = []
        
        # Query database for similar embeddings
        # This would use vector similarity search in production
        query = """
            SELECT id, image_hash, embedding, similarity_score 
            FROM candidates 
            WHERE session_id != :current_session
            ORDER BY similarity_score DESC
            LIMIT :limit
        """
        
        # For now, return empty results (no matches found in demo mode)
        return results
    
    async def download_candidate(self, result: SearchResult) -> bytes:
        """Download from internal storage"""
        return result.image_data or b""


class ExternalSearchProvider(SearchProvider):
    """External API search provider (PimEyes, etc.)"""
    
    def __init__(self, config: SearchProviderConfig, client: httpx.AsyncClient):
        super().__init__(config)
        self.client = client
    
    async def search(self, embedding: List[float], face_image_hash: str) -> List[SearchResult]:
        """Search external API for similar faces"""
        results = []
        
        if not self.enabled:
            return results
        
        try:
            # Prepare request payload
            payload = {
                "embedding": embedding,
                "image_hash": face_image_hash,
                "max_results": self.max_results
            }
            
            # Add API key to headers
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            
            # Make request
            response = await self.client.post(
                self.api_endpoint or f"https://api.example.com/search",
                json=payload,
                headers=headers,
                timeout=settings.download_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get("results", []):
                    results.append(SearchResult(
                        provider=self.name,
                        external_id=item.get("id"),
                        image_url=item.get("image_url"),
                        confidence=item.get("confidence", 0.0),
                        metadata=item.get("metadata", {})
                    ))
            
        except Exception as e:
            # Log error but continue with other providers
            print(f"External search error ({self.name}): {str(e)}")
        
        return results
    
    async def download_candidate(self, result: SearchResult) -> bytes:
        """Download candidate image from external URL"""
        if not result.image_url:
            return b""
        
        try:
            response = await self.client.get(
                result.image_url,
                timeout=settings.download_timeout
            )
            
            if response.status_code == 200:
                return response.content
            
        except Exception as e:
            print(f"Download error ({result.provider}): {str(e)}")
        
        return b""


class DiscoveryEngineService:
    """Service for face candidate discovery across providers"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.providers: Dict[str, SearchProvider] = {}
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize search providers from config"""
        provider_configs = [
            SearchProviderConfig(
                name="internal",
                enabled=True,
                max_results=50
            ),
            SearchProviderConfig(
                name="external",
                enabled=False,  # Requires API key
                api_key=None,
                api_endpoint=None,
                max_results=50
            )
        ]
        
        # Create client for HTTP requests
        self.http_client = httpx.AsyncClient(
            timeout=settings.download_timeout,
            follow_redirects=True
        )
        
        for config in provider_configs:
            if config.name == "internal":
                self.providers[config.name] = InternalSearchProvider(config, self.session)
            else:
                self.providers[config.name] = ExternalSearchProvider(config, self.http_client)
    
    def get_enabled_providers(self, provider_names: Optional[List[str]] = None) -> List[SearchProvider]:
        """Get list of enabled providers, optionally filtered by name"""
        if provider_names:
            return [
                p for name, p in self.providers.items()
                if name in provider_names and p.enabled
            ]
        
        return [p for p in self.providers.values() if p.enabled]
    
    async def discover_candidates(
        self,
        session_id: str,
        embedding: List[float],
        face_image_hash: str,
        request: Optional[DiscoveryRequest] = None
    ) -> DiscoveryResponse:
        """
        Search multiple providers for candidate matches
        
        Args:
            session_id: Current verification session
            embedding: Face embedding vector
            face_image_hash: Hash of input face image
            request: Optional discovery configuration
            
        Returns:
            DiscoveryResponse with candidate results
        """
        start_time = time.time()
        
        # Get providers to search
        provider_names = request.search_providers if request else None
        providers = self.get_enabled_providers(provider_names)
        
        max_candidates = request.max_candidates if request else settings.max_candidates
        
        # Search all providers in parallel
        all_results: List[SearchResult] = []
        
        search_tasks = [
            provider.search(embedding, face_image_hash)
            for provider in providers
        ]
        
        provider_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        for i, result in enumerate(provider_results):
            if isinstance(result, Exception):
                print(f"Provider {providers[i].name} error: {result}")
                continue
            
            all_results.extend(result)
        
        # Sort by provider confidence
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit results
        all_results = all_results[:max_candidates]
        
        # Create candidate records
        for rank, result in enumerate(all_results):
            candidate = Candidate(
                id=str(uuid.uuid4()),
                session_id=session_id,
                provider_name=result.provider,
                external_id=result.external_id,
                candidate_image_url=result.image_url,
                rank=rank + 1,
                is_top_match=rank == 0,
                metadata=result.metadata,
                downloaded_at=datetime.utcnow()
            )
            self.session.add(candidate)
        
        # Update session status
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one()
        session.status = "discovery_complete"
        
        await self.session.commit()
        
        discovery_time = (time.time() - start_time) * 1000
        
        return DiscoveryResponse(
            session_id=session_id,
            candidates_found=len(all_results),
            providers_searched=[p.name for p in providers],
            discovery_time_ms=discovery_time,
            status="success"
        )
    
    async def cleanup(self):
        """Cleanup HTTP client"""
        await self.http_client.aclose()


# Helper imports
import uuid
from datetime import datetime
from sqlalchemy import select