"""
VeriFace Chain - Discovery: Search Service
Coordinates visual search across providers with a lightweight audit trail.
Strict Privacy Standard: Search audit never records raw biometric vectors.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from config.settings import settings
from core.exceptions import SearchProviderError
from core.logging import get_logger
from discovery.provider_base import VisualSearchProvider
from discovery.provider_registry import ProviderRegistry
from evidence.evidence_models import CandidateResult

logger = get_logger("discovery.search_service")


class SearchAuditEntry(BaseModel):
    """Audit entry documenting an external or mock search query."""
    search_started_at: str
    search_completed_at: str
    provider_name: str
    candidate_count: int
    success: bool
    error: Optional[str] = None


class SearchService:
    """Orchestrates candidate search queries and maintains search audit logs."""
    
    def __init__(self, registry: ProviderRegistry = None):
        self.registry = registry or ProviderRegistry()
        self.last_audit: Optional[SearchAuditEntry] = None

    def search(
        self,
        image_path: Path,
        provider_name: str = None,
        search_mode: str = "web_discovery",
        **kwargs
    ) -> List[CandidateResult]:
        """
        Executes search via the selected provider and records an audit trail.
        
        Args:
            image_path: Validated input image path.
            provider_name: Specific provider name or None for default.
            search_mode: Mode for discovery simulation (web_discovery or private_unindexed).
            
        Returns:
            List of discovered CandidateResult objects.
        """
        provider_key = provider_name or settings.DEFAULT_SEARCH_PROVIDER
        provider = self.registry.get(provider_key)
        
        started_at = datetime.utcnow().isoformat() + "Z"
        logger.info(f"Starting visual search with provider: '{provider.name}' (mode: {search_mode})")
        
        try:
            candidates = provider.search(image_path, search_mode=search_mode, **kwargs)
            completed_at = datetime.utcnow().isoformat() + "Z"
            
            # Record audit entry
            self.last_audit = SearchAuditEntry(
                search_started_at=started_at,
                search_completed_at=completed_at,
                provider_name=provider.name,
                candidate_count=len(candidates),
                success=True
            )
            logger.info(f"Visual search complete: {len(candidates)} candidate(s) discovered")
            return candidates
            
        except Exception as e:
            completed_at = datetime.utcnow().isoformat() + "Z"
            self.last_audit = SearchAuditEntry(
                search_started_at=started_at,
                search_completed_at=completed_at,
                provider_name=provider.name,
                candidate_count=0,
                success=False,
                error=str(e)
            )
            raise SearchProviderError(f"Search provider '{provider.name}' failed: {e}") from e
