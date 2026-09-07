"""Discovery package for VeriFace Chain."""
from discovery.provider_base import VisualSearchProvider
from discovery.candidate_fetcher import CandidateFetcher
from discovery.provider_registry import MockVisualSearchProvider, AuthorizedSearchProvider, ProviderRegistry
from discovery.search_service import SearchService, SearchAuditEntry

__all__ = [
    "VisualSearchProvider",
    "CandidateFetcher",
    "MockVisualSearchProvider",
    "AuthorizedSearchProvider",
    "ProviderRegistry",
    "SearchService",
    "SearchAuditEntry"
]
