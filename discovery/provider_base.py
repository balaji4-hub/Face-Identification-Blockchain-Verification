"""
VeriFace Chain - Discovery: Abstract Visual Search Provider
Defines the clean, provider-agnostic interface for visual reverse-image search engines.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from evidence.evidence_models import CandidateResult


class VisualSearchProvider(ABC):
    """Abstract interface for all visual search providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name or identifier of the provider."""
        pass
        
    @abstractmethod
    def search(self, image_path: Path, **kwargs) -> List[CandidateResult]:
        """
        Executes dynamic visual search for matching public content using an input image.
        
        Args:
            image_path: Path to the validated input face image.
            **kwargs: Optional provider options such as search_mode.
            
        Returns:
            List of standardized CandidateResult objects.
        """
        pass
