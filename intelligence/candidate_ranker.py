"""
VeriFace Chain - Computer Vision: Candidate Ranking Engine
Ranks analyzed candidates based on biometric similarity scores and confidence tiers.
"""
from typing import List, Optional
from evidence.evidence_models import CandidateAnalysis
from core.logging import get_logger

logger = get_logger("intelligence.candidate_ranker")


class CandidateRanker:
    """Ranks and sorts candidate analyses in descending order of facial similarity."""
    
    @staticmethod
    def rank(candidates: List[CandidateAnalysis]) -> List[CandidateAnalysis]:
        """
        Sorts candidates by highest similarity score in descending order,
        assigning 1-indexed ranks.
        
        Args:
            candidates: List of CandidateAnalysis objects.
            
        Returns:
            Ranked list of candidates with updated rank attributes.
        """
        if not candidates:
            return []
            
        # Sort descending by highest_similarity
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.highest_similarity,
            reverse=True
        )
        
        # Update rank indices
        for index, candidate in enumerate(sorted_candidates, start=1):
            candidate.rank = index
            
        logger.info(f"Ranked {len(sorted_candidates)} candidate(s). Top score: {sorted_candidates[0].highest_similarity}")
        return sorted_candidates

    @staticmethod
    def get_strongest_match(candidates: List[CandidateAnalysis], threshold: float = None) -> Optional[CandidateAnalysis]:
        """
        Returns the top-ranked candidate if its similarity meets or exceeds the threshold.
        """
        ranked = CandidateRanker.rank(candidates)
        if not ranked:
            return None
            
        top_candidate = ranked[0]
        if threshold is not None and top_candidate.highest_similarity < threshold:
            return None
            
        return top_candidate
