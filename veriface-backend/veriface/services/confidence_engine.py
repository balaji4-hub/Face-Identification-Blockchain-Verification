"""
STAGE 5: CONFIDENCE ENGINE
Handles similarity scoring, candidate ranking, and decision threshold
"""
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np

from veriface.core.config import settings
from veriface.models.database import AsyncSession
from veriface.models.schemas import VerificationSession, Candidate
from veriface.models.schemas_api import (
    ConfidenceResponse, ConfidenceDecision, CandidateInfo
)


@dataclass
class ConfidenceConfig:
    """Configuration for confidence decisions"""
    similarity_threshold: float = 0.6
    ranking_top_k: int = 10
    min_candidates_for_confidence: int = 3
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.4
    tie_break_margin: float = 0.05


class ConfidenceEngineService:
    """Service for computing confidence scores and making match decisions"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.config = ConfidenceConfig(
            similarity_threshold=settings.similarity_threshold,
            ranking_top_k=settings.ranking_top_k
        )
    
    def calculate_ensemble_score(
        self, 
        similarity_score: float,
        detection_confidence: float,
        face_count: int,
        provider_trust: float = 0.8
    ) -> float:
        """
        Calculate ensemble confidence score from multiple signals
        
        Args:
            similarity_score: Embedding similarity (0-1)
            detection_confidence: Face detection confidence (0-1)
            face_count: Number of faces in candidate
            provider_trust: Trust score for the provider
            
        Returns:
            Ensemble confidence score (0-1)
        """
        # Weight factors
        w_similarity = 0.5
        w_detection = 0.2
        w_multi_face = 0.15
        w_provider = 0.15
        
        # Multi-face penalty (prefer single face images)
        multi_face_score = 1.0 / (1.0 + (face_count - 1) * 0.2)
        
        # Ensemble combination
        score = (
            w_similarity * similarity_score +
            w_detection * detection_confidence +
            w_multi_face * multi_face_score +
            w_provider * provider_trust
        )
        
        return min(1.0, score)
    
    def calculate_rank_score(self, rank: int, total_candidates: int) -> float:
        """
        Calculate score based on ranking position
        
        Args:
            rank: Position in ranked list (1-indexed)
            total_candidates: Total number of candidates
            
        Returns:
            Rank-based score (0-1)
        """
        if total_candidates == 0:
            return 0.0
        
        # Exponential decay based on rank
        decay_factor = 0.5
        base_score = np.power(decay_factor, rank - 1)
        
        # Normalize by candidate count
        normalization = 1.0 - (rank - 1) / max(total_candidates, 1)
        
        return base_score * 0.5 + normalization * 0.5
    
    def detect_tie(self, scores: List[float], margin: float = None) -> bool:
        """
        Detect if there's a tie between top candidates
        
        Args:
            scores: List of similarity scores
            margin: Minimum difference to avoid tie
            
        Returns:
            True if tie detected
        """
        if margin is None:
            margin = self.config.tie_break_margin
        
        if len(scores) < 2:
            return False
        
        return abs(scores[0] - scores[1]) < margin
    
    def make_decision(
        self,
        top_score: float,
        top_rank: int,
        total_candidates: int,
        is_tie: bool
    ) -> ConfidenceDecision:
        """
        Make match/no-match decision based on confidence scores
        
        Returns:
            ConfidenceDecision with reasoning
        """
        # Determine match threshold
        threshold = self.config.similarity_threshold
        
        # Check for match
        if top_score >= threshold:
            if is_tie:
                reasoning = (
                    f"MATCH with confidence {top_score:.3f}. "
                    f"Warning: Close scores detected with candidate at rank {top_rank + 1}."
                )
            else:
                reasoning = (
                    f"MATCH with confidence {top_score:.3f}. "
                    f"Clear best match at rank {top_rank} out of {total_candidates} candidates."
                )
            
            return ConfidenceDecision(
                match_found=True,
                confidence_score=top_score,
                decision_threshold=threshold,
                ranking_position=top_rank,
                reasoning=reasoning
            )
        else:
            if total_candidates > 0:
                reasoning = (
                    f"NO MATCH. Best confidence {top_score:.3f} below threshold {threshold}. "
                    f"Analyzed {total_candidates} candidates."
                )
            else:
                reasoning = "NO MATCH. No candidates to compare."
            
            return ConfidenceDecision(
                match_found=False,
                confidence_score=top_score,
                decision_threshold=threshold,
                ranking_position=None,
                reasoning=reasoning
            )
    
    def rank_candidates(
        self,
        candidates: List[Candidate],
        ensemble_scores: Dict[str, float]
    ) -> List[tuple]:
        """
        Rank candidates by ensemble score
        
        Args:
            candidates: List of candidate records
            ensemble_scores: Dict mapping candidate_id to score
            
        Returns:
            List of (candidate, score) tuples sorted by score
        """
        ranked = [
            (c, ensemble_scores.get(c.id, 0.0))
            for c in candidates
        ]
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    async def compute_confidence(
        self,
        session_id: str
    ) -> ConfidenceResponse:
        """
        Compute confidence scores and make decision for a session
        
        Args:
            session_id: Verification session ID
            
        Returns:
            ConfidenceResponse with decision and ranked candidates
        """
        start_time = time.time()
        
        # Get candidates
        result = await self.session.execute(
            select(Candidate).where(Candidate.session_id == session_id)
            .order_by(Candidate.rank)
        )
        candidates = result.scalars().all()
        
        if not candidates:
            decision = ConfidenceDecision(
                match_found=False,
                confidence_score=0.0,
                decision_threshold=self.config.similarity_threshold,
                ranking_position=None,
                reasoning="No candidates available for comparison"
            )
            
            return ConfidenceResponse(
                session_id=session_id,
                match_found=False,
                confidence_score=0.0,
                best_candidate_id=None,
                decision=decision,
                all_candidates_ranked=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Calculate ensemble scores for each candidate
        ensemble_scores = {}
        for candidate in candidates:
            score = self.calculate_ensemble_score(
                similarity_score=candidate.similarity_score,
                detection_confidence=candidate.detection_confidence,
                face_count=candidate.face_count
            )
            ensemble_scores[candidate.id] = score
        
        # Re-rank based on ensemble scores
        ranked = self.rank_candidates(candidates, ensemble_scores)
        
        # Update rankings in database
        for rank, (candidate, score) in enumerate(ranked):
            candidate.similarity_score = score
            candidate.rank = rank + 1
            candidate.is_top_match = rank == 0
        
        # Get top scores for decision
        top_score = ranked[0][1] if ranked else 0.0
        top_candidate_id = ranked[0][0].id if ranked else None
        
        # Check for ties
        scores = [s for _, s in ranked]
        is_tie = self.detect_tie(scores)
        
        # Make decision
        decision = self.make_decision(
            top_score=top_score,
            top_rank=ranked[0][0].rank if ranked else 1,
            total_candidates=len(candidates),
            is_tie=is_tie
        )
        
        # Create response candidates
        candidate_infos = []
        for candidate, score in ranked[:self.config.ranking_top_k]:
            info = CandidateInfo(
                candidate_id=candidate.id,
                provider_name=candidate.provider_name,
                external_id=candidate.external_id,
                image_url=candidate.candidate_image_url,
                face_detected=candidate.face_detected,
                face_count=candidate.face_count,
                detection_confidence=candidate.detection_confidence,
                similarity_score=score,
                distance=candidate.distance,
                rank=candidate.rank,
                is_top_match=candidate.is_top_match,
                metadata=candidate.extra_metadata
            )
            candidate_infos.append(info)
        
        # Update session
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one()
        session.status = "confidence_computed"
        session.match_found = decision.match_found
        session.match_confidence = decision.confidence_score
        
        await self.session.commit()
        
        processing_time = (time.time() - start_time) * 1000
        
        return ConfidenceResponse(
            session_id=session_id,
            match_found=decision.match_found,
            confidence_score=decision.confidence_score,
            best_candidate_id=top_candidate_id,
            decision=decision,
            all_candidates_ranked=candidate_infos,
            processing_time_ms=processing_time
        )
    
    async def get_decision(self, session_id: str) -> Optional[ConfidenceDecision]:
        """Get decision for a session"""
        session = (await self.session.execute(
            select(VerificationSession).where(VerificationSession.id == session_id)
        )).scalar_one_or_none()
        
        if not session or session.match_found is None:
            return None
        
        return ConfidenceDecision(
            match_found=session.match_found,
            confidence_score=session.match_confidence or 0.0,
            decision_threshold=self.config.similarity_threshold,
            ranking_position=None,
            reasoning=f"Decision made: {'MATCH' if session.match_found else 'NO MATCH'}"
        )


# Helper imports
from sqlalchemy import select