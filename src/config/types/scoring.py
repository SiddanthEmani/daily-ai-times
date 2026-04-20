#!/usr/bin/env python3
"""
Multi-Dimensional Scoring System for Daily AI Times

Defines the data structures and utilities for the enhanced multi-dimensional
scoring system that replaces binary ACCEPT/REJECT decisions.
"""

from typing import Dict, Any, Optional, NamedTuple, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json

@dataclass
class MultiDimensionalScore:
    """
    Multi-dimensional scoring structure for articles.
    
    Replaces binary ACCEPT/REJECT decisions with rich, multi-dimensional
    scoring that provides detailed metadata for deep intelligence agents.
    """
    
    # Core scoring dimensions (0-1 scale)
    relevance_score: float  # How relevant to technology/innovation
    quality_score: float    # Content quality and depth
    novelty_score: float    # How new/unique the information is
    impact_score: float     # Potential importance/influence
    
    # Confidence metrics
    confidence_mean: float       # Mean confidence level
    confidence_std: float        # Standard deviation (uncertainty)
    confidence_interval_low: float   # Lower bound of confidence interval
    confidence_interval_high: float  # Upper bound of confidence interval
    
    # Agent metadata
    agent_name: str
    processing_timestamp: str
    model_name: str
    specialization: str
    
    # Derived metrics
    overall_score: float = 0.0  # Computed overall score
    decision_threshold: float = 0.5  # Threshold for binary decision (lowered from 0.7)
    binary_decision: bool = False  # Computed binary decision for compatibility
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self.overall_score = self._calculate_overall_score()
        self.binary_decision = self.overall_score >= self.decision_threshold
        
        # Ensure confidence interval is valid
        if self.confidence_interval_low > self.confidence_interval_high:
            self.confidence_interval_low, self.confidence_interval_high = \
                self.confidence_interval_high, self.confidence_interval_low
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall score from individual dimensions."""
        # Weighted average of dimensions
        # These weights can be configured in the future
        weights = {
            'relevance': 0.3,
            'quality': 0.25,
            'novelty': 0.25,
            'impact': 0.2
        }
        
        overall = (
            self.relevance_score * weights['relevance'] +
            self.quality_score * weights['quality'] +
            self.novelty_score * weights['novelty'] +
            self.impact_score * weights['impact']
        )
        
        return min(1.0, max(0.0, overall))
    
    def get_confidence_range(self) -> float:
        """Get the confidence range (uncertainty)."""
        return self.confidence_interval_high - self.confidence_interval_low
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if this is a high-confidence decision."""
        return self.confidence_mean >= threshold
    
    def is_uncertain(self, max_range: float = 0.3) -> bool:
        """Check if this decision has high uncertainty."""
        return self.get_confidence_range() > max_range
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiDimensionalScore':
        """Create from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MultiDimensionalScore':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_legacy_format(self) -> Tuple[bool, float]:
        """Get legacy format (decision, confidence) for backward compatibility."""
        return self.binary_decision, self.confidence_mean

@dataclass
class ArticleScore:
    """
    Complete scoring result for an article, including the article data
    and its multi-dimensional score.
    """
    article: Dict[str, Any]
    score: MultiDimensionalScore
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'article': self.article,
            'score': self.score.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArticleScore':
        """Create from dictionary."""
        return cls(
            article=data['article'],
            score=MultiDimensionalScore.from_dict(data['score'])
        )
    
    def get_legacy_format(self) -> Tuple[Dict[str, Any], bool, float]:
        """Get legacy format (article, decision, confidence) for backward compatibility."""
        return self.article, self.score.binary_decision, self.score.confidence_mean

def create_multi_dimensional_score(
    relevance: float,
    quality: float,
    novelty: float,
    impact: float,
    confidence_mean: float,
    confidence_std: float,
    agent_name: str,
    model_name: str,
    specialization: str,
    decision_threshold: float = 0.5
) -> MultiDimensionalScore:
    """
    Factory function to create a multi-dimensional score with computed confidence interval.
    
    Args:
        relevance: Relevance score (0-1)
        quality: Quality score (0-1)
        novelty: Novelty score (0-1)
        impact: Impact score (0-1)
        confidence_mean: Mean confidence level (0-1)
        confidence_std: Standard deviation of confidence (0-1)
        agent_name: Name of the agent
        model_name: Name of the model
        specialization: Agent specialization
        decision_threshold: Threshold for binary decision
    
    Returns:
        MultiDimensionalScore instance
    """
    # Calculate confidence interval (assume normal distribution)
    # Using 1.96 * std for 95% confidence interval
    margin = 1.96 * confidence_std
    confidence_low = max(0.0, confidence_mean - margin)
    confidence_high = min(1.0, confidence_mean + margin)
    
    score = MultiDimensionalScore(
        relevance_score=relevance,
        quality_score=quality,
        novelty_score=novelty,
        impact_score=impact,
        confidence_mean=confidence_mean,
        confidence_std=confidence_std,
        confidence_interval_low=confidence_low,
        confidence_interval_high=confidence_high,
        agent_name=agent_name,
        processing_timestamp=datetime.now(timezone.utc).isoformat(),
        model_name=model_name,
        specialization=specialization,
        decision_threshold=decision_threshold
    )
    
    return score

def aggregate_multi_dimensional_scores(
    scores: List[MultiDimensionalScore],
    weights: Optional[Dict[str, float]] = None
) -> MultiDimensionalScore:
    """
    Aggregate multiple multi-dimensional scores into a single consensus score.
    
    Args:
        scores: List of multi-dimensional scores to aggregate
        weights: Optional weights for each score (by agent_name)
    
    Returns:
        Aggregated multi-dimensional score
    """
    if not scores:
        raise ValueError("Cannot aggregate empty list of scores")
    
    if weights is None:
        weights = {score.agent_name: 1.0 for score in scores}
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")
    
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Calculate weighted averages
    weighted_relevance = sum(score.relevance_score * normalized_weights.get(score.agent_name, 0) 
                           for score in scores)
    weighted_quality = sum(score.quality_score * normalized_weights.get(score.agent_name, 0) 
                         for score in scores)
    weighted_novelty = sum(score.novelty_score * normalized_weights.get(score.agent_name, 0) 
                         for score in scores)
    weighted_impact = sum(score.impact_score * normalized_weights.get(score.agent_name, 0) 
                        for score in scores)
    
    # Calculate confidence metrics
    weighted_confidence_mean = sum(score.confidence_mean * normalized_weights.get(score.agent_name, 0) 
                                 for score in scores)
    
    # Calculate uncertainty as max of individual uncertainties (conservative approach)
    max_uncertainty = max(score.confidence_std for score in scores)
    
    # Create aggregated score
    aggregated_score = create_multi_dimensional_score(
        relevance=weighted_relevance,
        quality=weighted_quality,
        novelty=weighted_novelty,
        impact=weighted_impact,
        confidence_mean=weighted_confidence_mean,
        confidence_std=max_uncertainty,
        agent_name="consensus_engine",
        model_name="aggregated",
        specialization="consensus",
        decision_threshold=0.7
    )
    
    return aggregated_score
