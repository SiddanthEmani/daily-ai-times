#!/usr/bin/env python3
"""
Result Types - Elegant data structures for pipeline results.

This module provides type-safe, structured result objects to replace raw dictionaries
throughout the pipeline, making the code more maintainable and self-documenting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from enum import Enum
import json


class ArticleCategory(str, Enum):
    """Article categories."""
    RESEARCH = "research"
    INDUSTRY = "industry"
    OPEN_SOURCE = "open-source"
    STARTUPS = "startups"
    MEDIA = "media"
    COMMUNITY = "community"
    ACADEMIC = "academic"
    GOVERNMENT = "government"
    SAFETY = "safety"
    BREAKTHROUGH = "breakthrough"


class QualityLevel(str, Enum):
    """Article quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Priority(str, Enum):
    """Article priority levels."""
    HIGH = "HIGH"
    MEDIUM = "MED"
    LOW = "LOW"


@dataclass
class ArticleScores:
    """Container for article scoring data."""
    stage1_score: Optional[float] = None
    stage2_score: Optional[float] = None
    stage3_score: Optional[float] = None
    final_score: Optional[float] = None
    quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'stage1_score': self.stage1_score,
            'stage2_score': self.stage2_score,
            'stage3_score': self.stage3_score,
            'final_score': self.final_score,
            'quality_score': self.quality_score
        }


@dataclass
class Article:
    """Structured article representation."""
    id: str
    title: str
    url: str
    description: str
    source: str
    category: ArticleCategory
    published_date: datetime
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scores: ArticleScores = field(default_factory=ArticleScores)
    quality: Optional[QualityLevel] = None
    priority: Optional[Priority] = None
    stage3_analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Article':
        """Create Article from dictionary data."""
        # Handle datetime parsing
        published_date = data.get('published_date')
        if isinstance(published_date, str):
            published_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
        elif published_date is None:
            published_date = datetime.now(timezone.utc)
        
        collected_at = data.get('collected_at')
        if isinstance(collected_at, str):
            collected_at = datetime.fromisoformat(collected_at.replace('Z', '+00:00'))
        elif collected_at is None:
            collected_at = datetime.now(timezone.utc)
        
        # Handle scores
        scores_data = data.get('scores', {})
        if isinstance(scores_data, dict):
            scores = ArticleScores(
                stage1_score=scores_data.get('stage1_score'),
                stage2_score=scores_data.get('stage2_score') or data.get('stage2_score'),
                stage3_score=scores_data.get('stage3_score'),
                final_score=scores_data.get('final_score') or data.get('final_score'),
                quality_score=scores_data.get('quality_score')
            )
        else:
            scores = ArticleScores()
        
        # Handle category
        category_str = data.get('category', 'industry')
        try:
            category = ArticleCategory(category_str.lower())
        except ValueError:
            category = ArticleCategory.INDUSTRY
        
        # Handle quality
        quality = None
        if data.get('quality'):
            try:
                quality = QualityLevel(data['quality'].lower())
            except ValueError:
                pass
        
        # Handle priority
        priority = None
        if data.get('priority'):
            try:
                priority = Priority(data['priority'].upper())
            except ValueError:
                pass
        
        return cls(
            id=data.get('id', ''),
            title=data.get('title', ''),
            url=data.get('url', ''),
            description=data.get('description', ''),
            source=data.get('source', ''),
            category=category,
            published_date=published_date,
            collected_at=collected_at,
            scores=scores,
            quality=quality,
            priority=priority,
            stage3_analysis=data.get('stage3_analysis'),
            metadata=data.get('metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API/JSON serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'url': self.url,
            'description': self.description,
            'source': self.source,
            'category': self.category.value,
            'published_date': self.published_date.isoformat(),
            'collected_at': self.collected_at.isoformat(),
            'quality': self.quality.value if self.quality else None,
            'priority': self.priority.value if self.priority else None,
            'stage2_score': self.scores.stage2_score,
            'final_score': self.scores.final_score,
            'scores': self.scores.to_dict(),
            'stage3_analysis': self.stage3_analysis,
            'metadata': self.metadata
        }


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    model: str
    input_count: int
    output_count: int
    pass_rate: float
    target_pass_rate: float
    requests_made: int
    processing_time: float
    articles_per_second: float
    skipped: bool = False
    reason: Optional[str] = None
    batch_size: Optional[int] = None
    batches_processed: Optional[int] = None
    categories_found: Optional[Dict[str, int]] = None
    quality_distribution: Optional[Dict[str, int]] = None
    acceptance_rate: Optional[float] = None
    average_score: Optional[float] = None
    score_distribution: Optional[Dict[str, int]] = None
    priority_distribution: Optional[Dict[str, int]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    successful_evaluations: Optional[int] = None
    failed_evaluations: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            'model': self.model,
            'input_count': self.input_count,
            'output_count': self.output_count,
            'pass_rate': self.pass_rate,
            'target_pass_rate': self.target_pass_rate,
            'requests_made': self.requests_made,
            'processing_time': self.processing_time,
            'articles_per_second': self.articles_per_second,
            'skipped': self.skipped
        }
        
        # Add optional fields if present
        optional_fields = [
            'reason', 'batch_size', 'batches_processed', 'categories_found',
            'quality_distribution', 'acceptance_rate', 'average_score',
            'score_distribution', 'priority_distribution', 'quality_metrics',
            'successful_evaluations', 'failed_evaluations'
        ]
        
        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                result[field] = value
        
        return result


@dataclass
class PipelineMetadata:
    """Metadata about the pipeline execution."""
    pipeline_version: str
    generated_at: datetime
    environment: str = 'development'
    api_key_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'pipeline_version': self.pipeline_version,
            'generated_at': self.generated_at.isoformat(),
            'environment': self.environment,
            'api_key_used': self.api_key_used
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for the pipeline."""
    total_processing_time: float
    overall_pass_rate: float
    input_count: int
    output_count: int
    stages_executed: int
    total_api_requests: int = 0
    total_tokens_used: int = 0
    estimated_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'total_processing_time': self.total_processing_time,
            'overall_pass_rate': self.overall_pass_rate,
            'input_count': self.input_count,
            'output_count': self.output_count,
            'stages_executed': self.stages_executed,
            'total_api_requests': self.total_api_requests,
            'total_tokens_used': self.total_tokens_used,
            'estimated_cost': self.estimated_cost
        }


@dataclass
class PipelineResult:
    """Complete result from pipeline execution."""
    articles: List[Article]
    stage_results: List[StageResult]
    metadata: PipelineMetadata
    performance_metrics: PerformanceMetrics
    
    @property
    def input_count(self) -> int:
        """Get input article count."""
        return self.performance_metrics.input_count
    
    @property
    def output_count(self) -> int:
        """Get output article count."""
        return len(self.articles)
    
    @property
    def processing_time(self) -> float:
        """Get total processing time."""
        return self.performance_metrics.total_processing_time
    
    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        """Get result for a specific stage."""
        for stage_result in self.stage_results:
            if stage_result.stage_name == stage_name:
                return stage_result
        return None
    
    def to_api_format(self) -> Dict[str, Any]:
        """Convert to API-friendly format for latest.json."""
        return {
            "generated_at": self.metadata.generated_at.isoformat(),
            "collection_time_seconds": self.performance_metrics.total_processing_time,
            "count": len(self.articles),
            "articles": [article.to_dict() for article in self.articles],
            "pipeline_info": {
                "version": self.metadata.pipeline_version,
                "input_count": self.performance_metrics.input_count,
                "output_count": len(self.articles),
                "overall_pass_rate": self.performance_metrics.overall_pass_rate,
                "processing_time": self.performance_metrics.total_processing_time,
                "stages": {
                    stage_result.stage_name: stage_result.to_dict()
                    for stage_result in self.stage_results
                }
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        return {
            'input_count': self.performance_metrics.input_count,
            'output_count': len(self.articles),
            'articles': [article.to_dict() for article in self.articles],
            'processing_time': self.performance_metrics.total_processing_time,
            'pipeline_stages': {
                stage_result.stage_name: stage_result.to_dict()
                for stage_result in self.stage_results
            },
            'pipeline_version': self.metadata.pipeline_version,
            'pipeline_config': {},  # Will be filled by orchestrator
            'generated_at': self.metadata.generated_at.isoformat(),
            'overall_pass_rate': self.performance_metrics.overall_pass_rate
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)


@dataclass
class CollectionResult:
    """Result from news collection."""
    articles: List[Article]
    total_collected: int
    successful_sources: int
    failed_sources: int
    empty_sources: int
    processing_time: float
    failure_reasons: Dict[str, str] = field(default_factory=dict)
    duplicates_removed: int = 0
    quality_filtered: int = 0
    
    @property
    def total_sources(self) -> int:
        """Get total number of sources attempted."""
        return self.successful_sources + self.failed_sources + self.empty_sources
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'articles': [article.to_dict() for article in self.articles],
            'total_collected': len(self.articles),
            'successful_sources': self.successful_sources,
            'failed_sources': self.failed_sources,
            'empty_sources': self.empty_sources,
            'processing_time': self.processing_time,
            'failure_reasons': self.failure_reasons,
            'duplicates_removed': self.duplicates_removed,
            'quality_filtered': self.quality_filtered,
            'timestamp': datetime.now(timezone.utc).timestamp()
        }
