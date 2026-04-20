#!/usr/bin/env python3
"""
Statistics tracking and collection utilities for Daily AI Times
Centralized stats management for collection workflows.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class CollectionStats:
    """Statistics for article collection operations."""
    successful_sources: int = 0
    empty_sources: int = 0
    failed_sources: int = 0
    total_sources: int = 0
    total_articles: int = 0
    processing_time: float = 0.0
    failure_reasons: Dict[str, str] = field(default_factory=dict)
    
    # Deduplication stats
    original_articles: int = 0
    duplicates_removed: int = 0
    deduplication_time: float = 0.0
    
    # Timing
    collection_start_time: Optional[float] = None
    
    def start_timing(self) -> None:
        """Start timing the collection process."""
        self.collection_start_time = time.time()
    
    def stop_timing(self) -> None:
        """Stop timing and calculate processing time."""
        if self.collection_start_time:
            self.processing_time = time.time() - self.collection_start_time
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_sources == 0:
            return 0.0
        return (self.successful_sources / self.total_sources) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary format."""
        return {
            'successful_sources': self.successful_sources,
            'empty_sources': self.empty_sources,
            'failed_sources': self.failed_sources,
            'total_sources': self.total_sources,
            'total_articles': self.total_articles,
            'processing_time': round(self.processing_time, 2),
            'success_rate_percent': round(self.success_rate, 1),
            'failure_reasons': self.failure_reasons,
            'deduplication_stats': {
                'original_articles': self.original_articles,
                'duplicates_removed': self.duplicates_removed,
                'deduplication_time': round(self.deduplication_time, 3)
            }
        }


@dataclass
class WorkflowStats:
    """Statistics for complete workflow operations."""
    workflow_type: str = "unknown"
    status: str = "pending"  # pending, success, error
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    start_time: Optional[float] = None
    
    # Collection stats
    collection_stats: Optional[CollectionStats] = None
    
    # Database stats
    articles_inserted: int = 0
    duplicates_skipped: int = 0
    articles_cleaned_up: int = 0
    total_articles_in_db: int = 0
    
    def start_workflow(self, workflow_type: str) -> None:
        """Start workflow timing."""
        self.workflow_type = workflow_type
        self.status = "running"
        self.start_time = time.time()
    
    def complete_workflow(self, success: bool = True, error: Optional[str] = None) -> None:
        """Complete workflow and calculate duration."""
        if self.start_time:
            self.duration_seconds = time.time() - self.start_time
        self.status = "success" if success else "error"
        self.error_message = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow stats to dictionary format."""
        from ..utils.time_helpers import get_current_timestamp
        
        result = {
            'workflow_info': {
                'workflow_type': self.workflow_type,
                'timestamp': get_current_timestamp(),
                'duration_seconds': round(self.duration_seconds, 2),
                'status': self.status,
                'error': self.error_message
            },
            'database_stats': {
                'articles_inserted': self.articles_inserted,
                'duplicates_skipped': self.duplicates_skipped,
                'articles_cleaned_up': self.articles_cleaned_up,
                'total_articles_in_db': self.total_articles_in_db
            }
        }
        
        if self.collection_stats:
            result['collection_stats'] = {
                'articles_collected': self.collection_stats.total_articles,
                'sources_successful': self.collection_stats.successful_sources,
                'sources_failed': self.collection_stats.failed_sources,
                'sources_empty': self.collection_stats.empty_sources,
                'total_sources': self.collection_stats.total_sources,
                'success_rate_percent': round(self.collection_stats.success_rate, 1),
                'failure_reasons': self.collection_stats.failure_reasons
            }
            result['deduplication_stats'] = {
                'original_articles': self.collection_stats.original_articles,
                'duplicates_removed': self.collection_stats.duplicates_removed,
                'deduplication_time': self.collection_stats.deduplication_time
            }
        
        return result
