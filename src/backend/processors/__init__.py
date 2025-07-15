#!/usr/bin/env python3
"""
NewsXP.ai Swarm Intelligence Processing Module

Multi-agent swarm implementation for high-throughput, fault-tolerant article processing.
Based on research-validated patterns from Stanford Medicine and ICLR 2025 studies.
"""

from .bulk_agent import BulkFilteringAgent
from .consensus_engine import ConsensusEngine
from .deduplication_utils import ArticleDeduplicator

__all__ = [
    'BulkFilteringAgent', 
    'ConsensusEngine', 
    'ArticleDeduplicator'
]
