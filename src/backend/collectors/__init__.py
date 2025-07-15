#!/usr/bin/env python3
"""
News Collection System

Simplified collectors package with clean imports.
"""

# Import main classes for easy external access
from .collectors import (
    NewsCollector,
    ArticleProcessor, 
    BatchCollector,
    ArticleParser,
    SourceCollector
)

from .core import (
    CollectionConfig,
    CollectionStats,
    ConfigManager,
    DateUtils,
    TextUtils,
    main,
    setup_logging
)

# Main exports
__all__ = [
    'NewsCollector',
    'ArticleProcessor',
    'BatchCollector', 
    'ArticleParser',
    'SourceCollector',
    'CollectionConfig',
    'CollectionStats',
    'ConfigManager',
    'DateUtils',
    'TextUtils',
    'main',
    'setup_logging'
]

# Version info
__version__ = '3.0.0'
