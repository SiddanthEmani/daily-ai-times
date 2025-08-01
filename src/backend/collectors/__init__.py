#!/usr/bin/env python3
"""
News Collection System

Simplified collectors package with clean imports.
"""

# Import all classes from the merged collectors.py file
from .collectors import (
    NewsCollector,
    BatchCollector,
    ArticleParser,
    SourceCollector,
    Config,
    Stats,
    main,
    setup_logging
)

# Main exports
__all__ = [
    'NewsCollector',
    'BatchCollector', 
    'ArticleParser',
    'SourceCollector',
    'Config',
    'Stats',
    'main',
    'setup_logging'
]

# Version info
__version__ = '3.0.0'
