#!/usr/bin/env python3
"""
News Collection Core - Configuration and data structures
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CollectionConfig:
    """Streamlined configuration for RSS news collection."""
    max_articles: int = 500
    max_age_days: int = 7
    batch_size: int = 10
    timeout_seconds: int = 30
    min_title_length: int = 5
    min_description_length: int = 10
    # RSS-specific configurations
    rss_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.rss_config is None:
            self.rss_config = {}

@dataclass
class CollectionStats:
    """Collection operation statistics."""
    successful: int = 0
    empty: int = 0
    failed: int = 0
    total_articles: int = 0
    processing_time: float = 0.0
    failure_reasons: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.failure_reasons is None:
            self.failure_reasons = {}

class ConfigManager:
    """Lightweight configuration loader."""
    
    @staticmethod
    def load() -> tuple[CollectionConfig, Dict[str, Dict]]:
        """Load configuration and sources."""
        try:
            # Import using absolute path instead of relative import
            import sys
            from pathlib import Path
            
            # Add src directory to path if not already there
            src_dir = Path(__file__).parent.parent.parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
                
            from shared.config.config_loader import ConfigLoader, load_sources_config, get_collectors_config
            
            # Load collection config
            collection_data = ConfigLoader.get('collection', {}, "app")
            
            # Load collector-specific config
            collectors_config = get_collectors_config()
            
            config = CollectionConfig(
                max_articles=collection_data.get('max_articles_to_collect', 500),
                max_age_days=collection_data.get('max_age_days', 7),
                batch_size=collection_data.get('performance', {}).get('batch_size', 10),
                timeout_seconds=collection_data.get('performance', {}).get('timeout_seconds', 30),
                min_title_length=collection_data.get('quality_filters', {}).get('min_title_length', 5),
                min_description_length=collection_data.get('quality_filters', {}).get('min_description_length', 10),
                rss_config=collectors_config
            )
            
            # Load sources
            sources_config = load_sources_config()
            sources = sources_config.get('sources', {})
            if isinstance(sources, dict) and 'sources' in sources:
                sources = sources['sources']
            
            # Filter enabled sources
            enabled_sources = {k: v for k, v in sources.items() if v.get('enabled', True)}
            logger.info(f"Loaded {len(enabled_sources)} enabled sources")
            
            return config, enabled_sources
            
        except Exception as e:
            logger.warning(f"Config loader not available: {e}")
            return ConfigManager._get_default_config()
    
    @staticmethod
    def _get_default_config() -> tuple[CollectionConfig, Dict]:
        """Fallback configuration when shared config is unavailable."""
        return CollectionConfig(), {}
    
    @staticmethod
    def load_config() -> tuple[CollectionConfig, Dict[str, Dict]]:
        """Legacy method name for backward compatibility."""
        return ConfigManager.load()

class DateUtils:
    """Date handling utilities."""
    
    @staticmethod
    def parse_date(date_value: Any) -> str:
        """Parse various date formats to ISO string."""
        if not date_value:
            return datetime.now(timezone.utc).isoformat()
        
        try:
            if hasattr(date_value, 'isoformat'):
                return date_value.isoformat()
            elif hasattr(date_value, 'timetuple'):
                import time
                return datetime.fromtimestamp(time.mktime(date_value.timetuple()), timezone.utc).isoformat()
            elif isinstance(date_value, str):
                from dateutil import parser as date_parser
                return date_parser.parse(date_value).isoformat()
            elif isinstance(date_value, (tuple, list)) and len(date_value) >= 6:
                import time
                time_tuple = tuple(date_value[:9]) if len(date_value) >= 9 else tuple(list(date_value) + [0] * (9 - len(date_value)))
                return datetime.fromtimestamp(time.mktime(time_tuple), timezone.utc).isoformat()
        except Exception:
            pass
        
        return datetime.now(timezone.utc).isoformat()
    
    @staticmethod
    def is_recent(date_str: str, max_age_days: int) -> bool:
        """Check if article is within age limit."""
        if max_age_days <= 0:
            return True
        
        try:
            article_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            if article_date.tzinfo is None:
                article_date = article_date.replace(tzinfo=timezone.utc)
            
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            return article_date >= cutoff
        except Exception:
            return True

class TextUtils:
    """Text processing utilities."""
    
    @staticmethod
    def clean_html(content: str, max_length: int = 500, config: Optional[Dict[str, Any]] = None) -> str:
        """Clean HTML content and truncate."""
        if not content:
            return ''
        
        # Get configuration for content extraction (simplified for RSS-only)
        text_config = {}
        if config:
            extraction_config = config.get('content_extraction', {})
            text_config = extraction_config.get('text_normalization', {})
            
            # Use configured max_length if available
            processing_config = config.get('processing', {})
            max_length = processing_config.get('max_description_truncate_length', max_length)
        
        # Handle dict content
        if isinstance(content, dict):
            content = content.get('rendered', str(content))
        
        # Remove HTML and clean whitespace
        from bs4 import BeautifulSoup
        import re
        cleaned = BeautifulSoup(str(content), 'html.parser').get_text()
        
        # Apply text normalization based on config
        if text_config:
            if text_config.get('collapse_spaces', True):
                cleaned = re.sub(r'\s+', ' ', cleaned)
            if text_config.get('trim_whitespace', True):
                cleaned = cleaned.strip()
            if text_config.get('remove_empty_lines', True):
                cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        else:
            # Default behavior for RSS feeds
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Truncate keeping word boundaries
        if len(cleaned) > max_length:
            truncated = cleaned[:max_length].rsplit(' ', 1)[0]
            return truncated + '...' if truncated else cleaned[:max_length]
        
        return cleaned
    
    @staticmethod
    def extract_field(source: Any, fields: List[str], default: str = '') -> str:
        """Extract field using multiple possible names."""
        for field in fields:
            value = None
            if hasattr(source, field):
                value = getattr(source, field, None)
            elif isinstance(source, dict):
                value = source.get(field)
            
            if value:
                # Process lists and objects
                if isinstance(value, list) and value:
                    value = value[0].get('value', str(value[0])) if isinstance(value[0], dict) else str(value[0])
                return str(value).strip() if value else ''
        
        return default
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Basic URL validation."""
        return bool(url and 
                   len(url) >= 10 and 
                   (url.startswith('http://') or url.startswith('https://')))

# Main entry point functionality
import asyncio
import logging

def setup_logging():
    """Configure logging to display in terminal."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    """Main entry point for standalone execution."""
    setup_logging()
    
    async def run_collection():
        from .collectors import NewsCollector
        collector = NewsCollector()
        # Use configuration default instead of hardcoded value
        max_articles = collector.config.max_articles
        articles = await collector.collect_all(max_articles=max_articles)
        print("=" * 40)
        print(f"Collected {len(articles)} articles")
        print(f"Sources: ✓{collector.stats.successful} ○{collector.stats.empty} ✗{collector.stats.failed}")
        print(f"Processing time: {collector.stats.processing_time:.2f}s")
    
    asyncio.run(run_collection())

# Export main classes for external import
def get_exports():
    """Get all main classes for external import."""
    from .collectors import NewsCollector, ArticleProcessor, BatchCollector
    return {
        'NewsCollector': NewsCollector,
        'ArticleProcessor': ArticleProcessor, 
        'BatchCollector': BatchCollector,
        'CollectionStats': CollectionStats,
        'CollectionConfig': CollectionConfig,
        'ConfigManager': ConfigManager
    }

__all__ = [
    'CollectionConfig',
    'CollectionStats', 
    'ConfigManager',
    'DateUtils',
    'TextUtils',
    'main',
    'setup_logging',
    'get_exports'
]

if __name__ == "__main__":
    main()
