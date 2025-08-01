#!/usr/bin/env python3
"""
Enhanced Sources Loader for Single YAML Configuration.
Loads all sources from a single sources.yaml file.
"""

import os
import yaml
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from functools import lru_cache
import threading

logger = logging.getLogger(__name__)

class SourcesLoader:
    """Single-file sources loader with caching."""
    
    def __init__(self, sources_file: Optional[str] = None):
        """Initialize the sources loader."""
        if sources_file is None:
            # Default to the sources.yaml file next to this file
            sources_file = str(Path(__file__).parent / "sources.yaml")
        self.sources_file = Path(sources_file)
        self._cache = None
        self._metadata = None
        self._lock = threading.Lock()
        if not self.sources_file.exists():
            logger.warning(f"Sources file not found: {self.sources_file}")

    @lru_cache(maxsize=1)
    def get_metadata(self) -> Dict[str, Any]:
        """Get basic metadata configuration."""
        if self._metadata is None:
            self._metadata = {
                "version": "4.0",
                "description": "Single YAML-based AI/ML news aggregation configuration"
            }
        return self._metadata

    def _load_sources(self) -> Dict[str, Dict[str, Any]]:
        """Load all sources from the single YAML file."""
        if not self.sources_file.exists():
            logger.error(f"Sources file not found: {self.sources_file}")
            return {}
        try:
            with open(self.sources_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            sources = data.get("sources", {})
            enabled_sources = {k: v for k, v in sources.items() if v.get("enabled", True)}
            return enabled_sources
        except Exception as e:
            logger.error(f"Failed to load sources from {self.sources_file}: {e}")
            return {}

    def get_sources(self, enabled_only: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get all sources from the single YAML file."""
        with self._lock:
            if self._cache is not None:
                return self._cache
            sources = self._load_sources()
            self._cache = sources
            return sources

    def get_sources_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all enabled sources for a specific category (from single YAML)."""
        sources = self.get_sources()
        return {k: v for k, v in sources.items() if v.get("category", "").lower() == category.lower()}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about sources."""
        stats = {
            "total_sources": 0,
            "enabled_sources": 0,
            "by_category": {},
            "by_type": {}
        }
        all_sources = self.get_sources(enabled_only=False)
        stats["total_sources"] = len(all_sources)
        enabled_sources = self.get_sources(enabled_only=True)
        stats["enabled_sources"] = len(enabled_sources)
        for name, config in enabled_sources.items():
            category = config.get("category", "Unknown")
            source_type = config.get("type", "unknown")
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            stats["by_type"][source_type] = stats["by_type"].get(source_type, 0) + 1
        return stats

    def reload_cache(self):
        """Clear cache and force reload of all sources."""
        with self._lock:
            self._cache = None
            self._metadata = None
        self.get_metadata.cache_clear()
        logger.info("Sources cache cleared and reloaded")

    def export_to_json(self, output_file: Optional[str] = None) -> str:
        """Export all sources to JSON format (for backward compatibility)."""
        all_sources = self.get_sources(enabled_only=False)
        metadata = self.get_metadata()
        json_data = {
            "metadata": metadata,
            "sources": all_sources
        }
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Exported {len(all_sources)} sources to {output_file}")
        return json_str

# Global instance for backward compatibility
def get_sources_loader() -> SourcesLoader:
    """Get the global sources loader instance."""
    global _sources_loader
    if '_sources_loader' not in globals() or globals()['_sources_loader'] is None:
        globals()['_sources_loader'] = SourcesLoader()
    return globals()['_sources_loader']

# Backward compatibility functions
def load_sources() -> Dict[str, Dict[str, Any]]:
    """Load all enabled sources (backward compatibility)."""
    return get_sources_loader().get_sources()

def load_sources_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """Load sources by category (backward compatibility)."""
    return get_sources_loader().get_sources_by_category(category)

def get_sources_config() -> Dict[str, Any]:
    """Get sources configuration in legacy format."""
    loader = get_sources_loader()
    all_sources = loader.get_sources(enabled_only=False)
    metadata = loader.get_metadata()
    return {
        "metadata": metadata,
        "sources": all_sources
    }
