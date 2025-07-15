#!/usr/bin/env python3
"""
Enhanced Sources Loader for Modular YAML Configuration.
Provides lazy loading, caching, and efficient source management.
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
    """High-performance modular sources loader with caching and lazy loading."""
    
    def __init__(self, sources_dir: Optional[str] = None):
        """Initialize the sources loader."""
        if sources_dir is None:
            # Default to the sources directory next to this file
            sources_dir = str(Path(__file__).parent / "sources")
        
        self.sources_dir = Path(sources_dir)
        self._cache = {}
        self._metadata = None
        self._lock = threading.Lock()
        
        if not self.sources_dir.exists():
            logger.warning(f"Sources directory not found: {self.sources_dir}")
    
    @lru_cache(maxsize=32)
    def get_metadata(self) -> Dict[str, Any]:
        """Get basic metadata configuration."""
        if self._metadata is None:
            self._metadata = {
                "version": "3.2",
                "description": "Dynamic YAML-based AI/ML news aggregation configuration"
            }
            logger.debug("Loaded basic sources metadata")
        
        return self._metadata
    
    def _discover_categories(self) -> List[str]:
        """Discover available categories by scanning YAML files in sources directory."""
        categories = []
        
        if not self.sources_dir.exists():
            logger.warning(f"Sources directory not found: {self.sources_dir}")
            return categories
        
        try:
            for file_path in self.sources_dir.glob("*.yaml"):
                # Extract category name from filename (remove .yaml extension)
                category = file_path.stem
                categories.append(category)
            
            logger.debug(f"Discovered {len(categories)} categories: {categories}")
            return sorted(categories)
            
        except Exception as e:
            logger.error(f"Failed to discover categories: {e}")
            return categories

    def load_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Load sources for a specific category with caching."""
        with self._lock:
            if category in self._cache:
                return self._cache[category]
            
            # Direct filename mapping based on category name
            category_file = f"{category.lower().replace(' ', '_')}.yaml"
            
            # Load category file
            sources_file = self.sources_dir / category_file
            try:
                with open(sources_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                sources = data.get("sources", {})
                enabled_sources = {k: v for k, v in sources.items() if v.get("enabled", True)}
                
                self._cache[category] = enabled_sources
                logger.debug(f"Loaded {len(enabled_sources)} enabled sources from {category_file}")
                
                return enabled_sources
                
            except Exception as e:
                logger.error(f"Failed to load category {category} from {sources_file}: {e}")
                self._cache[category] = {}
                return {}
    
    def get_sources(self, 
                   category: Optional[str] = None,
                   source_type: Optional[str] = None,
                   enabled_only: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get sources with efficient filtering."""
        
        if category:
            # Load single category
            sources = self.load_category(category)
        else:
            # Load all categories by discovering available YAML files
            sources = {}
            available_categories = self._discover_categories()
            
            for cat in available_categories:
                cat_sources = self.load_category(cat)
                sources.update(cat_sources)
        
        # Apply filters
        filtered_sources = {}
        for name, config in sources.items():
            # Skip disabled sources if enabled_only is True
            if enabled_only and not config.get("enabled", True):
                continue
            
            # Filter by source type
            if source_type and config.get("type") != source_type:
                continue
            
            filtered_sources[name] = config
        
        return filtered_sources
    
    def get_sources_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all enabled sources for a specific category."""
        return self.load_category(category)
    
    def get_sources_by_type(self, source_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all enabled sources of a specific type."""
        return self.get_sources(source_type=source_type)
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return self._discover_categories()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about sources."""
        stats = {
            "total_sources": 0,
            "enabled_sources": 0,
            "by_category": {},
            "by_type": {}
        }
        
        # Load all sources
        all_sources = self.get_sources(enabled_only=False)
        stats["total_sources"] = len(all_sources)
        
        enabled_sources = self.get_sources(enabled_only=True)
        stats["enabled_sources"] = len(enabled_sources)
        
        # Count by category and type
        for name, config in enabled_sources.items():
            category = config.get("category", "Unknown")
            source_type = config.get("type", "unknown")
            
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            stats["by_type"][source_type] = stats["by_type"].get(source_type, 0) + 1
        
        return stats
    
    def reload_cache(self):
        """Clear cache and force reload of all sources."""
        with self._lock:
            self._cache.clear()
            self._metadata = None
            self.get_metadata.cache_clear()
        logger.info("Sources cache cleared and reloaded")
    
    def export_to_json(self, output_file: Optional[str] = None) -> str:
        """Export all sources to JSON format (for backward compatibility)."""
        # Load all sources
        all_sources = {}
        available_categories = self._discover_categories()
        
        for category in available_categories:
            sources = self.load_category(category)
            all_sources.update(sources)
        
        # Create JSON structure similar to original
        metadata = self.get_metadata()
        json_data = {
            "metadata": {
                "version": metadata.get("version", "3.2"),
                "description": metadata.get("description", ""),
                "totalSources": len(all_sources),
                "format": "exported_from_yaml"
            },
            "sources": all_sources
        }
        
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Exported {len(all_sources)} sources to {output_file}")
        
        return json_str

# Global instance for backward compatibility
_sources_loader = None

def get_sources_loader() -> SourcesLoader:
    """Get the global sources loader instance."""
    global _sources_loader
    if _sources_loader is None:
        _sources_loader = SourcesLoader()
    return _sources_loader

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
