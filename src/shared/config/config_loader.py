#!/usr/bin/env python3
"""Unified Configuration Loader - centralized utility for loading YAML configuration files."""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import yaml

logger = logging.getLogger(__name__)
load_dotenv('.env.local')

class ConfigLoader:
    """Unified configuration loader for YAML files."""
    
    _config_cache: Dict[str, Dict[str, Any]] = {}
    
    @staticmethod
    def _get_config_dir() -> Path:
        """Get the configuration directory path."""
        # Use environment variable if set, otherwise use default
        config_dir_str = os.getenv('CONFIG_DIR')
        if config_dir_str:
            config_dir = Path(config_dir_str)
        else:
            # Default to the directory containing this file
            config_dir = Path(__file__).parent
        
        if not config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
        return config_dir
    
    @classmethod
    def _load_file(cls, file_path: Path) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load configuration file {file_path}: {e}")
            raise RuntimeError(f"Could not load configuration from {file_path}: {e}")
    
    @classmethod
    def load_config(cls, config_name: str = "app") -> Dict[str, Any]:
        """Load configuration by name (app, swarm, collectors)."""
        if config_name in cls._config_cache:
            return cls._config_cache[config_name]
        
        config_dir = cls._get_config_dir()
        
        # Try YAML extensions only
        for ext in ['.yaml', '.yml']:
            config_path = config_dir / f"{config_name}{ext}"
            if config_path.exists():
                config = cls._load_file(config_path)
                cls._config_cache[config_name] = config
                logger.debug(f"Loaded {config_name} configuration from {config_path}")
                return config
        
        raise FileNotFoundError(f"No YAML configuration file found for '{config_name}' in {config_dir}")
    
    @classmethod
    def get(cls, key: str, default: Any = None, config_name: str = "app") -> Any:
        """Get a setting using dot notation (e.g., 'collection.max_articles')."""
        config = cls.load_config(config_name)
        value = config
        for k in key.split('.'):
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the configuration cache."""
        cls._config_cache.clear()


# Essential convenience functions (keeping those with specific logic)
def get_collectors_config() -> Dict[str, Any]:
    """Load collector-specific configuration from app config."""
    try:
        return ConfigLoader.get('collectors', {}, "app")
    except Exception as e:
        logger.error(f"Failed to load collector config: {e}")
        return {}

def get_swarm_config() -> Dict[str, Any]:
    """Load swarm intelligence configuration."""
    try:
        return ConfigLoader.load_config("swarm")
    except Exception as e:
        logger.error(f"Failed to load swarm config: {e}")
        return {}


# Convenience functions for sources (keeping existing functionality)
def load_sources_config() -> Dict[str, Any]:
    """Load sources configuration (YAML format)."""
    try:
        # Use YAML modular format
        from .sources_loader import get_sources_loader
        loader = get_sources_loader()
        return {
            "sources": loader.get_sources(),
            "metadata": loader.get_metadata(),
            "format": "yaml_modular"
        }
    except ImportError:
        logger.error("Sources loader not available")
        return {"sources": {}, "format": "none"}

def get_sources_by_category(category: str) -> Dict[str, Any]:
    """Get sources for a specific category."""
    try:
        from .sources_loader import get_sources_loader
        return get_sources_loader().get_sources_by_category(category)
    except ImportError:
        # Fallback to loading all and filtering
        config = load_sources_config()
        sources = config.get("sources", {})
        if isinstance(sources, dict) and "sources" in sources:
            sources = sources["sources"]
        return {k: v for k, v in sources.items() if v.get("category") == category}
