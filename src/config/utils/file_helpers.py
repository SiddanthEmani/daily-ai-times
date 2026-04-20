#!/usr/bin/env python3
"""
File operations and I/O utilities for Daily AI Times
Centralized file handling functions used across the application.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_scripts_test_output_dir() -> Path:
    """Get the scripts/test_output directory path."""
    return Path(__file__).parent.parent.parent.parent / "scripts" / "test_output"


def ensure_directory_exists(directory: Path) -> bool:
    """Ensure a directory exists, create if it doesn't."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        return False


def save_json_file(data: Dict[str, Any], filename: str, directory: Optional[Path] = None) -> bool:
    """
    Save data as JSON file with error handling.
    
    Args:
        data: Data to save as JSON
        filename: Name of the file to save
        directory: Directory to save to (defaults to scripts/test_output)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        target_directory = directory or get_scripts_test_output_dir()
        
        if not ensure_directory_exists(target_directory):
            return False
            
        output_file = target_directory / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved {filename} to: {output_file}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to save {filename}: {e}")
        return False


def load_json_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Load JSON data from file with error handling.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Dict containing the JSON data, or None if failed
    """
    try:
        if not filepath.exists():
            logger.warning(f"JSON file not found: {filepath}")
            return None
            
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except Exception as e:
        logger.error(f"Failed to load JSON file {filepath}: {e}")
        return None


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes."""
    try:
        return filepath.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def safe_filename(filename: str) -> str:
    """Convert string to safe filename by removing/replacing problematic characters."""
    import re
    # Replace problematic characters with underscores
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores and spaces
    return safe_name.strip('_ ')
