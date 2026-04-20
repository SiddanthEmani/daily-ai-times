#!/usr/bin/env python3
"""
Time and date utilities for Daily AI Times
Centralized time-related functions used across the application.
"""

from datetime import datetime, timezone


def get_current_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def get_current_timestamp_for_filename() -> str:
    """Get current timestamp formatted for filenames (YYYYMMDD_HHMMSS)."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 1:
        return f"{seconds:.3f}s"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def validate_timestamp(timestamp_str: str) -> bool:
    """Validate if a string is a valid ISO timestamp."""
    try:
        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return True
    except (ValueError, TypeError):
        return False
