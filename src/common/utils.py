"""
Common utilities for Hephaestus RSI system.

Provides utility functions used across the system for various tasks.
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier."""
    timestamp = str(int(time.time() * 1000000))  # microseconds
    if prefix:
        return f"{prefix}_{timestamp}"
    return timestamp


def hash_data(data: Any, algorithm: str = "sha256") -> str:
    """Generate hash of data."""
    if not isinstance(data, (str, bytes)):
        data = json.dumps(data, sort_keys=True, default=str)
    
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely load JSON data with fallback."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(data: Any, default: Any = None) -> str:
    """Safely dump data to JSON string."""
    try:
        return json.dumps(data, default=str, indent=2)
    except (TypeError, ValueError):
        return str(default) if default is not None else "{}"


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes."""
    try:
        return Path(file_path).stat().st_size
    except (OSError, FileNotFoundError):
        return 0


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    # Remove or replace dangerous characters
    dangerous_chars = '<>:"/\\|?*'
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    filename = filename.strip(' .')
    
    # Ensure not empty
    if not filename:
        filename = "unnamed"
    
    return filename


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    try:
        return datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        return None


def retry_on_exception(
    func,
    exceptions=(Exception,),
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """Retry function on specified exceptions."""
    def wrapper(*args, **kwargs):
        current_delay = delay
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_retries:
                    raise
                
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= backoff
        
        return None
    
    return wrapper


def validate_environment() -> Dict[str, Any]:
    """Validate system environment and return status."""
    status = {
        "python_version": ".".join(map(str, os.sys.version_info[:3])),
        "platform": os.name,
        "working_directory": str(Path.cwd()),
        "available_memory_mb": None,
        "disk_space_mb": None
    }
    
    try:
        import psutil
        status["available_memory_mb"] = psutil.virtual_memory().available // (1024 * 1024)
        status["disk_space_mb"] = psutil.disk_usage('.').free // (1024 * 1024)
    except ImportError:
        pass
    
    return status


def clean_temp_files(temp_dir: Union[str, Path], max_age_hours: int = 24) -> int:
    """Clean temporary files older than specified age."""
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        return 0
    
    cutoff_time = time.time() - (max_age_hours * 3600)
    cleaned_count = 0
    
    try:
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    cleaned_count += 1
                except OSError:
                    pass
    except OSError:
        pass
    
    return cleaned_count