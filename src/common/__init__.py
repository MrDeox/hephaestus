"""
Common utilities and shared components for Hephaestus RSI.

This module provides essential utilities, exceptions, and shared functionality
used throughout the RSI system.
"""

# Core exceptions - comprehensive
from .exceptions import (
    HephaestusError,
    ValidationError as HephaestusValidationError,
    SecurityViolationError,
    DeploymentError,
    FeatureFlagError,
    ComponentError,
    LearningError,
    SafetyError,
    ExecutionError,
    StorageError,
    ResourceError
)

# Essential utilities only
from .utils import (
    generate_id,
    hash_data,
    ensure_directory,
    get_timestamp,
    format_bytes,
    format_duration
)

__all__ = [
    # Core exceptions
    'HephaestusError',
    'HephaestusValidationError', 
    'SecurityViolationError',
    'DeploymentError',
    'FeatureFlagError',
    'ComponentError',
    'LearningError',
    'SafetyError',
    'ExecutionError',
    'StorageError',
    'ResourceError',
    
    # Essential utilities
    'generate_id',
    'hash_data',
    'ensure_directory',
    'get_timestamp',
    'format_bytes',
    'format_duration'
]