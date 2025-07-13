"""
Common utilities and shared components for Hephaestus RSI.

This module provides reusable components, interfaces, and utilities
that are used across multiple subsystems.
"""

from .interfaces import (
    # Core interfaces
    Component, AsyncComponent, Configurable, Validateable,
    # Learning interfaces  
    Learner, Predictor, OnlineLearner, MetaLearner,
    # Safety interfaces
    SafetyValidator, CircuitBreaker, AnomalyDetector,
    # Storage interfaces
    StateStore, ModelStore, MetricsStore,
    # Execution interfaces
    Executor, CodeGenerator, HypothesisExecutor
)

from .exceptions import (
    # Base exceptions
    HephaestusError, ConfigurationError, ValidationError,
    # Component exceptions
    ComponentError, ComponentNotFoundError, ComponentInitializationError,
    # Learning exceptions
    LearningError, ModelError, PredictionError,
    # Safety exceptions
    SafetyError, CircuitBreakerOpenError, AnomalyDetectedError,
    # Execution exceptions
    ExecutionError, TimeoutError, SandboxError
)

from .utils import (
    # Configuration utilities
    get_config, set_config, load_config,
    # Logging utilities
    get_logger, setup_logging, log_performance,
    # Performance utilities
    measure_time, measure_memory, ProfileContext,
    # Validation utilities
    validate_input, validate_output, ensure_type,
    # File utilities
    ensure_directory, safe_path, atomic_write
)

from .decorators import (
    # Performance decorators
    timed, cached, rate_limited,
    # Safety decorators
    circuit_breaker, validated, logged,
    # Async decorators
    async_timed, async_cached, async_retry
)

from .constants import (
    # System constants
    DEFAULT_TIMEOUT, MAX_RETRIES, DEFAULT_BATCH_SIZE,
    # File constants
    LOG_FORMAT, DATE_FORMAT, CONFIG_FILE_EXTENSIONS,
    # Learning constants
    DEFAULT_LEARNING_RATE, MIN_LEARNING_RATE, MAX_LEARNING_RATE,
    # Safety constants
    CIRCUIT_BREAKER_THRESHOLD, ANOMALY_THRESHOLD, VALIDATION_TIMEOUT
)

__version__ = "1.0.0"
__all__ = [
    # Interfaces
    "Component", "AsyncComponent", "Configurable", "Validateable",
    "Learner", "Predictor", "OnlineLearner", "MetaLearner",
    "SafetyValidator", "CircuitBreaker", "AnomalyDetector",
    "StateStore", "ModelStore", "MetricsStore",
    "Executor", "CodeGenerator", "HypothesisExecutor",
    
    # Exceptions
    "HephaestusError", "ConfigurationError", "ValidationError",
    "ComponentError", "ComponentNotFoundError", "ComponentInitializationError",
    "LearningError", "ModelError", "PredictionError",
    "SafetyError", "CircuitBreakerOpenError", "AnomalyDetectedError",
    "ExecutionError", "TimeoutError", "SandboxError",
    
    # Utilities
    "get_config", "set_config", "load_config",
    "get_logger", "setup_logging", "log_performance",
    "measure_time", "measure_memory", "ProfileContext",
    "validate_input", "validate_output", "ensure_type",
    "ensure_directory", "safe_path", "atomic_write",
    
    # Decorators
    "timed", "cached", "rate_limited",
    "circuit_breaker", "validated", "logged",
    "async_timed", "async_cached", "async_retry",
    
    # Constants
    "DEFAULT_TIMEOUT", "MAX_RETRIES", "DEFAULT_BATCH_SIZE",
    "LOG_FORMAT", "DATE_FORMAT", "CONFIG_FILE_EXTENSIONS",
    "DEFAULT_LEARNING_RATE", "MIN_LEARNING_RATE", "MAX_LEARNING_RATE",
    "CIRCUIT_BREAKER_THRESHOLD", "ANOMALY_THRESHOLD", "VALIDATION_TIMEOUT",
]