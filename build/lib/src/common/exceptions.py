"""
Comprehensive exception hierarchy for Hephaestus RSI.

Provides structured error handling with context, recovery mechanisms,
and detailed error reporting for debugging and monitoring.
"""

from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime, timezone
import traceback
import uuid


class HephaestusError(Exception):
    """Base exception for all Hephaestus RSI errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        severity: str = "medium",
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.recoverable = recoverable
        self.severity = severity  # low, medium, high, critical
        self.cause = cause
        self.timestamp = datetime.now(timezone.utc)
        self.error_id = str(uuid.uuid4())[:8]
        
        # Add traceback information
        self.traceback_info = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_info
        }
    
    def __str__(self) -> str:
        """String representation with context."""
        context_str = f" | Context: {self.context}" if self.context else ""
        cause_str = f" | Caused by: {self.cause}" if self.cause else ""
        return f"[{self.error_code}:{self.error_id}] {self.message}{context_str}{cause_str}"


# Configuration Exceptions

class ConfigurationError(HephaestusError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key
        super().__init__(message, context=context, **kwargs)


class ValidationError(HephaestusError):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[List[str]] = None,
        field_name: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if validation_errors:
            context["validation_errors"] = validation_errors
        if field_name:
            context["field_name"] = field_name
        super().__init__(message, context=context, **kwargs)


# Component Exceptions

class ComponentError(HephaestusError):
    """Base exception for component-related errors."""
    
    def __init__(self, message: str, component_name: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if component_name:
            context["component_name"] = component_name
        super().__init__(message, context=context, **kwargs)


class ComponentNotFoundError(ComponentError):
    """Raised when a required component is not found."""
    
    def __init__(self, component_name: str, **kwargs):
        message = f"Component '{component_name}' not found or not available"
        super().__init__(message, component_name=component_name, **kwargs)


class ComponentInitializationError(ComponentError):
    """Raised when component initialization fails."""
    
    def __init__(self, component_name: str, initialization_error: Optional[str] = None, **kwargs):
        message = f"Failed to initialize component '{component_name}'"
        if initialization_error:
            message += f": {initialization_error}"
        
        context = kwargs.get("context", {})
        if initialization_error:
            context["initialization_error"] = initialization_error
        
        super().__init__(message, component_name=component_name, context=context, **kwargs)


class ComponentDependencyError(ComponentError):
    """Raised when component dependencies are not satisfied."""
    
    def __init__(
        self,
        component_name: str,
        missing_dependencies: List[str],
        **kwargs
    ):
        message = f"Component '{component_name}' missing dependencies: {', '.join(missing_dependencies)}"
        context = kwargs.get("context", {})
        context["missing_dependencies"] = missing_dependencies
        
        super().__init__(message, component_name=component_name, context=context, **kwargs)


# Learning Exceptions

class LearningError(HephaestusError):
    """Base exception for learning-related errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if model_name:
            context["model_name"] = model_name
        super().__init__(message, context=context, **kwargs)


class ModelError(LearningError):
    """Raised when model operations fail."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if operation:
            context["operation"] = operation
        super().__init__(message, model_name=model_name, context=context, **kwargs)


class PredictionError(LearningError):
    """Raised when prediction fails."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if input_shape:
            context["input_shape"] = input_shape
        super().__init__(message, model_name=model_name, context=context, **kwargs)


class TrainingError(LearningError):
    """Raised when model training fails."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        epoch: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if epoch is not None:
            context["epoch"] = epoch
        super().__init__(message, model_name=model_name, context=context, **kwargs)


class MetaLearningError(LearningError):
    """Raised when meta-learning operations fail."""
    
    def __init__(
        self,
        message: str,
        meta_learning_step: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if meta_learning_step:
            context["meta_learning_step"] = meta_learning_step
        super().__init__(message, context=context, **kwargs)


# Safety Exceptions

class SafetyError(HephaestusError):
    """Base exception for safety-related errors."""
    
    def __init__(self, message: str, safety_level: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if safety_level:
            context["safety_level"] = safety_level
        # Safety errors are typically not recoverable
        kwargs.setdefault("recoverable", False)
        kwargs.setdefault("severity", "high")
        super().__init__(message, context=context, **kwargs)


class CircuitBreakerOpenError(SafetyError):
    """Raised when circuit breaker is open and operation is blocked."""
    
    def __init__(
        self,
        circuit_name: str,
        failure_count: Optional[int] = None,
        **kwargs
    ):
        message = f"Circuit breaker '{circuit_name}' is open"
        if failure_count:
            message += f" (failure count: {failure_count})"
        
        context = kwargs.get("context", {})
        context["circuit_name"] = circuit_name
        if failure_count is not None:
            context["failure_count"] = failure_count
        
        super().__init__(message, context=context, **kwargs)


class AnomalyDetectedError(SafetyError):
    """Raised when an anomaly is detected and operation should be blocked."""
    
    def __init__(
        self,
        message: str,
        anomaly_score: Optional[float] = None,
        anomaly_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if anomaly_score is not None:
            context["anomaly_score"] = anomaly_score
        if anomaly_type:
            context["anomaly_type"] = anomaly_type
        
        super().__init__(message, context=context, **kwargs)


class SecurityViolationError(SafetyError):
    """Raised when a security violation is detected."""
    
    def __init__(
        self,
        message: str,
        violation_type: Optional[str] = None,
        severity_level: str = "critical",
        **kwargs
    ):
        context = kwargs.get("context", {})
        if violation_type:
            context["violation_type"] = violation_type
        
        super().__init__(
            message, 
            context=context, 
            severity=severity_level,
            recoverable=False,
            **kwargs
        )


# Execution Exceptions

class ExecutionError(HephaestusError):
    """Base exception for execution-related errors."""
    
    def __init__(
        self,
        message: str,
        execution_id: Optional[str] = None,
        code_snippet: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if execution_id:
            context["execution_id"] = execution_id
        if code_snippet:
            context["code_snippet"] = code_snippet
        super().__init__(message, context=context, **kwargs)


class TimeoutError(ExecutionError):
    """Raised when an operation times out."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        if operation:
            context["operation"] = operation
        super().__init__(message, context=context, **kwargs)


class SandboxError(ExecutionError):
    """Raised when sandbox execution fails."""
    
    def __init__(
        self,
        message: str,
        sandbox_type: Optional[str] = None,
        resource_limits: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if sandbox_type:
            context["sandbox_type"] = sandbox_type
        if resource_limits:
            context["resource_limits"] = resource_limits
        super().__init__(message, context=context, **kwargs)


class CodeGenerationError(ExecutionError):
    """Raised when code generation fails."""
    
    def __init__(
        self,
        message: str,
        specification: Optional[Dict[str, Any]] = None,
        generation_step: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if specification:
            context["specification"] = specification
        if generation_step:
            context["generation_step"] = generation_step
        super().__init__(message, context=context, **kwargs)


# Storage Exceptions

class StorageError(HephaestusError):
    """Base exception for storage-related errors."""
    
    def __init__(
        self,
        message: str,
        storage_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if storage_type:
            context["storage_type"] = storage_type
        if operation:
            context["operation"] = operation
        super().__init__(message, context=context, **kwargs)


class StateNotFoundError(StorageError):
    """Raised when requested state is not found."""
    
    def __init__(self, state_id: str, **kwargs):
        message = f"State '{state_id}' not found"
        context = kwargs.get("context", {})
        context["state_id"] = state_id
        super().__init__(message, context=context, **kwargs)


class ModelNotFoundError(StorageError):
    """Raised when requested model is not found."""
    
    def __init__(self, model_id: str, **kwargs):
        message = f"Model '{model_id}' not found"
        context = kwargs.get("context", {})
        context["model_id"] = model_id
        super().__init__(message, context=context, **kwargs)


class CorruptedDataError(StorageError):
    """Raised when stored data is corrupted."""
    
    def __init__(
        self,
        message: str,
        data_id: Optional[str] = None,
        checksum_mismatch: bool = False,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if data_id:
            context["data_id"] = data_id
        context["checksum_mismatch"] = checksum_mismatch
        
        super().__init__(
            message, 
            context=context, 
            recoverable=False,
            severity="high",
            **kwargs
        )


# Network and Communication Exceptions

class NetworkError(HephaestusError):
    """Base exception for network-related errors."""
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if endpoint:
            context["endpoint"] = endpoint
        if status_code is not None:
            context["status_code"] = status_code
        super().__init__(message, context=context, **kwargs)


class ConnectionError(NetworkError):
    """Raised when connection fails."""
    
    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if host:
            context["host"] = host
        if port is not None:
            context["port"] = port
        super().__init__(message, context=context, **kwargs)


class APIError(NetworkError):
    """Raised when API calls fail."""
    
    def __init__(
        self,
        message: str,
        api_endpoint: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if api_endpoint:
            context["api_endpoint"] = api_endpoint
        if response_data:
            context["response_data"] = response_data
        super().__init__(message, context=context, **kwargs)


# Resource Exceptions

class ResourceError(HephaestusError):
    """Base exception for resource-related errors."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        if resource_type:
            context["resource_type"] = resource_type
        if current_usage:
            context["current_usage"] = current_usage
        super().__init__(message, context=context, **kwargs)


class ResourceExhaustionError(ResourceError):
    """Raised when system resources are exhausted."""
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        limit: Optional[Any] = None,
        current_usage: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get("context", {})
        context["resource_type"] = resource_type
        if limit is not None:
            context["limit"] = limit
        if current_usage is not None:
            context["current_usage"] = current_usage
        
        super().__init__(
            message, 
            context=context,
            severity="high",
            **kwargs
        )


class MemoryError(ResourceExhaustionError):
    """Raised when memory is exhausted."""
    
    def __init__(
        self,
        message: str = "Memory exhausted",
        memory_limit: Optional[int] = None,
        current_memory: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            resource_type="memory",
            limit=memory_limit,
            current_usage=current_memory,
            **kwargs
        )


# Exception Handling Utilities

class ExceptionHandler:
    """Utility class for handling exceptions with recovery strategies."""
    
    def __init__(self):
        self.handlers: Dict[Type[Exception], callable] = {}
        self.default_handler: Optional[callable] = None
    
    def register_handler(
        self, 
        exception_type: Type[Exception], 
        handler: callable
    ) -> None:
        """Register exception handler for specific exception type."""
        self.handlers[exception_type] = handler
    
    def set_default_handler(self, handler: callable) -> None:
        """Set default handler for unregistered exception types."""
        self.default_handler = handler
    
    def handle(self, exception: Exception) -> Any:
        """Handle exception using registered handlers."""
        exception_type = type(exception)
        
        # Look for exact match first
        if exception_type in self.handlers:
            return self.handlers[exception_type](exception)
        
        # Look for parent class matches
        for registered_type, handler in self.handlers.items():
            if isinstance(exception, registered_type):
                return handler(exception)
        
        # Use default handler if available
        if self.default_handler:
            return self.default_handler(exception)
        
        # Re-raise if no handler found
        raise exception


def create_error_context(
    operation: str,
    component: Optional[str] = None,
    **additional_context
) -> Dict[str, Any]:
    """Create error context dictionary."""
    context = {
        "operation": operation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **additional_context
    }
    
    if component:
        context["component"] = component
    
    return context