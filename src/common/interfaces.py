"""
Core interfaces and abstract base classes for Hephaestus RSI.

Defines the contracts and interfaces that all components must implement
to ensure consistency, testability, and modularity.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Iterator
from dataclasses import dataclass
from enum import Enum
import asyncio

from config.base_config import HephaestusConfig


class ComponentStatus(str, Enum):
    """Component status enumeration."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class Priority(str, Enum):
    """Priority levels for operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComponentInfo:
    """Information about a component."""
    name: str
    version: str
    status: ComponentStatus
    dependencies: List[str]
    description: str
    metadata: Dict[str, Any]


class Component(ABC):
    """Base interface for all system components."""
    
    @abstractmethod
    def __init__(self, config: HephaestusConfig, **kwargs):
        """Initialize component with configuration."""
        pass
    
    @abstractmethod
    def get_info(self) -> ComponentInfo:
        """Get component information."""
        pass
    
    @abstractmethod
    def get_status(self) -> ComponentStatus:
        """Get current component status."""
        pass
    
    @abstractmethod
    def validate_dependencies(self) -> bool:
        """Validate that all dependencies are available."""
        pass


class AsyncComponent(Component):
    """Base interface for asynchronous components."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component asynchronously."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the component."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the component gracefully."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        pass


class Configurable(ABC):
    """Interface for configurable components."""
    
    @abstractmethod
    def update_config(self, config: HephaestusConfig) -> None:
        """Update component configuration."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        pass
    
    @abstractmethod
    def validate_config(self, config: HephaestusConfig) -> bool:
        """Validate configuration."""
        pass


class Validateable(ABC):
    """Interface for components that can validate data."""
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate input data."""
        pass
    
    @abstractmethod
    def get_validation_errors(self, data: Any) -> List[str]:
        """Get detailed validation errors."""
        pass


# Learning Interfaces

class Learner(ABC):
    """Base interface for learning components."""
    
    @abstractmethod
    def fit(self, X: Any, y: Any) -> None:
        """Train the learner on data."""
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Make predictions on data."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        pass


class Predictor(ABC):
    """Interface for prediction components."""
    
    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction given features."""
        pass
    
    @abstractmethod
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        pass
    
    @abstractmethod
    def get_prediction_confidence(self, features: Dict[str, Any]) -> float:
        """Get confidence score for prediction."""
        pass


class OnlineLearner(Learner):
    """Interface for online learning components."""
    
    @abstractmethod
    def partial_fit(self, X: Any, y: Any) -> None:
        """Incrementally train on new data."""
        pass
    
    @abstractmethod
    def forget(self, X: Any, y: Any) -> None:
        """Forget specific training examples."""
        pass
    
    @abstractmethod
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        pass


class MetaLearner(ABC):
    """Interface for meta-learning components."""
    
    @abstractmethod
    def meta_train(self, tasks: List[Any]) -> None:
        """Meta-train on multiple tasks."""
        pass
    
    @abstractmethod
    def adapt(self, task_data: Any) -> None:
        """Adapt to a new task."""
        pass
    
    @abstractmethod
    def get_adaptation_speed(self) -> float:
        """Get speed of adaptation to new tasks."""
        pass


# Safety Interfaces

class SafetyValidator(ABC):
    """Interface for safety validation components."""
    
    @abstractmethod
    def is_safe(self, operation: Any) -> bool:
        """Check if an operation is safe."""
        pass
    
    @abstractmethod
    def get_safety_score(self, operation: Any) -> float:
        """Get safety score (0.0 to 1.0)."""
        pass
    
    @abstractmethod
    def get_safety_violations(self, operation: Any) -> List[str]:
        """Get list of safety violations."""
        pass


class CircuitBreaker(ABC):
    """Interface for circuit breaker components."""
    
    @abstractmethod
    def call(self, func: callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        pass
    
    @abstractmethod
    def get_state(self) -> str:
        """Get current circuit breaker state."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        pass


class AnomalyDetector(ABC):
    """Interface for anomaly detection components."""
    
    @abstractmethod
    def fit(self, data: Any) -> None:
        """Train anomaly detector on normal data."""
        pass
    
    @abstractmethod
    def detect(self, data: Any) -> bool:
        """Detect if data is anomalous."""
        pass
    
    @abstractmethod
    def get_anomaly_score(self, data: Any) -> float:
        """Get anomaly score (0.0 to 1.0)."""
        pass


# Storage Interfaces

class StateStore(ABC):
    """Interface for state storage components."""
    
    @abstractmethod
    async def save_state(self, state_id: str, state: Any) -> None:
        """Save state with given ID."""
        pass
    
    @abstractmethod
    async def load_state(self, state_id: str) -> Any:
        """Load state by ID."""
        pass
    
    @abstractmethod
    async def list_states(self) -> List[str]:
        """List all stored state IDs."""
        pass
    
    @abstractmethod
    async def delete_state(self, state_id: str) -> None:
        """Delete state by ID."""
        pass


class ModelStore(ABC):
    """Interface for model storage components."""
    
    @abstractmethod
    async def save_model(self, model_id: str, model: Any, metadata: Dict[str, Any]) -> None:
        """Save model with metadata."""
        pass
    
    @abstractmethod
    async def load_model(self, model_id: str) -> Any:
        """Load model by ID."""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all stored models with metadata."""
        pass
    
    @abstractmethod
    async def delete_model(self, model_id: str) -> None:
        """Delete model by ID."""
        pass


class MetricsStore(ABC):
    """Interface for metrics storage components."""
    
    @abstractmethod
    async def record_metric(self, name: str, value: float, timestamp: Optional[float] = None, 
                          tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        pass
    
    @abstractmethod
    async def get_metrics(self, name: str, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Get metrics within time range."""
        pass
    
    @abstractmethod
    async def get_latest_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get latest value for a metric."""
        pass


# Execution Interfaces

class Executor(ABC):
    """Interface for code execution components."""
    
    @abstractmethod
    async def execute(self, code: str, context: Optional[Dict[str, Any]] = None, 
                     timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute code with optional context and timeout."""
        pass
    
    @abstractmethod
    def validate_code(self, code: str) -> bool:
        """Validate code before execution."""
        pass
    
    @abstractmethod
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        pass


class CodeGenerator(ABC):
    """Interface for code generation components."""
    
    @abstractmethod
    def generate_code(self, specification: Dict[str, Any]) -> str:
        """Generate code from specification."""
        pass
    
    @abstractmethod
    def validate_specification(self, specification: Dict[str, Any]) -> bool:
        """Validate code generation specification."""
        pass
    
    @abstractmethod
    def get_generation_confidence(self, specification: Dict[str, Any]) -> float:
        """Get confidence in code generation."""
        pass


class HypothesisExecutor(ABC):
    """Interface for hypothesis execution components."""
    
    @abstractmethod
    async def execute_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a hypothesis and return results."""
        pass
    
    @abstractmethod
    def validate_hypothesis(self, hypothesis: Dict[str, Any]) -> bool:
        """Validate hypothesis before execution."""
        pass
    
    @abstractmethod
    async def rollback_hypothesis(self, execution_id: str) -> bool:
        """Rollback hypothesis execution."""
        pass


# Factory Interfaces

class ComponentFactory(ABC):
    """Interface for component factories."""
    
    @abstractmethod
    def create_component(self, component_type: str, config: HephaestusConfig, **kwargs) -> Component:
        """Create component of specified type."""
        pass
    
    @abstractmethod
    def list_available_components(self) -> List[str]:
        """List available component types."""
        pass
    
    @abstractmethod
    def register_component(self, component_type: str, component_class: type) -> None:
        """Register new component type."""
        pass


# Observer Interfaces

class Observer(ABC):
    """Interface for observer pattern."""
    
    @abstractmethod
    def update(self, event: str, data: Any) -> None:
        """Handle update notification."""
        pass


class Observable(ABC):
    """Interface for observable pattern."""
    
    @abstractmethod
    def add_observer(self, observer: Observer) -> None:
        """Add an observer."""
        pass
    
    @abstractmethod
    def remove_observer(self, observer: Observer) -> None:
        """Remove an observer."""
        pass
    
    @abstractmethod
    def notify_observers(self, event: str, data: Any) -> None:
        """Notify all observers."""
        pass


# Plugin Interfaces

class Plugin(ABC):
    """Interface for plugin components."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version."""
        pass
    
    @abstractmethod
    def initialize(self, config: HephaestusConfig) -> None:
        """Initialize plugin."""
        pass
    
    @abstractmethod
    def finalize(self) -> None:
        """Cleanup plugin resources."""
        pass


class PluginManager(ABC):
    """Interface for plugin management."""
    
    @abstractmethod
    def load_plugin(self, plugin_path: str) -> Plugin:
        """Load plugin from path."""
        pass
    
    @abstractmethod
    def unload_plugin(self, plugin_name: str) -> None:
        """Unload plugin by name."""
        pass
    
    @abstractmethod
    def list_plugins(self) -> List[Plugin]:
        """List all loaded plugins."""
        pass
    
    @abstractmethod
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Get plugin by name."""
        pass