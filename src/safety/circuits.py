"""
Circuit breaker implementation for RSI system safety.
Prevents cascading failures and provides graceful degradation.
"""

import asyncio
import time
from typing import Any, Dict, Optional, Callable, List, Union
from enum import Enum
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import pybreaker
import redis
from loguru import logger
from pydantic import BaseModel


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureType(str, Enum):
    """Types of failures that can trigger circuit breakers."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SAFETY_VIOLATION = "safety_violation"


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker behavior."""
    fail_max: int = 5
    reset_timeout: int = 60
    expected_exception: Optional[type] = None
    excluded_exceptions: List[type] = field(default_factory=list)
    name: Optional[str] = None


class CircuitMetrics(BaseModel):
    """Metrics for circuit breaker monitoring."""
    
    circuit_name: str
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_calls: int = 0
    avg_response_time: float = 0.0
    
    class Config:
        use_enum_values = True


class RSICircuitBreaker:
    """
    Advanced circuit breaker for RSI system with Redis backing
    and comprehensive monitoring.
    """
    
    def __init__(
        self,
        config: CircuitConfig,
        redis_client: Optional[redis.Redis] = None,
        metrics_callback: Optional[Callable[[CircuitMetrics], None]] = None
    ):
        self.config = config
        self.redis_client = redis_client
        self.metrics_callback = metrics_callback
        
        # Initialize pybreaker with configuration
        self.breaker = pybreaker.CircuitBreaker(
            fail_max=config.fail_max,
            reset_timeout=config.reset_timeout,
            exclude=config.excluded_exceptions,
            name=config.name or "RSI_Circuit"
        )
        
        # Metrics tracking
        self._metrics = CircuitMetrics(
            circuit_name=config.name or "RSI_Circuit",
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=0
        )
        
        # Response time tracking
        self._response_times: List[float] = []
        self._max_response_times = 1000  # Keep last 1000 response times
        
    @property
    def metrics(self) -> CircuitMetrics:
        """Get current circuit metrics."""
        return self._metrics
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        if self.breaker.current_state == "open":
            return CircuitState.OPEN
        elif self.breaker.current_state == "half-open":
            return CircuitState.HALF_OPEN
        else:
            return CircuitState.CLOSED
    
    def _update_metrics(self, success: bool, response_time: float):
        """Update circuit metrics."""
        self._metrics.total_calls += 1
        self._metrics.state = self.state
        
        if success:
            self._metrics.success_count += 1
            self._metrics.last_success_time = datetime.now(timezone.utc)
        else:
            self._metrics.failure_count += 1
            self._metrics.last_failure_time = datetime.now(timezone.utc)
        
        # Update response time metrics
        self._response_times.append(response_time)
        if len(self._response_times) > self._max_response_times:
            self._response_times.pop(0)
        
        if self._response_times:
            self._metrics.avg_response_time = sum(self._response_times) / len(self._response_times)
        
        # Persist metrics to Redis if available
        if self.redis_client:
            self._persist_metrics()
        
        # Call metrics callback if provided
        if self.metrics_callback:
            self.metrics_callback(self._metrics)
    
    def _persist_metrics(self):
        """Persist metrics to Redis."""
        try:
            key = f"circuit_metrics:{self.config.name}"
            self.redis_client.setex(
                key,
                timedelta(hours=24),
                self._metrics.json()
            )
        except Exception as e:
            logger.warning(f"Failed to persist circuit metrics: {e}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for protecting functions with circuit breaker."""
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func)
        else:
            return self._sync_wrapper(func)
    
    def _sync_wrapper(self, func: Callable) -> Callable:
        """Synchronous wrapper for circuit breaker."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = self.breaker(func)(*args, **kwargs)
                response_time = time.time() - start_time
                self._update_metrics(success=True, response_time=response_time)
                return result
            except Exception as e:
                response_time = time.time() - start_time
                self._update_metrics(success=False, response_time=response_time)
                logger.error(f"Circuit breaker caught exception in {func.__name__}: {e}")
                raise
        return wrapper
    
    def _async_wrapper(self, func: Callable) -> Callable:
        """Asynchronous wrapper for circuit breaker."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                # Note: pybreaker doesn't natively support async, so we implement our own logic
                if self.state == CircuitState.OPEN:
                    if time.time() - self.breaker.last_failure_time < self.config.reset_timeout:
                        raise pybreaker.CircuitBreakerError("Circuit breaker is open")
                    else:
                        # Transition to half-open
                        self.breaker.half_open()
                
                result = await func(*args, **kwargs)
                response_time = time.time() - start_time
                
                # Reset failure count on success
                if self.state == CircuitState.HALF_OPEN:
                    self.breaker.close()
                
                self._update_metrics(success=True, response_time=response_time)
                return result
                
            except Exception as e:
                response_time = time.time() - start_time
                self._update_metrics(success=False, response_time=response_time)
                
                # Increment failure count
                self.breaker.last_failure_time = time.time()
                self.breaker.failure_count += 1
                
                if self.breaker.failure_count >= self.config.fail_max:
                    self.breaker.open()
                
                logger.error(f"Circuit breaker caught exception in {func.__name__}: {e}")
                raise
        
        return wrapper


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers in the RSI system.
    Provides centralized configuration and monitoring.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.circuits: Dict[str, RSICircuitBreaker] = {}
        self.global_metrics: Dict[str, CircuitMetrics] = {}
        
    def create_circuit(
        self,
        name: str,
        config: CircuitConfig,
        metrics_callback: Optional[Callable[[CircuitMetrics], None]] = None
    ) -> RSICircuitBreaker:
        """Create a new circuit breaker."""
        config.name = name
        
        def combined_callback(metrics: CircuitMetrics):
            self.global_metrics[name] = metrics
            if metrics_callback:
                metrics_callback(metrics)
        
        circuit = RSICircuitBreaker(
            config=config,
            redis_client=self.redis_client,
            metrics_callback=combined_callback
        )
        
        self.circuits[name] = circuit
        return circuit
    
    def get_circuit(self, name: str) -> Optional[RSICircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.circuits.get(name)
    
    def get_all_metrics(self) -> Dict[str, CircuitMetrics]:
        """Get metrics for all circuits."""
        return self.global_metrics.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all circuits."""
        health = {
            "total_circuits": len(self.circuits),
            "open_circuits": 0,
            "half_open_circuits": 0,
            "closed_circuits": 0,
            "circuits": {}
        }
        
        for name, circuit in self.circuits.items():
            state = circuit.state
            metrics = circuit.metrics
            
            health["circuits"][name] = {
                "state": state.value,
                "failure_count": metrics.failure_count,
                "success_count": metrics.success_count,
                "avg_response_time": metrics.avg_response_time
            }
            
            if state == CircuitState.OPEN:
                health["open_circuits"] += 1
            elif state == CircuitState.HALF_OPEN:
                health["half_open_circuits"] += 1
            else:
                health["closed_circuits"] += 1
        
        return health
    
    def emergency_open_all(self):
        """Emergency open all circuits (safety measure)."""
        logger.critical("Emergency opening all circuits")
        for circuit in self.circuits.values():
            circuit.breaker.open()
    
    def reset_all_circuits(self):
        """Reset all circuits to closed state."""
        logger.info("Resetting all circuits")
        for circuit in self.circuits.values():
            circuit.breaker.close()


# Pre-configured circuit breakers for common RSI operations
def create_model_update_circuit(redis_client: Optional[redis.Redis] = None) -> RSICircuitBreaker:
    """Create circuit breaker for model update operations."""
    config = CircuitConfig(
        fail_max=3,
        reset_timeout=300,  # 5 minutes
        name="model_update",
        excluded_exceptions=[ValueError]  # Allow validation errors through
    )
    return RSICircuitBreaker(config, redis_client)


def create_learning_circuit(redis_client: Optional[redis.Redis] = None) -> RSICircuitBreaker:
    """Create circuit breaker for learning operations."""
    config = CircuitConfig(
        fail_max=10,
        reset_timeout=120,  # 2 minutes
        name="learning",
        excluded_exceptions=[ValueError, TypeError]
    )
    return RSICircuitBreaker(config, redis_client)


def create_safety_circuit(redis_client: Optional[redis.Redis] = None) -> RSICircuitBreaker:
    """Create circuit breaker for safety validations."""
    config = CircuitConfig(
        fail_max=1,  # Very low tolerance for safety failures
        reset_timeout=600,  # 10 minutes
        name="safety_validation"
    )
    return RSICircuitBreaker(config, redis_client)


def create_database_circuit(redis_client: Optional[redis.Redis] = None) -> RSICircuitBreaker:
    """Create circuit breaker for database operations."""
    config = CircuitConfig(
        fail_max=5,
        reset_timeout=60,
        name="database",
        excluded_exceptions=[ValueError]
    )
    return RSICircuitBreaker(config, redis_client)


# Context manager for temporary circuit bypassing (use with extreme caution)
@asynccontextmanager
async def bypass_circuit(circuit: RSICircuitBreaker):
    """
    Temporarily bypass a circuit breaker.
    WARNING: Use only for critical operations that must succeed.
    """
    original_state = circuit.breaker.current_state
    try:
        circuit.breaker.close()
        yield
    finally:
        if original_state == "open":
            circuit.breaker.open()
        elif original_state == "half-open":
            circuit.breaker.half_open()