"""
Comprehensive monitoring and observability for RSI system using OpenTelemetry.
Provides distributed tracing, metrics, and structured logging.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timezone
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import psutil
import socket

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
# from opentelemetry.semantic_conventions.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat

from loguru import logger


class MonitoringLevel(str, Enum):
    """Monitoring detail levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    DEBUG = "debug"


class MetricType(str, Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


@dataclass
class RSIMetrics:
    """Core RSI system metrics."""
    
    # Performance metrics
    request_count: int = 0
    request_duration_ms: List[float] = field(default_factory=list)
    error_count: int = 0
    success_rate: float = 0.0
    
    # Learning metrics
    learning_iterations: int = 0
    model_accuracy: float = 0.0
    concept_drift_events: int = 0
    adaptation_speed: float = 0.0
    
    # Safety metrics
    circuit_breaker_trips: int = 0
    validation_failures: int = 0
    security_violations: int = 0
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_bytes: int = 0
    
    # System metrics
    active_connections: int = 0
    queue_depth: int = 0
    cache_hit_rate: float = 0.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TelemetryCollector:
    """
    Centralized telemetry provider for RSI system.
    Manages OpenTelemetry setup and custom metrics collection.
    """
    
    def __init__(
        self,
        service_name: str = "hephaestus-rsi",
        service_version: str = "1.0.0",
        environment: str = "production",
        monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
        jaeger_endpoint: Optional[str] = None,
        enable_console_export: bool = False
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.monitoring_level = monitoring_level
        self.jaeger_endpoint = jaeger_endpoint
        self.enable_console_export = enable_console_export
        
        # Initialize telemetry
        self._setup_tracing()
        self._setup_metrics()
        self._setup_logging()
        
        # Custom metrics
        self.custom_metrics = RSIMetrics()
        self.metrics_lock = threading.Lock()
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Active spans tracking
        self.active_spans = {}
        
        logger.info(f"RSI Telemetry initialized for {service_name} v{service_version}")
    
    def _setup_tracing(self):
        """Setup distributed tracing with OpenTelemetry."""
        # Create tracer provider
        tracer_provider = TracerProvider(
            resource=self._create_resource()
        )
        
        # Setup exporters
        if self.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
                collector_endpoint=self.jaeger_endpoint,
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
        
        if self.enable_console_export:
            from opentelemetry.exporter.console import ConsoleSpanExporter
            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Set up propagators
        set_global_textmap(B3MultiFormat())
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
    
    def _setup_metrics(self):
        """Setup metrics collection."""
        # Create metric reader
        metric_reader = PeriodicExportingMetricReader(
            exporter=self._create_metrics_exporter(),
            export_interval_millis=10000  # 10 seconds
        )
        
        # Create meter provider
        meter_provider = MeterProvider(
            resource=self._create_resource(),
            metric_readers=[metric_reader]
        )
        
        # Set global meter provider
        metrics.set_meter_provider(meter_provider)
        
        # Get meter
        self.meter = metrics.get_meter(__name__)
        
        # Create custom metrics
        self._create_custom_metrics()
    
    def _create_resource(self):
        """Create OpenTelemetry resource."""
        from opentelemetry.sdk.resources import Resource
        
        return Resource.create({
            "service.name": self.service_name,
            "service.version": self.service_version,
            "service.environment": self.environment,
            "host.name": socket.gethostname(),
            "process.pid": str(psutil.Process().pid)
        })
    
    def _create_metrics_exporter(self):
        """Create metrics exporter."""
        if self.enable_console_export:
            from opentelemetry.exporter.console import ConsoleMetricsExporter
            return ConsoleMetricsExporter()
        else:
            # Return console exporter for now
            from opentelemetry.exporter.console import ConsoleMetricsExporter
            return ConsoleMetricsExporter()
    
    def _create_custom_metrics(self):
        """Create custom metrics for RSI system."""
        # Counters
        self.request_counter = self.meter.create_counter(
            name="rsi_requests_total",
            description="Total number of RSI requests",
            unit="1"
        )
        
        self.error_counter = self.meter.create_counter(
            name="rsi_errors_total",
            description="Total number of RSI errors",
            unit="1"
        )
        
        self.learning_iterations_counter = self.meter.create_counter(
            name="rsi_learning_iterations_total",
            description="Total number of learning iterations",
            unit="1"
        )
        
        # Histograms
        self.request_duration_histogram = self.meter.create_histogram(
            name="rsi_request_duration_seconds",
            description="RSI request duration in seconds",
            unit="s"
        )
        
        # Gauges
        self.model_accuracy_gauge = self.meter.create_gauge(
            name="rsi_model_accuracy",
            description="Current model accuracy",
            unit="1"
        )
        
        self.cpu_usage_gauge = self.meter.create_gauge(
            name="rsi_cpu_usage_percent",
            description="CPU usage percentage",
            unit="%"
        )
        
        self.memory_usage_gauge = self.meter.create_gauge(
            name="rsi_memory_usage_mb",
            description="Memory usage in MB",
            unit="MB"
        )
    
    def _setup_logging(self):
        """Setup structured logging integration."""
        # Configure loguru to include trace context
        logger.configure(
            handlers=[
                {
                    "sink": lambda msg: self._log_with_trace_context(msg),
                    "format": "{time} | {level} | {message}",
                    "serialize": True
                }
            ]
        )
    
    def _log_with_trace_context(self, message: str):
        """Log message with trace context."""
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            trace_id = current_span.get_span_context().trace_id
            span_id = current_span.get_span_context().span_id
            
            # Add trace context to log
            enhanced_message = f"[trace_id={trace_id:032x}][span_id={span_id:016x}] {message}"
            print(enhanced_message)
        else:
            print(message)
    
    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True
    ):
        """Context manager for tracing operations."""
        with self.tracer.start_as_current_span(
            operation_name,
            attributes=attributes or {}
        ) as span:
            try:
                # Record span start
                span.set_attribute("operation.start_time", time.time())
                
                # Store span for potential later access
                span_id = span.get_span_context().span_id
                self.active_spans[span_id] = span
                
                yield span
                
                # Record success
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                # Record exception
                if record_exception:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                
                # Update error metrics
                self.error_counter.add(1, {"operation": operation_name})
                
                raise
            finally:
                # Record span end
                span.set_attribute("operation.end_time", time.time())
                
                # Remove from active spans
                if span_id in self.active_spans:
                    del self.active_spans[span_id]
    
    @asynccontextmanager
    async def trace_async_operation(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
        record_exception: bool = True
    ):
        """Async context manager for tracing operations."""
        with self.trace_operation(operation_name, attributes, record_exception) as span:
            yield span
    
    def record_request(
        self,
        operation: str,
        duration_seconds: float,
        success: bool = True,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Record a request with timing and success metrics."""
        # Update counters
        self.request_counter.add(1, {"operation": operation})
        
        if not success:
            self.error_counter.add(1, {"operation": operation})
        
        # Record duration
        self.request_duration_histogram.record(
            duration_seconds,
            {"operation": operation}
        )
        
        # Update custom metrics
        with self.metrics_lock:
            self.custom_metrics.request_count += 1
            self.custom_metrics.request_duration_ms.append(duration_seconds * 1000)
            
            if not success:
                self.custom_metrics.error_count += 1
            
            # Calculate success rate
            if self.custom_metrics.request_count > 0:
                self.custom_metrics.success_rate = (
                    (self.custom_metrics.request_count - self.custom_metrics.error_count) / 
                    self.custom_metrics.request_count
                )
    
    def record_learning_iteration(
        self,
        accuracy: float,
        learning_rate: float,
        concept_drift_detected: bool = False,
        adaptation_speed: float = 0.0
    ):
        """Record learning iteration metrics."""
        # Update counters
        self.learning_iterations_counter.add(1)
        
        # Update gauges
        self.model_accuracy_gauge.set(accuracy)
        
        # Update custom metrics
        with self.metrics_lock:
            self.custom_metrics.learning_iterations += 1
            self.custom_metrics.model_accuracy = accuracy
            self.custom_metrics.adaptation_speed = adaptation_speed
            
            if concept_drift_detected:
                self.custom_metrics.concept_drift_events += 1
    
    def record_safety_event(
        self,
        event_type: str,
        severity: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Record safety-related events."""
        # Create span for safety event
        with self.trace_operation(
            f"safety_event_{event_type}",
            attributes={
                "safety.event_type": event_type,
                "safety.severity": severity,
                **(details or {})
            }
        ):
            # Update custom metrics
            with self.metrics_lock:
                if event_type == "circuit_breaker_trip":
                    self.custom_metrics.circuit_breaker_trips += 1
                elif event_type == "validation_failure":
                    self.custom_metrics.validation_failures += 1
                elif event_type == "security_violation":
                    self.custom_metrics.security_violations += 1
            
            logger.warning(f"Safety event: {event_type} (severity: {severity})", extra=details)
    
    def update_resource_metrics(self):
        """Update system resource metrics."""
        try:
            # Get current resource usage
            resource_data = self.resource_monitor.get_current_usage()
            
            # Update gauges
            self.cpu_usage_gauge.set(resource_data["cpu_percent"])
            self.memory_usage_gauge.set(resource_data["memory_mb"])
            
            # Update custom metrics
            with self.metrics_lock:
                self.custom_metrics.cpu_usage_percent = resource_data["cpu_percent"]
                self.custom_metrics.memory_usage_mb = resource_data["memory_mb"]
                self.custom_metrics.disk_usage_percent = resource_data["disk_percent"]
                self.custom_metrics.network_io_bytes = resource_data["network_io_bytes"]
            
        except Exception as e:
            logger.error(f"Failed to update resource metrics: {e}")
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self.metrics_lock:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": {
                    "name": self.service_name,
                    "version": self.service_version,
                    "environment": self.environment
                },
                "performance": {
                    "request_count": self.custom_metrics.request_count,
                    "error_count": self.custom_metrics.error_count,
                    "success_rate": self.custom_metrics.success_rate,
                    "avg_request_duration_ms": (
                        sum(self.custom_metrics.request_duration_ms) / 
                        len(self.custom_metrics.request_duration_ms)
                        if self.custom_metrics.request_duration_ms else 0
                    )
                },
                "learning": {
                    "iterations": self.custom_metrics.learning_iterations,
                    "accuracy": self.custom_metrics.model_accuracy,
                    "drift_events": self.custom_metrics.concept_drift_events,
                    "adaptation_speed": self.custom_metrics.adaptation_speed
                },
                "safety": {
                    "circuit_breaker_trips": self.custom_metrics.circuit_breaker_trips,
                    "validation_failures": self.custom_metrics.validation_failures,
                    "security_violations": self.custom_metrics.security_violations
                },
                "resources": {
                    "cpu_usage_percent": self.custom_metrics.cpu_usage_percent,
                    "memory_usage_mb": self.custom_metrics.memory_usage_mb,
                    "disk_usage_percent": self.custom_metrics.disk_usage_percent,
                    "network_io_bytes": self.custom_metrics.network_io_bytes
                }
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check with telemetry data."""
        with self.trace_operation("health_check"):
            snapshot = self.get_metrics_snapshot()
            
            # Determine health status
            health_status = "healthy"
            issues = []
            
            # Check error rate
            if snapshot["performance"]["success_rate"] < 0.95:
                health_status = "degraded"
                issues.append("High error rate")
            
            # Check resource usage
            if snapshot["resources"]["cpu_usage_percent"] > 90:
                health_status = "degraded"
                issues.append("High CPU usage")
            
            if snapshot["resources"]["memory_usage_mb"] > 8192:  # 8GB
                health_status = "degraded"
                issues.append("High memory usage")
            
            # Check safety events
            if snapshot["safety"]["security_violations"] > 0:
                health_status = "unhealthy"
                issues.append("Security violations detected")
            
            return {
                "status": health_status,
                "issues": issues,
                "metrics": snapshot,
                "active_spans": len(self.active_spans)
            }


class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.last_network_io = self.process.io_counters()
        self.last_check_time = time.time()
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = disk_usage.percent
            
            # Network I/O
            current_io = self.process.io_counters()
            current_time = time.time()
            
            time_delta = current_time - self.last_check_time
            bytes_delta = (current_io.read_bytes + current_io.write_bytes) - (
                self.last_network_io.read_bytes + self.last_network_io.write_bytes
            )
            
            network_io_bytes = bytes_delta / time_delta if time_delta > 0 else 0
            
            # Update for next calculation
            self.last_network_io = current_io
            self.last_check_time = current_time
            
            return {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "disk_percent": disk_percent,
                "network_io_bytes": network_io_bytes,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {
                "cpu_percent": 0.0,
                "memory_mb": 0.0,
                "disk_percent": 0.0,
                "network_io_bytes": 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# Global telemetry provider instance
_telemetry_provider: Optional[TelemetryCollector] = None


def initialize_telemetry(
    service_name: str = "hephaestus-rsi",
    service_version: str = "1.0.0",
    environment: str = "production",
    monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
    jaeger_endpoint: Optional[str] = None,
    enable_console_export: bool = False
) -> TelemetryCollector:
    """Initialize global telemetry provider."""
    global _telemetry_provider
    
    _telemetry_provider = RSITelemetryProvider(
        service_name=service_name,
        service_version=service_version,
        environment=environment,
        monitoring_level=monitoring_level,
        jaeger_endpoint=jaeger_endpoint,
        enable_console_export=enable_console_export
    )
    
    return _telemetry_provider


def get_telemetry_provider() -> Optional[TelemetryCollector]:
    """Get the global telemetry provider."""
    return _telemetry_provider


# Convenience functions for common operations
def trace_operation(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Decorator for tracing operations."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                if _telemetry_provider:
                    async with _telemetry_provider.trace_async_operation(
                        operation_name, attributes
                    ):
                        return await func(*args, **kwargs)
                else:
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                if _telemetry_provider:
                    with _telemetry_provider.trace_operation(
                        operation_name, attributes
                    ):
                        return func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


def record_learning_metric(
    accuracy: float,
    learning_rate: float,
    concept_drift_detected: bool = False,
    adaptation_speed: float = 0.0
):
    """Record learning metrics."""
    if _telemetry_provider:
        _telemetry_provider.record_learning_iteration(
            accuracy, learning_rate, concept_drift_detected, adaptation_speed
        )


def record_safety_event(
    event_type: str,
    severity: str,
    details: Optional[Dict[str, Any]] = None
):
    """Record safety events."""
    if _telemetry_provider:
        _telemetry_provider.record_safety_event(event_type, severity, details)