"""
Anomaly detection and behavioral monitoring for RSI system.
Uses PyOD for comprehensive anomaly detection with multiple algorithms.
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
import numpy as np
import pandas as pd

from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.combination import aom, moa, average, maximization
import psutil

from ..core.state import RSIState, RSIStateManager
from ..monitoring.telemetry import trace_operation, record_safety_event
from ..safety.circuits import RSICircuitBreaker, create_safety_circuit
from loguru import logger


class AnomalyType(str, Enum):
    """Types of anomalies detected."""
    BEHAVIORAL = "behavioral"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    SECURITY = "security"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    SYSTEM = "system"


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyAlert:
    """Represents an anomaly detection alert."""
    
    id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float
    description: str
    affected_components: List[str]
    metrics: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "affected_components": self.affected_components,
            "metrics": self.metrics,
            "context": self.context,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }


class BehavioralMonitor:
    """
    Monitors system behavior and detects anomalies.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        anomaly_threshold: float = 0.1,
        algorithm: str = "ecod",
        state_manager: Optional[RSIStateManager] = None,
        circuit_breaker: Optional[RSICircuitBreaker] = None
    ):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.algorithm = algorithm
        self.state_manager = state_manager
        self.circuit_breaker = circuit_breaker or create_safety_circuit()
        
        # Data storage
        self.behavioral_data: deque = deque(maxlen=window_size)
        self.performance_data: deque = deque(maxlen=window_size)
        self.resource_data: deque = deque(maxlen=window_size)
        
        # Anomaly detection models
        self.behavioral_detector = self._create_detector(algorithm)
        self.performance_detector = self._create_detector(algorithm)
        self.resource_detector = self._create_detector(algorithm)
        
        # Model state
        self.models_trained = False
        self.training_data_collected = False
        
        # Alert management
        self.active_alerts: List[AnomalyAlert] = []
        self.alert_history: List[AnomalyAlert] = []
        self.alert_callbacks: List[Callable[[AnomalyAlert], None]] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        logger.info(f"Behavioral Monitor initialized with {algorithm} algorithm")
    
    def _create_detector(self, algorithm: str):
        """Create anomaly detector based on algorithm."""
        detectors = {
            "ecod": ECOD(contamination=self.anomaly_threshold),
            "iforest": IForest(contamination=self.anomaly_threshold, random_state=42),
            "knn": KNN(contamination=self.anomaly_threshold),
            "lof": LOF(contamination=self.anomaly_threshold),
            "ocsvm": OCSVM(contamination=self.anomaly_threshold),
            "pca": PCA(contamination=self.anomaly_threshold)
        }
        
        if algorithm not in detectors:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return detectors[algorithm]
    
    def add_alert_callback(self, callback: Callable[[AnomalyAlert], None]):
        """Add callback for anomaly alerts."""
        self.alert_callbacks.append(callback)
    
    def collect_behavioral_data(self, data: Dict[str, Any]):
        """Collect behavioral data point."""
        timestamp = datetime.now(timezone.utc)
        
        # Normalize data
        normalized_data = self._normalize_behavioral_data(data)
        
        # Store data
        self.behavioral_data.append({
            "timestamp": timestamp,
            "data": normalized_data
        })
        
        # Check if we have enough data to train
        if len(self.behavioral_data) >= 100 and not self.training_data_collected:
            self._train_behavioral_detector()
            self.training_data_collected = True
    
    def collect_performance_data(self, metrics: Dict[str, float]):
        """Collect performance metrics."""
        timestamp = datetime.now(timezone.utc)
        
        # Store data
        self.performance_data.append({
            "timestamp": timestamp,
            "metrics": metrics
        })
        
        # Check for anomalies if model is trained
        if self.models_trained:
            self._check_performance_anomalies(metrics)
    
    def collect_resource_data(self, usage: Dict[str, float]):
        """Collect resource usage data."""
        timestamp = datetime.now(timezone.utc)
        
        # Store data
        self.resource_data.append({
            "timestamp": timestamp,
            "usage": usage
        })
        
        # Check for anomalies if model is trained
        if self.models_trained:
            self._check_resource_anomalies(usage)
    
    def _normalize_behavioral_data(self, data: Dict[str, Any]) -> List[float]:
        """Normalize behavioral data for anomaly detection."""
        normalized = []
        
        # Extract numerical features
        for key, value in data.items():
            if isinstance(value, (int, float)):
                normalized.append(float(value))
            elif isinstance(value, bool):
                normalized.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Simple string hash for categorical data
                normalized.append(float(hash(value) % 1000) / 1000.0)
            elif isinstance(value, list):
                # List length and average if numerical
                normalized.append(float(len(value)))
                if value and isinstance(value[0], (int, float)):
                    normalized.append(float(np.mean(value)))
        
        return normalized
    
    def _train_behavioral_detector(self):
        """Train the behavioral anomaly detector."""
        if len(self.behavioral_data) < 100:
            return
        
        try:
            # Prepare training data
            X = np.array([
                point["data"] for point in list(self.behavioral_data)[-100:]
            ])
            
            # Train detector
            self.behavioral_detector.fit(X)
            
            # Train other detectors if we have enough data
            if len(self.performance_data) >= 100:
                perf_X = np.array([
                    list(point["metrics"].values()) 
                    for point in list(self.performance_data)[-100:]
                ])
                self.performance_detector.fit(perf_X)
            
            if len(self.resource_data) >= 100:
                res_X = np.array([
                    list(point["usage"].values()) 
                    for point in list(self.resource_data)[-100:]
                ])
                self.resource_detector.fit(res_X)
            
            self.models_trained = True
            logger.info("Anomaly detection models trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train anomaly detectors: {e}")
    
    def _check_performance_anomalies(self, metrics: Dict[str, float]):
        """Check for performance anomalies."""
        if not self.models_trained:
            return
        
        try:
            # Prepare data
            X = np.array([list(metrics.values())]).reshape(1, -1)
            
            # Get anomaly score
            anomaly_score = self.performance_detector.decision_function(X)[0]
            is_anomaly = self.performance_detector.predict(X)[0] == 1
            
            if is_anomaly:
                # Determine severity based on score
                severity = self._determine_severity(anomaly_score, AnomalyType.PERFORMANCE)
                
                # Create alert
                alert = AnomalyAlert(
                    id=f"perf_{int(time.time())}",
                    timestamp=datetime.now(timezone.utc),
                    anomaly_type=AnomalyType.PERFORMANCE,
                    severity=severity,
                    confidence=abs(anomaly_score),
                    description=f"Performance anomaly detected: {self._describe_performance_anomaly(metrics)}",
                    affected_components=["performance_monitor"],
                    metrics=metrics.copy()
                )
                
                self._trigger_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to check performance anomalies: {e}")
    
    def _check_resource_anomalies(self, usage: Dict[str, float]):
        """Check for resource usage anomalies."""
        if not self.models_trained:
            return
        
        try:
            # Prepare data
            X = np.array([list(usage.values())]).reshape(1, -1)
            
            # Get anomaly score
            anomaly_score = self.resource_detector.decision_function(X)[0]
            is_anomaly = self.resource_detector.predict(X)[0] == 1
            
            if is_anomaly:
                # Determine severity
                severity = self._determine_severity(anomaly_score, AnomalyType.RESOURCE)
                
                # Create alert
                alert = AnomalyAlert(
                    id=f"res_{int(time.time())}",
                    timestamp=datetime.now(timezone.utc),
                    anomaly_type=AnomalyType.RESOURCE,
                    severity=severity,
                    confidence=abs(anomaly_score),
                    description=f"Resource anomaly detected: {self._describe_resource_anomaly(usage)}",
                    affected_components=["resource_monitor"],
                    metrics=usage.copy()
                )
                
                self._trigger_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to check resource anomalies: {e}")
    
    def _determine_severity(self, score: float, anomaly_type: AnomalyType) -> AnomalySeverity:
        """Determine anomaly severity based on score."""
        abs_score = abs(score)
        
        if anomaly_type == AnomalyType.SECURITY:
            # Security anomalies are always treated as high severity
            return AnomalySeverity.CRITICAL if abs_score > 0.5 else AnomalySeverity.HIGH
        elif anomaly_type == AnomalyType.RESOURCE:
            # Resource anomalies can be critical
            if abs_score > 0.8:
                return AnomalySeverity.CRITICAL
            elif abs_score > 0.5:
                return AnomalySeverity.HIGH
            elif abs_score > 0.3:
                return AnomalySeverity.MEDIUM
            else:
                return AnomalySeverity.LOW
        else:
            # General severity mapping
            if abs_score > 0.7:
                return AnomalySeverity.HIGH
            elif abs_score > 0.4:
                return AnomalySeverity.MEDIUM
            else:
                return AnomalySeverity.LOW
    
    def _describe_performance_anomaly(self, metrics: Dict[str, float]) -> str:
        """Describe performance anomaly."""
        anomalous_metrics = []
        
        if "accuracy" in metrics and metrics["accuracy"] < 0.5:
            anomalous_metrics.append("low accuracy")
        if "response_time" in metrics and metrics["response_time"] > 1000:
            anomalous_metrics.append("high response time")
        if "error_rate" in metrics and metrics["error_rate"] > 0.1:
            anomalous_metrics.append("high error rate")
        
        if anomalous_metrics:
            return ", ".join(anomalous_metrics)
        else:
            return "unusual performance pattern"
    
    def _describe_resource_anomaly(self, usage: Dict[str, float]) -> str:
        """Describe resource anomaly."""
        anomalous_resources = []
        
        if "cpu_percent" in usage and usage["cpu_percent"] > 90:
            anomalous_resources.append("high CPU usage")
        if "memory_percent" in usage and usage["memory_percent"] > 90:
            anomalous_resources.append("high memory usage")
        if "disk_usage" in usage and usage["disk_usage"] > 95:
            anomalous_resources.append("high disk usage")
        
        if anomalous_resources:
            return ", ".join(anomalous_resources)
        else:
            return "unusual resource usage pattern"
    
    def _trigger_alert(self, alert: AnomalyAlert):
        """Trigger an anomaly alert."""
        # Add to active alerts
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Keep only recent alerts in history
        if len(self.alert_history) > 10000:
            self.alert_history = self.alert_history[-10000:]
        
        # Record safety event
        record_safety_event(
            f"anomaly_detected_{alert.anomaly_type.value}",
            alert.severity.value,
            alert.to_dict()
        )
        
        # Call callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"Anomaly alert triggered: {alert.description}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolution_time = datetime.now(timezone.utc)
                self.active_alerts.remove(alert)
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def get_active_alerts(self) -> List[AnomalyAlert]:
        """Get all active alerts."""
        return self.active_alerts.copy()
    
    def get_alert_history(self, limit: int = 100) -> List[AnomalyAlert]:
        """Get alert history."""
        return self.alert_history[-limit:]
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started anomaly monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped anomaly monitoring")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Auto-resolve old alerts
                self._auto_resolve_alerts()
                
                # Sleep
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self):
        """Collect system metrics automatically."""
        try:
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get network I/O
            net_io = psutil.net_io_counters()
            
            # Collect resource data
            resource_usage = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_usage": disk.percent,
                "network_bytes_sent": net_io.bytes_sent,
                "network_bytes_recv": net_io.bytes_recv
            }
            
            self.collect_resource_data(resource_usage)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _auto_resolve_alerts(self):
        """Auto-resolve alerts that are older than threshold."""
        current_time = datetime.now(timezone.utc)
        auto_resolve_threshold = timedelta(hours=24)
        
        alerts_to_resolve = []
        for alert in self.active_alerts:
            if current_time - alert.timestamp > auto_resolve_threshold:
                alerts_to_resolve.append(alert.id)
        
        for alert_id in alerts_to_resolve:
            self.resolve_alert(alert_id)
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "monitoring_active": self.monitoring_active,
            "models_trained": self.models_trained,
            "behavioral_data_points": len(self.behavioral_data),
            "performance_data_points": len(self.performance_data),
            "resource_data_points": len(self.resource_data),
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "alert_types": {
                anomaly_type.value: len([
                    a for a in self.alert_history 
                    if a.anomaly_type == anomaly_type
                ])
                for anomaly_type in AnomalyType
            },
            "alert_severities": {
                severity.value: len([
                    a for a in self.alert_history 
                    if a.severity == severity
                ])
                for severity in AnomalySeverity
            }
        }


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector using multiple algorithms.
    """
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.detectors = {
            "ecod": ECOD(contamination=contamination),
            "iforest": IForest(contamination=contamination, random_state=42),
            "knn": KNN(contamination=contamination),
            "lof": LOF(contamination=contamination),
            "ocsvm": OCSVM(contamination=contamination)
        }
        self.fitted = False
    
    def fit(self, X: np.ndarray):
        """Fit all detectors."""
        for name, detector in self.detectors.items():
            try:
                detector.fit(X)
            except Exception as e:
                logger.error(f"Failed to fit {name} detector: {e}")
        
        self.fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble."""
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = []
        for name, detector in self.detectors.items():
            try:
                pred = detector.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to predict with {name} detector: {e}")
        
        if not predictions:
            return np.zeros(X.shape[0])
        
        # Use majority voting
        ensemble_pred = np.array(predictions).mean(axis=0)
        return (ensemble_pred > 0.5).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get decision scores using ensemble."""
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        scores = []
        for name, detector in self.detectors.items():
            try:
                score = detector.decision_function(X)
                scores.append(score)
            except Exception as e:
                logger.error(f"Failed to get scores from {name} detector: {e}")
        
        if not scores:
            return np.zeros(X.shape[0])
        
        # Use average combination
        return np.array(scores).mean(axis=0)


# Factory functions
def create_behavioral_monitor(
    state_manager: Optional[RSIStateManager] = None,
    algorithm: str = "ecod",
    threshold: float = 0.1
) -> BehavioralMonitor:
    """Create behavioral monitor with default settings."""
    return BehavioralMonitor(
        algorithm=algorithm,
        anomaly_threshold=threshold,
        state_manager=state_manager
    )


def create_ensemble_monitor(
    state_manager: Optional[RSIStateManager] = None,
    threshold: float = 0.1
) -> BehavioralMonitor:
    """Create behavioral monitor with ensemble detector."""
    monitor = BehavioralMonitor(
        algorithm="ecod",  # Will be replaced with ensemble
        anomaly_threshold=threshold,
        state_manager=state_manager
    )
    
    # Replace with ensemble detector
    monitor.behavioral_detector = EnsembleAnomalyDetector(contamination=threshold)
    monitor.performance_detector = EnsembleAnomalyDetector(contamination=threshold)
    monitor.resource_detector = EnsembleAnomalyDetector(contamination=threshold)
    
    return monitor


def create_security_monitor(
    state_manager: Optional[RSIStateManager] = None
) -> BehavioralMonitor:
    """Create monitor focused on security anomalies."""
    return BehavioralMonitor(
        algorithm="ocsvm",  # Good for security anomalies
        anomaly_threshold=0.05,  # Lower threshold for security
        state_manager=state_manager
    )
