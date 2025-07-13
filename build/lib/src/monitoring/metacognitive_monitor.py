"""
Advanced Metacognitive Monitoring System for RSI AI.
Implements real-time self-assessment capabilities with sub-second latencies.
"""

import asyncio
import psutil
import time
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from loguru import logger
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

class AlertSeverity(Enum):
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"
    DEBUG = "debug"

class SystemHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class MetacognitiveMetrics:
    """Real-time metacognitive assessment metrics"""
    timestamp: float
    confidence_score: float
    uncertainty_level: float
    self_assessment_accuracy: float
    learning_efficiency: float
    adaptation_speed: float
    introspection_depth: float
    cognitive_load: float
    metacognitive_awareness: float

@dataclass
class SystemStateSnapshot:
    """Comprehensive system state snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available: int
    gpu_utilization: Optional[float]
    network_io: Dict[str, int]
    process_memory_mb: float
    thread_count: int
    file_descriptors: int
    model_inference_rate: float
    prediction_confidence: float
    anomaly_score: float

class RSISystemMonitor:
    """High-performance real-time system monitoring with <100ms latency"""
    
    def __init__(self, collection_interval: float = 0.1):
        self.interval = collection_interval
        self.metrics_history: List[SystemStateSnapshot] = []
        self.process = psutil.Process()
        self.monitoring_active = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.performance_baselines = {}
        self.anomaly_thresholds = {
            'cpu_critical': 95.0,
            'memory_critical': 90.0,
            'memory_leak_threshold_mb': 1000.0,
            'inference_rate_drop': 0.5
        }
        
    async def start_monitoring(self):
        """Start real-time monitoring with background collection"""
        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("RSI system monitoring started with {}ms interval", 
                   int(self.interval * 1000))
        
    async def stop_monitoring(self):
        """Stop monitoring and cleanup resources"""
        self.monitoring_active = False
        self.executor.shutdown(wait=True)
        logger.info("RSI system monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop with high-frequency collection"""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Collect metrics asynchronously
                metrics = await self.collect_system_metrics()
                
                # Store in history with size limit
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 1000:  # Keep last 1000 snapshots
                    self.metrics_history.pop(0)
                
                # Detect anomalies in real-time
                anomalies = await self.detect_resource_anomalies(metrics)
                if anomalies:
                    await self._handle_anomalies(anomalies, metrics)
                
                # Maintain target interval
                collection_time = time.time() - start_time
                sleep_time = max(0, self.interval - collection_time)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error("Error in monitoring loop: {}", str(e))
                await asyncio.sleep(self.interval)
    
    async def collect_system_metrics(self) -> SystemStateSnapshot:
        """Collect comprehensive system metrics with <50ms latency"""
        try:
            # Parallel collection for performance
            cpu_task = asyncio.create_task(self._get_cpu_percent())
            memory_task = asyncio.create_task(self._get_memory_info())
            process_task = asyncio.create_task(self._get_process_info())
            network_task = asyncio.create_task(self._get_network_info())
            gpu_task = asyncio.create_task(self._get_gpu_metrics())
            
            # Wait for all metrics
            cpu_percent = await cpu_task
            memory_info = await memory_task
            process_info = await process_task
            network_info = await network_task
            gpu_util = await gpu_task
            
            return SystemStateSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_info['percent'],
                memory_available=memory_info['available'],
                gpu_utilization=gpu_util,
                network_io=network_info,
                process_memory_mb=process_info['memory_mb'],
                thread_count=process_info['threads'],
                file_descriptors=process_info['fds'],
                model_inference_rate=await self._calculate_inference_rate(),
                prediction_confidence=await self._get_avg_prediction_confidence(),
                anomaly_score=await self._calculate_anomaly_score()
            )
            
        except Exception as e:
            logger.warning("Failed to collect some system metrics: {}", str(e))
            return self._get_fallback_metrics()
    
    async def _get_cpu_percent(self) -> float:
        """Get CPU utilization with non-blocking call"""
        def get_cpu():
            return psutil.cpu_percent(interval=None)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, get_cpu)
    
    async def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        def get_memory():
            vm = psutil.virtual_memory()
            return {'percent': vm.percent, 'available': vm.available}
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, get_memory)
    
    async def _get_process_info(self) -> Dict[str, Any]:
        """Get process-specific information"""
        def get_process():
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            threads = self.process.num_threads()
            try:
                fds = self.process.num_fds()
            except AttributeError:
                fds = 0  # Windows doesn't have file descriptors
            
            return {
                'memory_mb': memory_mb,
                'threads': threads,
                'fds': fds
            }
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, get_process)
    
    async def _get_network_info(self) -> Dict[str, int]:
        """Get network I/O information"""
        def get_network():
            net_io = psutil.net_io_counters()
            if net_io:
                return {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            return {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, get_network)
    
    async def _get_gpu_metrics(self) -> Optional[float]:
        """Get GPU utilization if available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except:
            return None
    
    async def _calculate_inference_rate(self) -> float:
        """Calculate current model inference rate"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Simple rate calculation based on timestamp differences
        recent_snapshots = self.metrics_history[-10:]
        if len(recent_snapshots) > 1:
            time_diff = recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp
            if time_diff > 0:
                return len(recent_snapshots) / time_diff
        
        return 0.0
    
    async def _get_avg_prediction_confidence(self) -> float:
        """Get average prediction confidence from recent predictions"""
        # This would integrate with the actual prediction system
        # For now, return a simulated confidence based on system load
        if len(self.metrics_history) > 0:
            latest = self.metrics_history[-1]
            # Lower confidence with higher CPU/memory usage
            base_confidence = 0.9
            cpu_penalty = latest.cpu_percent / 100.0 * 0.2
            memory_penalty = latest.memory_percent / 100.0 * 0.1
            return max(0.5, base_confidence - cpu_penalty - memory_penalty)
        
        return 0.8
    
    async def _calculate_anomaly_score(self) -> float:
        """Calculate real-time anomaly score"""
        if len(self.metrics_history) < 10:
            return 0.0
        
        recent_metrics = self.metrics_history[-10:]
        anomaly_score = 0.0
        
        # Check for anomalous patterns
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        # High variance indicates potential issues
        cpu_variance = np.var(cpu_values)
        memory_variance = np.var(memory_values)
        
        # Normalize to 0-1 scale
        anomaly_score = min(1.0, (cpu_variance + memory_variance) / 1000.0)
        
        return anomaly_score
    
    async def detect_resource_anomalies(self, metrics: SystemStateSnapshot) -> Dict[str, bool]:
        """Real-time anomaly detection with <10ms latency"""
        anomalies = {}
        
        # CPU overload detection
        if metrics.cpu_percent > self.anomaly_thresholds['cpu_critical']:
            anomalies['cpu_overload'] = True
            
        # Memory pressure detection
        if metrics.memory_percent > self.anomaly_thresholds['memory_critical']:
            anomalies['memory_pressure'] = True
            
        # Memory leak detection
        if len(self.metrics_history) > 100:
            recent_memory = [m.process_memory_mb for m in self.metrics_history[-10:]]
            memory_growth = max(recent_memory) - min(recent_memory)
            if memory_growth > self.anomaly_thresholds['memory_leak_threshold_mb']:
                anomalies['memory_leak'] = True
                
        # Inference rate degradation (reduce false positives)
        if (metrics.model_inference_rate > 0 and 
            len(self.metrics_history) > 50):
            historical_rate = np.mean([m.model_inference_rate 
                                     for m in self.metrics_history[-50:-10]])
            # Only trigger for significant degradation with meaningful baseline
            if (historical_rate > 1.0 and 
                metrics.model_inference_rate < historical_rate * 0.5):  # 50% degradation
                anomalies['inference_degradation'] = True
        
        # High anomaly score (reduce sensitivity)
        if metrics.anomaly_score > 0.9:
            anomalies['system_anomaly'] = True
                
        return anomalies
    
    async def _handle_anomalies(self, anomalies: Dict[str, bool], 
                               metrics: SystemStateSnapshot):
        """Handle detected anomalies with appropriate responses"""
        for anomaly_type, detected in anomalies.items():
            if detected:
                severity = self._determine_anomaly_severity(anomaly_type, metrics)
                logger.log(severity.value.upper(), 
                          "Anomaly detected: {} at {}", 
                          anomaly_type, 
                          datetime.fromtimestamp(metrics.timestamp))
                
                # Trigger automatic responses
                await self._trigger_automatic_response(anomaly_type, metrics)
    
    def _determine_anomaly_severity(self, anomaly_type: str, 
                                  metrics: SystemStateSnapshot) -> AlertSeverity:
        """Determine severity level for anomaly"""
        critical_anomalies = ['cpu_overload', 'memory_pressure', 'memory_leak']
        
        if anomaly_type in critical_anomalies:
            return AlertSeverity.CRITICAL
        elif anomaly_type == 'inference_degradation':
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    async def _trigger_automatic_response(self, anomaly_type: str, 
                                        metrics: SystemStateSnapshot):
        """Trigger automatic responses to anomalies"""
        if anomaly_type == 'memory_pressure':
            logger.info("Triggering memory cleanup due to memory pressure")
            # Could trigger garbage collection, cache cleanup, etc.
            
        elif anomaly_type == 'cpu_overload':
            logger.info("Reducing processing load due to CPU overload")
            # Could reduce batch sizes, pause non-critical tasks, etc.
            
        elif anomaly_type == 'inference_degradation':
            logger.info("Investigating inference performance degradation")
            # Could trigger model reloading, optimization, etc.
    
    def _get_fallback_metrics(self) -> SystemStateSnapshot:
        """Return fallback metrics when collection fails"""
        return SystemStateSnapshot(
            timestamp=time.time(),
            cpu_percent=0.0,
            memory_percent=0.0,
            memory_available=0,
            gpu_utilization=None,
            network_io={'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0},
            process_memory_mb=0.0,
            thread_count=0,
            file_descriptors=0,
            model_inference_rate=0.0,
            prediction_confidence=0.0,
            anomaly_score=0.0
        )
    
    def get_system_health_status(self) -> SystemHealth:
        """Assess overall system health"""
        if not self.metrics_history:
            return SystemHealth.UNKNOWN
        
        latest = self.metrics_history[-1]
        
        # Critical conditions
        if (latest.cpu_percent > 95 or 
            latest.memory_percent > 90 or 
            latest.anomaly_score > 0.8):
            return SystemHealth.CRITICAL
        
        # Degraded conditions
        if (latest.cpu_percent > 80 or 
            latest.memory_percent > 75 or 
            latest.prediction_confidence < 0.6):
            return SystemHealth.DEGRADED
        
        return SystemHealth.HEALTHY
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 snapshots
        
        return {
            'avg_cpu_percent': np.mean([m.cpu_percent for m in recent_metrics]),
            'avg_memory_percent': np.mean([m.memory_percent for m in recent_metrics]),
            'avg_inference_rate': np.mean([m.model_inference_rate for m in recent_metrics]),
            'avg_confidence': np.mean([m.prediction_confidence for m in recent_metrics]),
            'avg_anomaly_score': np.mean([m.anomaly_score for m in recent_metrics]),
            'system_health': self.get_system_health_status().value,
            'monitoring_duration_hours': (time.time() - recent_metrics[0].timestamp) / 3600,
            'total_snapshots': len(self.metrics_history)
        }

class MetacognitiveAssessment:
    """Advanced metacognitive assessment system for self-awareness"""
    
    def __init__(self):
        self.assessment_history: List[MetacognitiveMetrics] = []
        self.confidence_model = None
        self.uncertainty_tracker = UncertaintyTracker()
        self.self_monitoring_accuracy = 0.8
        
    async def assess_metacognitive_state(self, 
                                       system_metrics: SystemStateSnapshot,
                                       recent_predictions: List[Dict[str, Any]]) -> MetacognitiveMetrics:
        """Comprehensive metacognitive assessment"""
        
        # Calculate core metacognitive metrics
        confidence_score = await self._calculate_confidence_score(recent_predictions)
        uncertainty_level = await self._assess_uncertainty_level(recent_predictions)
        self_assessment_accuracy = await self._evaluate_self_assessment_accuracy()
        learning_efficiency = await self._measure_learning_efficiency()
        adaptation_speed = await self._calculate_adaptation_speed()
        introspection_depth = await self._assess_introspection_depth()
        cognitive_load = self._calculate_cognitive_load(system_metrics)
        metacognitive_awareness = await self._assess_metacognitive_awareness()
        
        metrics = MetacognitiveMetrics(
            timestamp=time.time(),
            confidence_score=confidence_score,
            uncertainty_level=uncertainty_level,
            self_assessment_accuracy=self_assessment_accuracy,
            learning_efficiency=learning_efficiency,
            adaptation_speed=adaptation_speed,
            introspection_depth=introspection_depth,
            cognitive_load=cognitive_load,
            metacognitive_awareness=metacognitive_awareness
        )
        
        self.assessment_history.append(metrics)
        if len(self.assessment_history) > 1000:
            self.assessment_history.pop(0)
        
        return metrics
    
    async def _calculate_confidence_score(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in recent predictions"""
        if not predictions:
            return 0.5
        
        confidences = [p.get('confidence', 0.5) for p in predictions]
        return np.mean(confidences)
    
    async def _assess_uncertainty_level(self, predictions: List[Dict[str, Any]]) -> float:
        """Assess uncertainty level in decision-making"""
        if not predictions:
            return 0.5
        
        # Calculate uncertainty based on prediction variance and confidence
        confidences = [p.get('confidence', 0.5) for p in predictions]
        confidence_variance = np.var(confidences)
        avg_confidence = np.mean(confidences)
        
        # High variance or low confidence indicates high uncertainty
        uncertainty = 1.0 - avg_confidence + confidence_variance
        return min(1.0, max(0.0, uncertainty))
    
    async def _evaluate_self_assessment_accuracy(self) -> float:
        """Evaluate how accurate the system's self-assessment is"""
        # This would compare predicted confidence with actual accuracy
        # For now, return a moving average with some learning
        if len(self.assessment_history) > 10:
            recent_assessments = self.assessment_history[-10:]
            return np.mean([a.self_assessment_accuracy for a in recent_assessments]) * 0.95 + 0.05
        
        return self.self_monitoring_accuracy
    
    async def _measure_learning_efficiency(self) -> float:
        """Measure how efficiently the system is learning"""
        # This would analyze learning curves and adaptation rates
        # For now, return a simulated efficiency metric
        if len(self.assessment_history) > 20:
            recent_scores = [a.confidence_score for a in self.assessment_history[-20:]]
            # If confidence is improving, learning efficiency is high
            if len(recent_scores) > 10:
                early_avg = np.mean(recent_scores[:10])
                late_avg = np.mean(recent_scores[10:])
                improvement = (late_avg - early_avg) + 0.5
                return min(1.0, max(0.0, improvement))
        
        return 0.7
    
    async def _calculate_adaptation_speed(self) -> float:
        """Calculate how quickly the system adapts to new conditions"""
        # This would measure response time to distribution shifts
        # For now, return based on recent confidence variance
        if len(self.assessment_history) > 10:
            recent_confidences = [a.confidence_score for a in self.assessment_history[-10:]]
            variance = np.var(recent_confidences)
            # Lower variance indicates stable adaptation
            adaptation_speed = 1.0 - min(1.0, variance * 10)
            return max(0.0, adaptation_speed)
        
        return 0.6
    
    async def _assess_introspection_depth(self) -> float:
        """Assess the depth of self-analysis capabilities"""
        # This would measure the system's ability to analyze its own processes
        # For now, return based on assessment history richness
        if len(self.assessment_history) > 50:
            return 0.8
        elif len(self.assessment_history) > 20:
            return 0.6
        else:
            return 0.4
    
    def _calculate_cognitive_load(self, system_metrics: SystemStateSnapshot) -> float:
        """Calculate current cognitive load based on system metrics"""
        # Cognitive load correlates with system resource usage
        cpu_load = system_metrics.cpu_percent / 100.0
        memory_load = system_metrics.memory_percent / 100.0
        
        # Weighted average with CPU having more impact
        cognitive_load = (cpu_load * 0.6 + memory_load * 0.4)
        return min(1.0, cognitive_load)
    
    async def _assess_metacognitive_awareness(self) -> float:
        """Assess overall metacognitive awareness level"""
        if not self.assessment_history:
            return 0.5
        
        # Metacognitive awareness is based on consistency and accuracy of self-assessment
        recent_assessments = self.assessment_history[-20:]
        
        accuracy_scores = [a.self_assessment_accuracy for a in recent_assessments]
        confidence_scores = [a.confidence_score for a in recent_assessments]
        
        # High awareness = consistent and accurate self-assessment
        accuracy_consistency = 1.0 - np.var(accuracy_scores)
        confidence_consistency = 1.0 - np.var(confidence_scores)
        avg_accuracy = np.mean(accuracy_scores)
        
        awareness = (accuracy_consistency * 0.4 + 
                    confidence_consistency * 0.3 + 
                    avg_accuracy * 0.3)
        
        return min(1.0, max(0.0, awareness))

class UncertaintyTracker:
    """Track and quantify different types of uncertainty"""
    
    def __init__(self):
        self.epistemic_uncertainty_history = []
        self.aleatoric_uncertainty_history = []
        
    def track_epistemic_uncertainty(self, model_variance: float):
        """Track model/knowledge uncertainty"""
        self.epistemic_uncertainty_history.append({
            'timestamp': time.time(),
            'variance': model_variance
        })
        
        # Keep last 1000 measurements
        if len(self.epistemic_uncertainty_history) > 1000:
            self.epistemic_uncertainty_history.pop(0)
    
    def track_aleatoric_uncertainty(self, data_noise: float):
        """Track data/observation uncertainty"""
        self.aleatoric_uncertainty_history.append({
            'timestamp': time.time(),
            'noise': data_noise
        })
        
        if len(self.aleatoric_uncertainty_history) > 1000:
            self.aleatoric_uncertainty_history.pop(0)
    
    def get_uncertainty_summary(self) -> Dict[str, float]:
        """Get summary of uncertainty metrics"""
        epistemic_recent = [u['variance'] for u in self.epistemic_uncertainty_history[-10:]]
        aleatoric_recent = [u['noise'] for u in self.aleatoric_uncertainty_history[-10:]]
        
        return {
            'avg_epistemic_uncertainty': np.mean(epistemic_recent) if epistemic_recent else 0.0,
            'avg_aleatoric_uncertainty': np.mean(aleatoric_recent) if aleatoric_recent else 0.0,
            'total_uncertainty': np.mean(epistemic_recent + aleatoric_recent) if (epistemic_recent or aleatoric_recent) else 0.0
        }