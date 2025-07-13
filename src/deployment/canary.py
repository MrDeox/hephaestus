"""
Canary deployment system for Hephaestus RSI.

Implements gradual rollout with automatic rollback based on success metrics
and comprehensive monitoring of deployment health.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from ..common.exceptions import DeploymentError, create_error_context
from ..common.feature_flags import FeatureFlagManager, FeatureConfig, FeatureStrategy
from ..monitoring.telemetry import get_telemetry_collector
from ..common.performance import get_performance_monitor

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Canary deployment status."""
    PENDING = "pending"
    ROLLING_OUT = "rolling_out"
    MONITORING = "monitoring"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class RollbackTrigger(Enum):
    """Reasons for triggering rollback."""
    ERROR_RATE_HIGH = "error_rate_high"
    LATENCY_HIGH = "latency_high"
    SUCCESS_RATE_LOW = "success_rate_low"
    MANUAL = "manual"
    TIMEOUT = "timeout"
    HEALTH_CHECK_FAILED = "health_check_failed"


@dataclass
class CanaryMetrics:
    """Metrics for canary deployment monitoring."""
    
    success_rate: float = 0.0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    health_check_success_rate: float = 0.0
    
    # Comparison with control group
    success_rate_delta: float = 0.0
    error_rate_delta: float = 0.0
    latency_delta_ms: float = 0.0


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    
    name: str
    description: str = ""
    
    # Rollout configuration
    initial_percentage: float = 5.0
    max_percentage: float = 100.0
    increment_percentage: float = 10.0
    increment_interval_minutes: int = 10
    
    # Success criteria
    success_threshold: float = 0.95
    error_threshold: float = 0.05
    latency_threshold_ms: float = 1000.0
    
    # Monitoring configuration
    monitoring_duration_minutes: int = 30
    health_check_interval_seconds: int = 30
    metrics_collection_interval_seconds: int = 60
    
    # Rollback configuration
    auto_rollback_enabled: bool = True
    rollback_on_first_failure: bool = False
    max_consecutive_failures: int = 3
    
    # Notifications
    notification_channels: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"


@dataclass
class CanaryDeployment:
    """Represents an active canary deployment."""
    
    deployment_id: str
    config: CanaryConfig
    status: DeploymentStatus = DeploymentStatus.PENDING
    
    # Current state
    current_percentage: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Metrics tracking
    metrics_history: List[CanaryMetrics] = field(default_factory=list)
    current_metrics: Optional[CanaryMetrics] = None
    
    # Rollback information
    rollback_reason: Optional[RollbackTrigger] = None
    rollback_details: str = ""
    
    # Health tracking
    consecutive_failures: int = 0
    last_health_check: Optional[datetime] = None
    
    def is_active(self) -> bool:
        """Check if deployment is currently active."""
        return self.status in [
            DeploymentStatus.ROLLING_OUT,
            DeploymentStatus.MONITORING
        ]
    
    def should_increment(self) -> bool:
        """Check if deployment should increment percentage."""
        if not self.start_time:
            return False
        
        time_since_start = datetime.now() - self.start_time
        interval = timedelta(minutes=self.config.increment_interval_minutes)
        
        return time_since_start >= interval


class CanaryDeploymentManager:
    """Manages canary deployments with automatic rollback."""
    
    def __init__(
        self,
        feature_manager: Optional[FeatureFlagManager] = None,
        telemetry_collector: Optional[Any] = None
    ):
        self.feature_manager = feature_manager or FeatureFlagManager()
        self.telemetry = telemetry_collector or get_telemetry_collector()
        self.performance_monitor = get_performance_monitor()
        
        # Active deployments
        self.deployments: Dict[str, CanaryDeployment] = {}
        
        # Health check functions
        self.health_checks: List[Callable[[], bool]] = []
        
        # Metrics collectors
        self.metrics_collectors: List[Callable[[], Dict[str, float]]] = []
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Notification callbacks
        self.notification_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    async def start(self) -> None:
        """Start the canary deployment manager."""
        await self.feature_manager.start()
        
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Canary deployment manager started")
    
    async def stop(self) -> None:
        """Stop the canary deployment manager."""
        self._shutdown_event.set()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        
        await self.feature_manager.stop()
        logger.info("Canary deployment manager stopped")
    
    async def deploy(self, config: CanaryConfig) -> str:
        """Start a new canary deployment."""
        deployment_id = f"canary_{int(time.time())}"
        
        # Create deployment
        deployment = CanaryDeployment(
            deployment_id=deployment_id,
            config=config
        )
        
        try:
            # Create feature flag for canary
            feature_config = FeatureConfig(
                name=f"canary_{config.name}",
                strategy=FeatureStrategy.CANARY,
                enabled=True,
                canary_percentage=config.initial_percentage,
                success_threshold=config.success_threshold,
                failure_threshold=config.error_threshold,
                description=f"Canary deployment: {config.description}"
            )
            
            await self.feature_manager.create_feature(feature_config)
            
            # Initialize deployment
            deployment.status = DeploymentStatus.ROLLING_OUT
            deployment.current_percentage = config.initial_percentage
            deployment.start_time = datetime.now()
            
            self.deployments[deployment_id] = deployment
            
            # Send notification
            await self._notify_deployment_event(
                "deployment_started",
                deployment_id,
                {"config": config.name, "initial_percentage": config.initial_percentage}
            )
            
            logger.info(f"Started canary deployment: {deployment_id}")
            return deployment_id
        
        except Exception as e:
            logger.error(f"Failed to start canary deployment: {e}")
            raise DeploymentError(
                f"Canary deployment failed: {e}",
                context=create_error_context("canary_deploy", config=config.name)
            )
    
    async def rollback(self, deployment_id: str, reason: RollbackTrigger, details: str = "") -> None:
        """Manually trigger rollback of a canary deployment."""
        if deployment_id not in self.deployments:
            raise DeploymentError(
                f"Deployment not found: {deployment_id}",
                context=create_error_context("canary_rollback")
            )
        
        deployment = self.deployments[deployment_id]
        
        if not deployment.is_active():
            raise DeploymentError(
                f"Deployment not active: {deployment_id}",
                context=create_error_context("canary_rollback")
            )
        
        await self._execute_rollback(deployment, reason, details)
    
    async def promote(self, deployment_id: str) -> None:
        """Promote canary to full deployment."""
        if deployment_id not in self.deployments:
            raise DeploymentError(
                f"Deployment not found: {deployment_id}",
                context=create_error_context("canary_promote")
            )
        
        deployment = self.deployments[deployment_id]
        
        try:
            # Update feature flag to 100%
            await self.feature_manager.update_feature(
                f"canary_{deployment.config.name}",
                {"canary_percentage": 100.0}
            )
            
            # Mark as successful
            deployment.status = DeploymentStatus.SUCCESS
            deployment.current_percentage = 100.0
            deployment.end_time = datetime.now()
            
            # Send notification
            await self._notify_deployment_event(
                "deployment_promoted",
                deployment_id,
                {"final_percentage": 100.0}
            )
            
            logger.info(f"Promoted canary deployment: {deployment_id}")
        
        except Exception as e:
            logger.error(f"Failed to promote canary: {e}")
            raise DeploymentError(
                f"Canary promotion failed: {e}",
                context=create_error_context("canary_promote")
            )
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of a canary deployment."""
        if deployment_id not in self.deployments:
            raise DeploymentError(
                f"Deployment not found: {deployment_id}",
                context=create_error_context("canary_status")
            )
        
        deployment = self.deployments[deployment_id]
        
        return {
            "deployment_id": deployment_id,
            "status": deployment.status.value,
            "current_percentage": deployment.current_percentage,
            "start_time": deployment.start_time.isoformat() if deployment.start_time else None,
            "end_time": deployment.end_time.isoformat() if deployment.end_time else None,
            "config": {
                "name": deployment.config.name,
                "description": deployment.config.description,
                "max_percentage": deployment.config.max_percentage
            },
            "current_metrics": deployment.current_metrics.__dict__ if deployment.current_metrics else None,
            "rollback_reason": deployment.rollback_reason.value if deployment.rollback_reason else None,
            "rollback_details": deployment.rollback_details,
            "consecutive_failures": deployment.consecutive_failures
        }
    
    async def list_deployments(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """List all canary deployments."""
        deployments = []
        
        for deployment_id, deployment in self.deployments.items():
            if active_only and not deployment.is_active():
                continue
            
            status = await self.get_deployment_status(deployment_id)
            deployments.append(status)
        
        return deployments
    
    def add_health_check(self, health_check: Callable[[], bool]) -> None:
        """Add a health check function."""
        self.health_checks.append(health_check)
    
    def add_metrics_collector(self, collector: Callable[[], Dict[str, float]]) -> None:
        """Add a metrics collector function."""
        self.metrics_collectors.append(collector)
    
    def add_notification_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add a notification callback."""
        self.notification_callbacks.append(callback)
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for active deployments."""
        while not self._shutdown_event.is_set():
            try:
                # Monitor active deployments
                for deployment_id, deployment in list(self.deployments.items()):
                    if deployment.is_active():
                        await self._monitor_deployment(deployment)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _monitor_deployment(self, deployment: CanaryDeployment) -> None:
        """Monitor a single canary deployment."""
        try:
            # Collect metrics
            metrics = await self._collect_metrics(deployment)
            deployment.current_metrics = metrics
            deployment.metrics_history.append(metrics)
            
            # Perform health checks
            health_status = await self._perform_health_checks()
            deployment.last_health_check = datetime.now()
            
            if not health_status:
                deployment.consecutive_failures += 1
                logger.warning(f"Health check failed for deployment {deployment.deployment_id}")
                
                # Check if we should rollback
                if deployment.config.auto_rollback_enabled:
                    if deployment.config.rollback_on_first_failure or \
                       deployment.consecutive_failures >= deployment.config.max_consecutive_failures:
                        await self._execute_rollback(
                            deployment,
                            RollbackTrigger.HEALTH_CHECK_FAILED,
                            f"Health check failed {deployment.consecutive_failures} times"
                        )
                        return
            else:
                deployment.consecutive_failures = 0
            
            # Check metrics for rollback conditions
            if deployment.config.auto_rollback_enabled:
                rollback_trigger = self._check_rollback_conditions(deployment, metrics)
                if rollback_trigger:
                    await self._execute_rollback(
                        deployment,
                        rollback_trigger[0],
                        rollback_trigger[1]
                    )
                    return
            
            # Check if we should increment percentage
            if deployment.should_increment() and \
               deployment.current_percentage < deployment.config.max_percentage:
                await self._increment_percentage(deployment)
            
            # Check if deployment is complete
            elif deployment.current_percentage >= deployment.config.max_percentage and \
                 deployment.status == DeploymentStatus.ROLLING_OUT:
                # Enter monitoring phase
                deployment.status = DeploymentStatus.MONITORING
                await self._notify_deployment_event(
                    "deployment_monitoring",
                    deployment.deployment_id,
                    {"percentage": deployment.current_percentage}
                )
        
        except Exception as e:
            logger.error(f"Failed to monitor deployment {deployment.deployment_id}: {e}")
    
    async def _collect_metrics(self, deployment: CanaryDeployment) -> CanaryMetrics:
        """Collect metrics for canary deployment."""
        metrics = CanaryMetrics()
        
        try:
            # Collect from registered collectors
            for collector in self.metrics_collectors:
                try:
                    collected = collector()
                    
                    # Update metrics with collected data
                    if 'success_rate' in collected:
                        metrics.success_rate = collected['success_rate']
                    if 'error_rate' in collected:
                        metrics.error_rate = collected['error_rate']
                    if 'avg_latency_ms' in collected:
                        metrics.avg_latency_ms = collected['avg_latency_ms']
                    if 'p95_latency_ms' in collected:
                        metrics.p95_latency_ms = collected['p95_latency_ms']
                    if 'throughput_per_second' in collected:
                        metrics.throughput_per_second = collected['throughput_per_second']
                
                except Exception as e:
                    logger.warning(f"Metrics collector failed: {e}")
            
            # Get performance monitor data
            perf_stats = self.performance_monitor.get_operation_stats()
            if perf_stats:
                # Calculate derived metrics
                if 'total_operations' in perf_stats and perf_stats['total_operations'] > 0:
                    success_count = perf_stats.get('success_count', 0)
                    metrics.success_rate = success_count / perf_stats['total_operations']
                    metrics.error_rate = 1.0 - metrics.success_rate
                
                metrics.avg_latency_ms = perf_stats.get('avg_duration_ms', 0)
        
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
        
        return metrics
    
    async def _perform_health_checks(self) -> bool:
        """Perform all registered health checks."""
        if not self.health_checks:
            return True  # No health checks configured
        
        for health_check in self.health_checks:
            try:
                if not health_check():
                    return False
            except Exception as e:
                logger.warning(f"Health check failed with exception: {e}")
                return False
        
        return True
    
    def _check_rollback_conditions(
        self,
        deployment: CanaryDeployment,
        metrics: CanaryMetrics
    ) -> Optional[tuple[RollbackTrigger, str]]:
        """Check if deployment should be rolled back based on metrics."""
        config = deployment.config
        
        # Check error rate
        if metrics.error_rate > config.error_threshold:
            return (
                RollbackTrigger.ERROR_RATE_HIGH,
                f"Error rate {metrics.error_rate:.2%} exceeds threshold {config.error_threshold:.2%}"
            )
        
        # Check success rate
        if metrics.success_rate < config.success_threshold:
            return (
                RollbackTrigger.SUCCESS_RATE_LOW,
                f"Success rate {metrics.success_rate:.2%} below threshold {config.success_threshold:.2%}"
            )
        
        # Check latency
        if metrics.avg_latency_ms > config.latency_threshold_ms:
            return (
                RollbackTrigger.LATENCY_HIGH,
                f"Latency {metrics.avg_latency_ms:.1f}ms exceeds threshold {config.latency_threshold_ms}ms"
            )
        
        # Check deployment timeout
        if deployment.start_time:
            max_duration = timedelta(minutes=config.monitoring_duration_minutes)
            if datetime.now() - deployment.start_time > max_duration:
                return (
                    RollbackTrigger.TIMEOUT,
                    f"Deployment exceeded maximum duration of {config.monitoring_duration_minutes} minutes"
                )
        
        return None
    
    async def _increment_percentage(self, deployment: CanaryDeployment) -> None:
        """Increment the canary percentage."""
        new_percentage = min(
            deployment.current_percentage + deployment.config.increment_percentage,
            deployment.config.max_percentage
        )
        
        # Update feature flag
        await self.feature_manager.update_feature(
            f"canary_{deployment.config.name}",
            {"canary_percentage": new_percentage}
        )
        
        deployment.current_percentage = new_percentage
        
        # Send notification
        await self._notify_deployment_event(
            "deployment_incremented",
            deployment.deployment_id,
            {
                "new_percentage": new_percentage,
                "target_percentage": deployment.config.max_percentage
            }
        )
        
        logger.info(f"Incremented canary {deployment.deployment_id} to {new_percentage}%")
    
    async def _execute_rollback(
        self,
        deployment: CanaryDeployment,
        reason: RollbackTrigger,
        details: str
    ) -> None:
        """Execute rollback of canary deployment."""
        try:
            deployment.status = DeploymentStatus.ROLLING_BACK
            deployment.rollback_reason = reason
            deployment.rollback_details = details
            
            # Disable feature flag
            await self.feature_manager.update_feature(
                f"canary_{deployment.config.name}",
                {"enabled": False, "canary_percentage": 0.0}
            )
            
            # Mark as rolled back
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.current_percentage = 0.0
            deployment.end_time = datetime.now()
            
            # Send notification
            await self._notify_deployment_event(
                "deployment_rolled_back",
                deployment.deployment_id,
                {
                    "reason": reason.value,
                    "details": details
                }
            )
            
            logger.warning(f"Rolled back canary {deployment.deployment_id}: {reason.value} - {details}")
        
        except Exception as e:
            logger.error(f"Failed to rollback deployment {deployment.deployment_id}: {e}")
            deployment.status = DeploymentStatus.FAILED
    
    async def _notify_deployment_event(
        self,
        event_type: str,
        deployment_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Send deployment event notification."""
        notification_data = {
            "event_type": event_type,
            "deployment_id": deployment_id,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        for callback in self.notification_callbacks:
            try:
                callback(event_type, notification_data)
            except Exception as e:
                logger.warning(f"Notification callback failed: {e}")


# Global canary deployment manager instance
_canary_manager: Optional[CanaryDeploymentManager] = None


def get_canary_manager() -> CanaryDeploymentManager:
    """Get global canary deployment manager instance."""
    global _canary_manager
    if _canary_manager is None:
        _canary_manager = CanaryDeploymentManager()
    return _canary_manager


def set_canary_manager(manager: CanaryDeploymentManager) -> None:
    """Set global canary deployment manager instance."""
    global _canary_manager
    _canary_manager = manager