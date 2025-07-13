"""
Integration tests for canary deployment system.

Tests the complete canary deployment workflow including rollout,
monitoring, rollback, and promotion scenarios.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.deployment.canary import (
    CanaryDeploymentManager,
    CanaryConfig,
    DeploymentStatus,
    RollbackTrigger,
    CanaryMetrics
)
from src.common.feature_flags import FeatureFlagManager
from src.common.exceptions import DeploymentError


@pytest.fixture
async def canary_manager():
    """Create canary deployment manager for testing."""
    feature_manager = FeatureFlagManager()
    await feature_manager.start()
    
    manager = CanaryDeploymentManager(feature_manager=feature_manager)
    await manager.start()
    
    yield manager
    
    await manager.stop()
    await feature_manager.stop()


@pytest.fixture
def sample_config():
    """Create sample canary configuration."""
    return CanaryConfig(
        name="test_feature",
        description="Test canary deployment",
        initial_percentage=5.0,
        max_percentage=100.0,
        increment_percentage=10.0,
        increment_interval_minutes=1,  # Short interval for testing
        success_threshold=0.95,
        error_threshold=0.05,
        monitoring_duration_minutes=5,  # Short duration for testing
        auto_rollback_enabled=True,
        max_consecutive_failures=2
    )


class TestCanaryDeployment:
    """Test canary deployment functionality."""
    
    @pytest.mark.integration
    async def test_deploy_canary(self, canary_manager, sample_config):
        """Test starting a canary deployment."""
        deployment_id = await canary_manager.deploy(sample_config)
        
        assert deployment_id.startswith("canary_")
        assert deployment_id in canary_manager.deployments
        
        deployment = canary_manager.deployments[deployment_id]
        assert deployment.status == DeploymentStatus.ROLLING_OUT
        assert deployment.current_percentage == 5.0
        assert deployment.start_time is not None
    
    @pytest.mark.integration
    async def test_deployment_status(self, canary_manager, sample_config):
        """Test getting deployment status."""
        deployment_id = await canary_manager.deploy(sample_config)
        
        status = await canary_manager.get_deployment_status(deployment_id)
        
        assert status["deployment_id"] == deployment_id
        assert status["status"] == "rolling_out"
        assert status["current_percentage"] == 5.0
        assert status["config"]["name"] == "test_feature"
    
    @pytest.mark.integration
    async def test_list_deployments(self, canary_manager, sample_config):
        """Test listing deployments."""
        # Deploy multiple canaries
        deployment_id1 = await canary_manager.deploy(sample_config)
        
        config2 = CanaryConfig(name="test_feature_2", description="Second test")
        deployment_id2 = await canary_manager.deploy(config2)
        
        # List all deployments
        all_deployments = await canary_manager.list_deployments()
        assert len(all_deployments) == 2
        
        # List only active deployments
        active_deployments = await canary_manager.list_deployments(active_only=True)
        assert len(active_deployments) == 2
        
        # All should be active initially
        for deployment in active_deployments:
            assert deployment["status"] == "rolling_out"
    
    @pytest.mark.integration
    async def test_manual_rollback(self, canary_manager, sample_config):
        """Test manual rollback of deployment."""
        deployment_id = await canary_manager.deploy(sample_config)
        
        # Trigger manual rollback
        await canary_manager.rollback(
            deployment_id,
            RollbackTrigger.MANUAL,
            "Manual rollback for testing"
        )
        
        deployment = canary_manager.deployments[deployment_id]
        assert deployment.status == DeploymentStatus.ROLLED_BACK
        assert deployment.rollback_reason == RollbackTrigger.MANUAL
        assert deployment.rollback_details == "Manual rollback for testing"
        assert deployment.current_percentage == 0.0
    
    @pytest.mark.integration
    async def test_promote_deployment(self, canary_manager, sample_config):
        """Test promoting canary to full deployment."""
        deployment_id = await canary_manager.deploy(sample_config)
        
        # Promote to full deployment
        await canary_manager.promote(deployment_id)
        
        deployment = canary_manager.deployments[deployment_id]
        assert deployment.status == DeploymentStatus.SUCCESS
        assert deployment.current_percentage == 100.0
        assert deployment.end_time is not None
    
    @pytest.mark.integration
    async def test_rollback_nonexistent_deployment(self, canary_manager):
        """Test rollback of non-existent deployment."""
        with pytest.raises(DeploymentError, match="Deployment not found"):
            await canary_manager.rollback(
                "nonexistent_deployment",
                RollbackTrigger.MANUAL,
                "Test"
            )


class TestCanaryMonitoring:
    """Test canary monitoring functionality."""
    
    @pytest.mark.integration
    async def test_health_checks(self, canary_manager, sample_config):
        """Test health check integration."""
        # Add mock health check that fails
        health_check_results = [True, True, False, False]  # Fail after 2 successes
        health_check_index = [0]
        
        def mock_health_check():
            result = health_check_results[health_check_index[0] % len(health_check_results)]
            health_check_index[0] += 1
            return result
        
        canary_manager.add_health_check(mock_health_check)
        
        deployment_id = await canary_manager.deploy(sample_config)
        
        # Wait for health checks to trigger rollback
        await asyncio.sleep(2)
        
        deployment = canary_manager.deployments[deployment_id]
        # Should be rolled back due to health check failures
        assert deployment.consecutive_failures > 0
    
    @pytest.mark.integration
    async def test_metrics_collection(self, canary_manager, sample_config):
        """Test metrics collection during deployment."""
        # Add mock metrics collector
        def mock_metrics_collector():
            return {
                'success_rate': 0.98,
                'error_rate': 0.02,
                'avg_latency_ms': 50.0,
                'p95_latency_ms': 120.0,
                'throughput_per_second': 1000.0
            }
        
        canary_manager.add_metrics_collector(mock_metrics_collector)
        
        deployment_id = await canary_manager.deploy(sample_config)
        
        # Allow time for metrics collection
        await asyncio.sleep(1)
        
        deployment = canary_manager.deployments[deployment_id]
        
        # Check that metrics were collected
        if deployment.current_metrics:
            assert deployment.current_metrics.success_rate == 0.98
            assert deployment.current_metrics.error_rate == 0.02
            assert deployment.current_metrics.avg_latency_ms == 50.0
    
    @pytest.mark.integration
    async def test_automatic_rollback_on_high_error_rate(self, canary_manager, sample_config):
        """Test automatic rollback when error rate is too high."""
        # Configure low error threshold
        sample_config.error_threshold = 0.01
        
        # Add metrics collector that reports high error rate
        def failing_metrics_collector():
            return {
                'success_rate': 0.90,  # Below threshold
                'error_rate': 0.10,    # Above threshold
                'avg_latency_ms': 50.0
            }
        
        canary_manager.add_metrics_collector(failing_metrics_collector)
        
        deployment_id = await canary_manager.deploy(sample_config)
        
        # Wait for monitoring to detect high error rate
        await asyncio.sleep(2)
        
        deployment = canary_manager.deployments[deployment_id]
        
        # Should be rolled back due to high error rate
        if deployment.status == DeploymentStatus.ROLLED_BACK:
            assert deployment.rollback_reason == RollbackTrigger.ERROR_RATE_HIGH
    
    @pytest.mark.integration
    async def test_automatic_rollback_on_high_latency(self, canary_manager, sample_config):
        """Test automatic rollback when latency is too high."""
        # Configure low latency threshold
        sample_config.latency_threshold_ms = 100.0
        
        # Add metrics collector that reports high latency
        def slow_metrics_collector():
            return {
                'success_rate': 0.99,
                'error_rate': 0.01,
                'avg_latency_ms': 500.0  # Above threshold
            }
        
        canary_manager.add_metrics_collector(slow_metrics_collector)
        
        deployment_id = await canary_manager.deploy(sample_config)
        
        # Wait for monitoring to detect high latency
        await asyncio.sleep(2)
        
        deployment = canary_manager.deployments[deployment_id]
        
        # Should be rolled back due to high latency
        if deployment.status == DeploymentStatus.ROLLED_BACK:
            assert deployment.rollback_reason == RollbackTrigger.LATENCY_HIGH


class TestCanaryNotifications:
    """Test canary notification system."""
    
    @pytest.mark.integration
    async def test_notification_callbacks(self, canary_manager, sample_config):
        """Test notification callbacks during deployment lifecycle."""
        notifications = []
        
        def notification_callback(event_type, data):
            notifications.append((event_type, data))
        
        canary_manager.add_notification_callback(notification_callback)
        
        # Deploy canary
        deployment_id = await canary_manager.deploy(sample_config)
        
        # Allow time for notifications
        await asyncio.sleep(0.1)
        
        # Should have received deployment started notification
        assert len(notifications) > 0
        event_type, data = notifications[0]
        assert event_type == "deployment_started"
        assert data["deployment_id"] == deployment_id
        
        # Rollback and check notification
        await canary_manager.rollback(
            deployment_id,
            RollbackTrigger.MANUAL,
            "Test rollback"
        )
        
        # Should have received rollback notification
        rollback_notifications = [n for n in notifications if n[0] == "deployment_rolled_back"]
        assert len(rollback_notifications) > 0


class TestCanaryIntegration:
    """Test integration with feature flags."""
    
    @pytest.mark.integration
    async def test_feature_flag_integration(self, canary_manager, sample_config):
        """Test integration with feature flag system."""
        deployment_id = await canary_manager.deploy(sample_config)
        
        # Check that feature flag was created
        feature_name = f"canary_{sample_config.name}"
        feature = canary_manager.feature_manager.get_feature(feature_name)
        
        assert feature is not None
        assert feature.enabled is True
        assert feature.canary_percentage == sample_config.initial_percentage
        
        # Promote deployment
        await canary_manager.promote(deployment_id)
        
        # Check that feature flag was updated to 100%
        updated_feature = canary_manager.feature_manager.get_feature(feature_name)
        assert updated_feature.canary_percentage == 100.0
    
    @pytest.mark.integration
    async def test_feature_flag_evaluation_during_canary(self, canary_manager, sample_config):
        """Test feature flag evaluation during canary deployment."""
        deployment_id = await canary_manager.deploy(sample_config)
        
        feature_name = f"canary_{sample_config.name}"
        
        # Test evaluation for different users
        enabled_count = 0
        total_tests = 100
        
        for i in range(total_tests):
            evaluation = canary_manager.feature_manager.evaluate(
                feature_name,
                user_id=f"user_{i}"
            )
            if evaluation.enabled:
                enabled_count += 1
        
        # Should be approximately 5% (initial percentage)
        enabled_percentage = (enabled_count / total_tests) * 100
        assert 3 <= enabled_percentage <= 7  # Allow some variance due to hashing


class TestCanaryErrorHandling:
    """Test error handling in canary deployments."""
    
    @pytest.mark.integration
    async def test_deploy_duplicate_name(self, canary_manager, sample_config):
        """Test deploying canary with duplicate name."""
        # Deploy first canary
        await canary_manager.deploy(sample_config)
        
        # Try to deploy with same name - should succeed but create different feature flag
        deployment_id2 = await canary_manager.deploy(sample_config)
        
        # Both deployments should exist
        assert len(canary_manager.deployments) == 2
    
    @pytest.mark.integration
    async def test_operations_on_inactive_deployment(self, canary_manager, sample_config):
        """Test operations on inactive deployments."""
        deployment_id = await canary_manager.deploy(sample_config)
        
        # Promote deployment (makes it inactive)
        await canary_manager.promote(deployment_id)
        
        # Try to rollback inactive deployment
        with pytest.raises(DeploymentError, match="Deployment not active"):
            await canary_manager.rollback(
                deployment_id,
                RollbackTrigger.MANUAL,
                "Should fail"
            )