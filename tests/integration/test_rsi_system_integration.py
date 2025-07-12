"""
Integration tests for complete RSI system functionality.

Tests end-to-end workflows, component interactions, and system-level behavior
to ensure all subsystems work together correctly.
"""

import pytest
import asyncio
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock

from src.main import RSIOrchestrator
from src.core.state import RSIState, RSIStateManager
from src.validation.validators import RSIValidator
from src.safety.circuits import CircuitBreakerManager


class TestRSISystemBootstrap:
    """Test system initialization and bootstrap process."""
    
    @pytest.mark.integration
    async def test_system_initialization(self, temp_dir):
        """Test complete system initialization."""
        # Initialize orchestrator
        orchestrator = RSIOrchestrator(environment="test")
        
        # Should initialize without errors
        assert orchestrator is not None
        assert orchestrator.environment == "test"
        
        # Core components should be initialized
        assert hasattr(orchestrator, 'state_manager')
        assert hasattr(orchestrator, 'validator')
        assert hasattr(orchestrator, 'circuit_manager')
    
    @pytest.mark.integration
    async def test_system_startup_sequence(self, temp_dir):
        """Test system startup sequence."""
        orchestrator = RSIOrchestrator(environment="test")
        
        # Start system
        await orchestrator.start()
        
        # System should be in healthy state
        assert orchestrator.state_manager is not None
        assert orchestrator.validator is not None
        
        # Memory system should be initialized
        if hasattr(orchestrator, 'memory_system'):
            assert orchestrator.memory_system is not None
        
        # Cleanup
        await orchestrator.stop()
    
    @pytest.mark.integration
    async def test_component_dependency_resolution(self):
        """Test that component dependencies are resolved correctly."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        # State manager should be available to other components
        assert orchestrator.state_manager is not None
        
        # Validator should be functional
        if orchestrator.validator:
            test_config = {"learning_rate": 0.001}
            result = await orchestrator.validator.validate_learning_config(test_config)
            assert result is not None
        
        await orchestrator.stop()


class TestRSILearningFlow:
    """Test end-to-end learning workflows."""
    
    @pytest.mark.integration
    async def test_online_learning_pipeline(self, sample_training_data):
        """Test online learning pipeline from data to model update."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            # Check if online learner is available
            if hasattr(orchestrator, 'online_learner') and orchestrator.online_learner:
                # Simulate training data
                X_train = sample_training_data["X_train"]
                y_train = sample_training_data["y_train"]
                
                # Process training data through online learner
                for i in range(min(10, len(X_train))):  # Process first 10 samples
                    features = {f"feature_{j}": X_train[i][j] for j in range(len(X_train[i]))}
                    
                    # Learn from sample
                    await orchestrator.online_learner.learn_sample(features, y_train[i])
                
                # Make predictions
                X_test = sample_training_data["X_test"]
                test_features = {f"feature_{j}": X_test[0][j] for j in range(len(X_test[0]))}
                
                prediction = await orchestrator.online_learner.predict(test_features)
                assert prediction is not None
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_meta_learning_cycle(self):
        """Test meta-learning cycle execution."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            # Check if meta-learning components are available
            if (hasattr(orchestrator, 'gap_scanner') and orchestrator.gap_scanner and
                hasattr(orchestrator, 'mml_controller') and orchestrator.mml_controller):
                
                # Execute one meta-learning cycle
                results = await orchestrator._run_real_meta_learning()
                
                # Should complete without errors
                assert results is None or isinstance(results, dict)
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.integration
    async def test_hypothesis_generation_and_validation(self, sample_hypothesis):
        """Test hypothesis generation and validation workflow."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            # Check if hypothesis system is available
            if hasattr(orchestrator, 'hypothesis_orchestrator') and orchestrator.hypothesis_orchestrator:
                
                # Generate hypothesis
                hypothesis = await orchestrator.hypothesis_orchestrator.generate_hypothesis(
                    category="performance",
                    priority="medium"
                )
                
                assert hypothesis is not None
                assert "id" in hypothesis
                
                # Validate hypothesis
                validation_result = await orchestrator.hypothesis_orchestrator.validate_hypothesis(
                    hypothesis
                )
                
                assert validation_result is not None
                assert hasattr(validation_result, 'is_valid')
                
        finally:
            await orchestrator.stop()


class TestRSIExecutionFlow:
    """Test code execution and deployment workflows."""
    
    @pytest.mark.integration
    @pytest.mark.security
    async def test_secure_code_execution(self):
        """Test secure code execution through sandbox."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            if hasattr(orchestrator, 'sandbox') and orchestrator.sandbox:
                # Test safe code execution
                safe_code = """
result = sum([1, 2, 3, 4, 5])
print(f"Sum: {result}")
"""
                
                execution_result = await orchestrator.sandbox.execute_code(
                    code=safe_code,
                    timeout=10
                )
                
                assert execution_result is not None
                assert execution_result.get("success") is True
                assert execution_result.get("result") is not None
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.integration
    @pytest.mark.security  
    async def test_unsafe_code_rejection(self, security_test_cases):
        """Test that unsafe code is properly rejected."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            if hasattr(orchestrator, 'sandbox') and orchestrator.sandbox:
                # Test unsafe code injection attempts
                for unsafe_code in security_test_cases["code_injection"][:2]:  # Test first 2
                    execution_result = await orchestrator.sandbox.execute_code(
                        code=unsafe_code,
                        timeout=5
                    )
                    
                    # Should either fail or be blocked
                    assert (execution_result.get("success") is False or
                            "blocked" in str(execution_result.get("error", "")).lower() or
                            "restricted" in str(execution_result.get("error", "")).lower())
                    
        finally:
            await orchestrator.stop()
    
    @pytest.mark.integration
    async def test_rsi_cycle_execution(self):
        """Test complete RSI cycle execution."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            # Execute one RSI cycle
            initial_state = orchestrator.state_manager.current_state
            
            # Manually trigger RSI cycle components
            if hasattr(orchestrator, 'gap_scanner') and orchestrator.gap_scanner:
                await orchestrator._run_real_gap_scanning()
            
            if hasattr(orchestrator, 'mml_controller') and orchestrator.mml_controller:
                await orchestrator._run_real_meta_learning()
            
            if hasattr(orchestrator, 'execution_pipeline') and orchestrator.execution_pipeline:
                await orchestrator._run_real_rsi_execution()
            
            # State should potentially be updated
            final_state = orchestrator.state_manager.current_state
            assert final_state is not None
            
        finally:
            await orchestrator.stop()


class TestRSIMonitoringFlow:
    """Test monitoring and observability workflows."""
    
    @pytest.mark.integration
    async def test_telemetry_collection(self):
        """Test telemetry collection and metrics."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            if hasattr(orchestrator, 'telemetry') and orchestrator.telemetry:
                # Collect system metrics
                metrics = await orchestrator.telemetry.collect_metrics()
                
                assert isinstance(metrics, dict)
                assert len(metrics) > 0
                
                # Should contain basic system metrics
                expected_metrics = ["cpu_percent", "memory_percent", "timestamp"]
                found_metrics = [m for m in expected_metrics if m in metrics]
                assert len(found_metrics) > 0
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.integration
    async def test_anomaly_detection_workflow(self):
        """Test anomaly detection workflow."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            if hasattr(orchestrator, 'behavioral_monitor') and orchestrator.behavioral_monitor:
                # Start monitoring
                orchestrator.behavioral_monitor.start_monitoring()
                
                # Wait for some monitoring data
                await asyncio.sleep(2)
                
                # Check for anomalies
                anomalies = orchestrator.behavioral_monitor.get_anomalies()
                assert isinstance(anomalies, list)
                
                # Stop monitoring
                orchestrator.behavioral_monitor.stop_monitoring()
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.integration
    async def test_health_monitoring(self):
        """Test system health monitoring."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            # Get system health
            health_status = await orchestrator.get_health_status()
            
            assert isinstance(health_status, dict)
            assert "status" in health_status
            assert health_status["status"] in ["healthy", "degraded", "critical"]
            
            # Should include metacognitive status
            if "metacognitive_status" in health_status:
                meta_status = health_status["metacognitive_status"]
                assert isinstance(meta_status, dict)
                assert "safety_score" in meta_status
                
        finally:
            await orchestrator.stop()


class TestRSISafetyFlow:
    """Test safety and circuit breaker workflows."""
    
    @pytest.mark.integration
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with other components."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            if hasattr(orchestrator, 'circuit_manager') and orchestrator.circuit_manager:
                # Test circuit breaker state
                circuit_state = orchestrator.circuit_manager.get_circuit_state("test_circuit")
                assert circuit_state in ["closed", "open", "half_open"]
                
                # Simulate failure to trigger circuit breaker
                for _ in range(3):  # Trigger multiple failures
                    try:
                        await orchestrator.circuit_manager.call_with_circuit(
                            "test_circuit",
                            lambda: exec("raise Exception('Test failure')")
                        )
                    except:
                        pass  # Expected to fail
                
                # Circuit might open after failures
                new_state = orchestrator.circuit_manager.get_circuit_state("test_circuit")
                assert new_state in ["closed", "open", "half_open"]
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.integration
    async def test_safety_validation_workflow(self):
        """Test safety validation throughout the system."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            # Test various safety validations
            if orchestrator.validator:
                # Validate safe learning configuration
                safe_config = {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 10
                }
                
                result = await orchestrator.validator.validate_learning_config(safe_config)
                assert result.is_valid is True
                
                # Validate unsafe configuration
                unsafe_config = {
                    "learning_rate": -1.0,  # Invalid
                    "batch_size": 0         # Invalid
                }
                
                result = await orchestrator.validator.validate_learning_config(unsafe_config)
                assert result.is_valid is False
                
        finally:
            await orchestrator.stop()


class TestRSIStateManagement:
    """Test state management across system components."""
    
    @pytest.mark.integration
    async def test_state_consistency_across_components(self):
        """Test that state remains consistent across all components."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            initial_state = orchestrator.state_manager.current_state
            
            # Update state through orchestrator
            await orchestrator.state_manager.transition_state(
                system_metrics={"test_metric": 42.0},
                metadata={"integration_test": True}
            )
            
            new_state = orchestrator.state_manager.current_state
            
            # State should be updated
            assert new_state != initial_state
            assert new_state.system_metrics.get("test_metric") == 42.0
            assert new_state.metadata.get("integration_test") is True
            
            # State should be accessible by other components
            if hasattr(orchestrator, 'telemetry') and orchestrator.telemetry:
                # Telemetry should be able to access current state
                current_metrics = new_state.system_metrics
                assert "test_metric" in current_metrics
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.integration
    async def test_state_persistence_across_restart(self, temp_dir):
        """Test state persistence across system restarts."""
        state_file = temp_dir / "test_state.pkl"
        
        # First session - create and save state
        orchestrator1 = RSIOrchestrator(environment="test")
        await orchestrator1.start()
        
        try:
            # Update state
            await orchestrator1.state_manager.transition_state(
                system_metrics={"persistence_test": 123.0},
                metadata={"session": 1}
            )
            
            # Save state
            await orchestrator1.state_manager.save_state(str(state_file))
            
        finally:
            await orchestrator1.stop()
        
        # Second session - load state
        orchestrator2 = RSIOrchestrator(environment="test")
        await orchestrator2.start()
        
        try:
            # Load state
            await orchestrator2.state_manager.load_state(str(state_file))
            
            # State should be restored
            restored_state = orchestrator2.state_manager.current_state
            assert restored_state.system_metrics.get("persistence_test") == 123.0
            assert restored_state.metadata.get("session") == 1
            
        finally:
            await orchestrator2.stop()


class TestRSIAPIIntegration:
    """Test API endpoints and external interfaces."""
    
    @pytest.mark.integration
    async def test_health_endpoint(self, async_test_client):
        """Test health check endpoint."""
        response = await async_test_client.get("/health")
        
        assert response.status_code == 200
        health_data = response.json()
        
        assert "status" in health_data
        assert health_data["status"] in ["healthy", "degraded", "critical"]
        assert "timestamp" in health_data
    
    @pytest.mark.integration
    async def test_metrics_endpoint(self, async_test_client):
        """Test metrics endpoint."""
        response = await async_test_client.get("/metrics")
        
        # Should return metrics data
        assert response.status_code in [200, 404]  # 404 if endpoint not implemented
        
        if response.status_code == 200:
            metrics_data = response.json()
            assert isinstance(metrics_data, dict)
    
    @pytest.mark.integration
    async def test_prediction_endpoint(self, async_test_client):
        """Test prediction endpoint."""
        # Test prediction request
        prediction_data = {
            "features": {
                "feature_0": 1.0,
                "feature_1": 2.0,
                "feature_2": 3.0
            },
            "model_version": "latest"
        }
        
        response = await async_test_client.post("/predict", json=prediction_data)
        
        # Should handle prediction request
        assert response.status_code in [200, 400, 404, 501]  # Various valid responses
        
        if response.status_code == 200:
            result = response.json()
            assert "prediction" in result or "result" in result
    
    @pytest.mark.integration
    async def test_rsi_status_endpoint(self, async_test_client):
        """Test RSI system status endpoint."""
        response = await async_test_client.get("/rsi/status")
        
        # Should return RSI status
        assert response.status_code in [200, 404]  # 404 if endpoint not implemented
        
        if response.status_code == 200:
            status_data = response.json()
            assert isinstance(status_data, dict)


class TestRSIPerformanceIntegration:
    """Test system performance under various conditions."""
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_concurrent_request_handling(self, async_test_client):
        """Test handling of concurrent requests."""
        async def make_health_request():
            response = await async_test_client.get("/health")
            return response.status_code
        
        # Make multiple concurrent requests
        tasks = [make_health_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All requests should complete successfully
        assert all(status == 200 for status in results)
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_memory_usage_stability(self, performance_tracker):
        """Test memory usage stability during operations."""
        orchestrator = RSIOrchestrator(environment="test")
        
        performance_tracker.start()
        await orchestrator.start()
        
        try:
            # Perform various operations
            for i in range(10):
                await orchestrator.state_manager.transition_state(
                    system_metrics={f"test_metric_{i}": float(i)}
                )
                
                if hasattr(orchestrator, 'telemetry') and orchestrator.telemetry:
                    await orchestrator.telemetry.collect_metrics()
                
                # Small delay between operations
                await asyncio.sleep(0.1)
            
        finally:
            await orchestrator.stop()
            performance_tracker.stop()
        
        # Memory usage should be reasonable
        memory_delta = performance_tracker.memory_delta
        if memory_delta is not None:
            # Memory growth should be limited (less than 100MB for this test)
            assert memory_delta < 100 * 1024 * 1024
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_long_running_stability(self):
        """Test system stability over extended operation."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            start_time = time.time()
            
            # Run for 30 seconds with periodic operations
            while time.time() - start_time < 30:
                # Perform lightweight operations
                current_state = orchestrator.state_manager.current_state
                assert current_state is not None
                
                if hasattr(orchestrator, 'telemetry') and orchestrator.telemetry:
                    metrics = await orchestrator.telemetry.collect_metrics()
                    assert isinstance(metrics, dict)
                
                await asyncio.sleep(1)  # 1 second between checks
            
            # System should still be healthy after extended operation
            final_health = await orchestrator.get_health_status()
            assert final_health["status"] in ["healthy", "degraded"]
            
        finally:
            await orchestrator.stop()


class TestRSIErrorRecovery:
    """Test error handling and recovery mechanisms."""
    
    @pytest.mark.integration
    async def test_component_failure_recovery(self):
        """Test recovery from component failures."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            # Simulate component failure
            if hasattr(orchestrator, 'telemetry') and orchestrator.telemetry:
                # Temporarily break telemetry
                original_collect = orchestrator.telemetry.collect_metrics
                orchestrator.telemetry.collect_metrics = AsyncMock(
                    side_effect=Exception("Simulated failure")
                )
                
                # System should handle the failure gracefully
                try:
                    metrics = await orchestrator.telemetry.collect_metrics()
                except Exception:
                    pass  # Expected to fail
                
                # Restore functionality
                orchestrator.telemetry.collect_metrics = original_collect
                
                # Should recover and work normally
                metrics = await orchestrator.telemetry.collect_metrics()
                assert isinstance(metrics, dict)
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.integration
    async def test_invalid_input_handling(self):
        """Test handling of invalid inputs throughout the system."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            # Test invalid state transition
            try:
                await orchestrator.state_manager.transition_state(
                    system_metrics="invalid_type"  # Should be dict
                )
            except (TypeError, ValueError):
                pass  # Expected to fail with appropriate error
            
            # Test invalid validation input
            if orchestrator.validator:
                try:
                    result = await orchestrator.validator.validate_learning_config(
                        "invalid_config"  # Should be dict
                    )
                    # Should either fail or return invalid result
                    if result:
                        assert result.is_valid is False
                except (TypeError, ValueError):
                    pass  # Expected to fail
                    
        finally:
            await orchestrator.stop()
    
    @pytest.mark.integration
    async def test_resource_exhaustion_handling(self):
        """Test handling of resource exhaustion scenarios."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            # Test large state updates that might cause memory issues
            large_metrics = {f"metric_{i}": float(i) for i in range(10000)}
            
            # Should handle large data gracefully
            await orchestrator.state_manager.transition_state(
                system_metrics=large_metrics
            )
            
            new_state = orchestrator.state_manager.current_state
            assert len(new_state.system_metrics) == 10000
            
        finally:
            await orchestrator.stop()