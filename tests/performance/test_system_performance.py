"""
Performance tests for the Hephaestus RSI system.

Tests system performance under various loads, stress conditions,
and scaling scenarios to ensure production readiness.
"""

import pytest
import asyncio
import time
import threading
import multiprocessing
import gc
import psutil
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import AsyncMock, MagicMock

from src.main import RSIOrchestrator
from src.core.state import RSIStateManager, RSIState
from src.learning.online_learning import RSIOnlineLearner
from src.monitoring.telemetry import TelemetryCollector


class TestSystemStartupPerformance:
    """Test system startup and initialization performance."""
    
    @pytest.mark.performance
    async def test_cold_start_performance(self, performance_tracker):
        """Test cold start performance."""
        performance_tracker.start()
        
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        performance_tracker.stop()
        
        try:
            # Cold start should complete within reasonable time
            assert performance_tracker.duration < 30  # 30 seconds max for cold start
            
            # Memory usage should be reasonable
            if performance_tracker.memory_delta:
                assert performance_tracker.memory_delta < 500 * 1024 * 1024  # 500MB max
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.performance
    async def test_warm_start_performance(self, performance_tracker):
        """Test warm start performance (after components are initialized)."""
        # Pre-initialize to warm up
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        await orchestrator.stop()
        
        # Measure warm start
        performance_tracker.start()
        
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        performance_tracker.stop()
        
        try:
            # Warm start should be faster than cold start
            assert performance_tracker.duration < 15  # 15 seconds max for warm start
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.performance
    async def test_component_initialization_timing(self):
        """Test individual component initialization timing."""
        timing_results = {}
        
        # Test state manager initialization
        start_time = time.time()
        state_manager = RSIStateManager()
        timing_results["state_manager"] = time.time() - start_time
        
        # Test online learner initialization
        start_time = time.time()
        online_learner = RSIOnlineLearner()
        timing_results["online_learner"] = time.time() - start_time
        
        # Test telemetry initialization
        start_time = time.time()
        telemetry = TelemetryCollector()
        timing_results["telemetry"] = time.time() - start_time
        
        # All components should initialize quickly
        for component, duration in timing_results.items():
            assert duration < 5, f"{component} took {duration}s to initialize"


class TestLoadPerformance:
    """Test system performance under various loads."""
    
    @pytest.mark.performance
    async def test_concurrent_api_requests(self, async_test_client, performance_tracker):
        """Test performance under concurrent API requests."""
        async def make_health_request():
            response = await async_test_client.get("/health")
            return response.status_code, response.elapsed.total_seconds()
        
        # Test with increasing concurrency
        concurrency_levels = [1, 5, 10, 20, 50]
        results = {}
        
        for concurrency in concurrency_levels:
            performance_tracker.start()
            
            # Make concurrent requests
            tasks = [make_health_request() for _ in range(concurrency)]
            responses = await asyncio.gather(*tasks)
            
            performance_tracker.stop()
            
            # Analyze results
            response_times = [elapsed for status, elapsed in responses]
            success_rate = sum(1 for status, _ in responses if status == 200) / len(responses)
            
            results[concurrency] = {
                "total_time": performance_tracker.duration,
                "avg_response_time": np.mean(response_times),
                "max_response_time": np.max(response_times),
                "success_rate": success_rate,
                "throughput": concurrency / performance_tracker.duration
            }
            
            # Basic performance requirements
            assert success_rate >= 0.95  # 95% success rate
            assert np.mean(response_times) < 1.0  # Average response time < 1s
        
        # Throughput should scale reasonably with concurrency
        assert results[10]["throughput"] > results[1]["throughput"]
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_sustained_load_performance(self, performance_tracker):
        """Test performance under sustained load."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            performance_tracker.start()
            
            # Simulate sustained operations for 60 seconds
            end_time = time.time() + 60
            operation_count = 0
            
            while time.time() < end_time:
                # Perform various operations
                await orchestrator.state_manager.transition_state(
                    system_metrics={"operation_count": operation_count}
                )
                
                if hasattr(orchestrator, 'telemetry') and orchestrator.telemetry:
                    await orchestrator.telemetry.collect_metrics()
                
                operation_count += 1
                await asyncio.sleep(0.1)  # 10 operations per second
            
            performance_tracker.stop()
            
            # Analyze sustained performance
            ops_per_second = operation_count / performance_tracker.duration
            
            # Should maintain reasonable throughput
            assert ops_per_second >= 8  # At least 8 ops/second
            
            # Memory usage should be stable (no significant leaks)
            if performance_tracker.memory_delta:
                # Memory growth should be limited during sustained operation
                assert performance_tracker.memory_delta < 100 * 1024 * 1024  # 100MB max growth
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.performance
    async def test_batch_processing_performance(self, sample_training_data, performance_tracker):
        """Test batch processing performance."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            if hasattr(orchestrator, 'online_learner') and orchestrator.online_learner:
                X_train = sample_training_data["X_train"]
                y_train = sample_training_data["y_train"]
                
                # Test small batches
                performance_tracker.start()
                
                batch_size = 10
                for i in range(0, min(100, len(X_train)), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    for j in range(len(batch_X)):
                        features = {f"feature_{k}": batch_X[j][k] for k in range(len(batch_X[j]))}
                        await orchestrator.online_learner.learn_sample(features, batch_y[j])
                
                performance_tracker.stop()
                
                # Batch processing should be efficient
                samples_per_second = 100 / performance_tracker.duration
                assert samples_per_second >= 50  # At least 50 samples/second
                
        finally:
            await orchestrator.stop()


class TestMemoryPerformance:
    """Test memory usage and garbage collection performance."""
    
    @pytest.mark.performance
    async def test_memory_usage_patterns(self, performance_tracker):
        """Test memory usage patterns during normal operation."""
        orchestrator = RSIOrchestrator(environment="test")
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss
        
        await orchestrator.start()
        
        try:
            # Measure memory after startup
            startup_memory = psutil.Process().memory_info().rss
            startup_overhead = startup_memory - baseline_memory
            
            # Perform operations and measure memory growth
            initial_memory = psutil.Process().memory_info().rss
            
            for i in range(1000):
                await orchestrator.state_manager.transition_state(
                    system_metrics={f"test_{i}": float(i)},
                    metadata={"iteration": i}
                )
                
                # Periodic garbage collection
                if i % 100 == 0:
                    gc.collect()
            
            final_memory = psutil.Process().memory_info().rss
            operation_memory_growth = final_memory - initial_memory
            
            # Memory requirements
            assert startup_overhead < 200 * 1024 * 1024  # 200MB startup overhead
            assert operation_memory_growth < 50 * 1024 * 1024  # 50MB growth for 1000 operations
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.performance
    async def test_memory_leak_detection(self, performance_tracker):
        """Test for memory leaks during repeated operations."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            memory_samples = []
            
            # Perform operations in cycles and measure memory
            for cycle in range(5):
                # Perform a batch of operations
                for i in range(100):
                    await orchestrator.state_manager.transition_state(
                        system_metrics={"cycle": cycle, "iteration": i}
                    )
                
                # Force garbage collection
                gc.collect()
                await asyncio.sleep(0.1)
                
                # Sample memory usage
                memory_usage = psutil.Process().memory_info().rss
                memory_samples.append(memory_usage)
            
            # Analyze memory trend
            memory_growth = memory_samples[-1] - memory_samples[0]
            avg_growth_per_cycle = memory_growth / len(memory_samples)
            
            # Memory growth should be minimal
            assert avg_growth_per_cycle < 10 * 1024 * 1024  # 10MB per cycle max
            
            # Memory usage should stabilize (last 3 samples should be similar)
            recent_samples = memory_samples[-3:]
            memory_variance = np.var(recent_samples)
            assert memory_variance < (5 * 1024 * 1024) ** 2  # Low variance in recent samples
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.performance
    async def test_large_state_performance(self, performance_tracker):
        """Test performance with large state objects."""
        state_manager = RSIStateManager()
        
        # Create progressively larger states
        sizes = [100, 1000, 10000, 50000]
        timing_results = {}
        
        for size in sizes:
            # Create large state
            large_metrics = {f"metric_{i}": float(i) for i in range(size)}
            
            performance_tracker.start()
            
            # Test state transition with large data
            await state_manager.transition_state(system_metrics=large_metrics)
            
            performance_tracker.stop()
            
            timing_results[size] = performance_tracker.duration
            
            # Operations should complete in reasonable time
            assert performance_tracker.duration < 1.0  # 1 second max
        
        # Performance should scale reasonably with size
        assert timing_results[1000] < timing_results[10000] * 2  # Not exponential growth


class TestConcurrencyPerformance:
    """Test performance under concurrent access patterns."""
    
    @pytest.mark.performance
    async def test_concurrent_state_updates(self, performance_tracker):
        """Test concurrent state updates performance."""
        state_manager = RSIStateManager()
        
        async def update_state(worker_id: int, updates: int):
            for i in range(updates):
                await state_manager.transition_state(
                    system_metrics={f"worker_{worker_id}_update_{i}": float(i)}
                )
        
        performance_tracker.start()
        
        # Run concurrent state updates
        num_workers = 10
        updates_per_worker = 50
        
        tasks = [update_state(i, updates_per_worker) for i in range(num_workers)]
        await asyncio.gather(*tasks)
        
        performance_tracker.stop()
        
        # Verify all updates completed
        total_updates = num_workers * updates_per_worker
        assert len(state_manager.state_history) == total_updates + 1  # +1 for initial state
        
        # Performance should be reasonable
        updates_per_second = total_updates / performance_tracker.duration
        assert updates_per_second >= 100  # At least 100 updates/second
    
    @pytest.mark.performance
    async def test_concurrent_learning_performance(self, sample_training_data):
        """Test concurrent learning performance."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            if hasattr(orchestrator, 'online_learner') and orchestrator.online_learner:
                X_train = sample_training_data["X_train"]
                y_train = sample_training_data["y_train"]
                
                async def learning_worker(worker_id: int, data_slice: slice):
                    X_slice = X_train[data_slice]
                    y_slice = y_train[data_slice]
                    
                    for i, (x, y) in enumerate(zip(X_slice, y_slice)):
                        features = {f"feature_{j}": x[j] for j in range(len(x))}
                        await orchestrator.online_learner.learn_sample(features, y)
                
                start_time = time.time()
                
                # Divide data among workers
                num_workers = 4
                data_per_worker = len(X_train) // num_workers
                
                tasks = [
                    learning_worker(i, slice(i * data_per_worker, (i + 1) * data_per_worker))
                    for i in range(num_workers)
                ]
                
                await asyncio.gather(*tasks)
                
                duration = time.time() - start_time
                
                # Concurrent learning should be efficient
                samples_processed = len(X_train)
                samples_per_second = samples_processed / duration
                assert samples_per_second >= 20  # At least 20 samples/second
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.performance
    async def test_thread_safety_performance(self, performance_tracker):
        """Test thread safety doesn't significantly impact performance."""
        state_manager = RSIStateManager()
        
        def sync_update_state(worker_id: int, updates: int):
            """Synchronous version for thread testing."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def async_updates():
                for i in range(updates):
                    await state_manager.transition_state(
                        system_metrics={f"thread_{worker_id}_update_{i}": float(i)}
                    )
            
            loop.run_until_complete(async_updates())
            loop.close()
        
        performance_tracker.start()
        
        # Test with thread pool
        num_threads = 4
        updates_per_thread = 25
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(sync_update_state, i, updates_per_thread)
                for i in range(num_threads)
            ]
            
            # Wait for completion
            for future in futures:
                future.result()
        
        performance_tracker.stop()
        
        # Thread safety overhead should be minimal
        total_updates = num_threads * updates_per_thread
        updates_per_second = total_updates / performance_tracker.duration
        assert updates_per_second >= 50  # Reasonable performance with threads


class TestScalabilityPerformance:
    """Test system scalability characteristics."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_data_size_scalability(self):
        """Test performance scaling with data size."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            if hasattr(orchestrator, 'online_learner') and orchestrator.online_learner:
                # Test with increasing data sizes
                data_sizes = [10, 100, 1000, 5000]
                timing_results = {}
                
                for size in data_sizes:
                    # Generate test data
                    X = np.random.randn(size, 10)
                    y = np.random.randint(0, 2, size)
                    
                    start_time = time.time()
                    
                    # Process all data
                    for i in range(size):
                        features = {f"feature_{j}": X[i][j] for j in range(10)}
                        await orchestrator.online_learner.learn_sample(features, y[i])
                    
                    duration = time.time() - start_time
                    timing_results[size] = duration
                    
                    # Performance should scale reasonably
                    samples_per_second = size / duration
                    assert samples_per_second >= 10  # Minimum throughput
                
                # Scaling should be roughly linear, not exponential
                efficiency_1000 = 1000 / timing_results[1000]
                efficiency_100 = 100 / timing_results[100]
                
                # Efficiency shouldn't degrade too much with scale
                assert efficiency_1000 >= efficiency_100 * 0.5  # At least 50% efficiency
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.performance
    async def test_feature_dimensionality_scalability(self):
        """Test performance scaling with feature dimensionality."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            if hasattr(orchestrator, 'online_learner') and orchestrator.online_learner:
                # Test with increasing feature dimensions
                dimensions = [10, 50, 100, 500]
                timing_results = {}
                
                for dim in dimensions:
                    # Generate high-dimensional data
                    X = np.random.randn(100, dim)
                    y = np.random.randint(0, 2, 100)
                    
                    start_time = time.time()
                    
                    # Process samples
                    for i in range(100):
                        features = {f"feature_{j}": X[i][j] for j in range(dim)}
                        await orchestrator.online_learner.learn_sample(features, y[i])
                    
                    duration = time.time() - start_time
                    timing_results[dim] = duration
                
                # Higher dimensions should not cause exponential slowdown
                for dim in dimensions:
                    samples_per_second = 100 / timing_results[dim]
                    assert samples_per_second >= 5  # Minimum performance
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.performance
    async def test_state_history_scalability(self, performance_tracker):
        """Test performance with large state history."""
        state_manager = RSIStateManager()
        
        # Build up large state history
        for i in range(1000):
            await state_manager.transition_state(
                system_metrics={"iteration": i, "value": float(i)}
            )
        
        # Test operations on state manager with large history
        performance_tracker.start()
        
        # Perform operations that might be affected by history size
        for i in range(100):
            current_state = state_manager.current_state
            assert current_state is not None
            
            await state_manager.transition_state(
                system_metrics={"final_iteration": i}
            )
        
        performance_tracker.stop()
        
        # Operations should remain fast even with large history
        assert performance_tracker.duration < 2.0  # 2 seconds for 100 operations
        
        # History management should be efficient
        assert len(state_manager.state_history) <= state_manager.max_history_size


class TestResourceUtilization:
    """Test resource utilization efficiency."""
    
    @pytest.mark.performance
    async def test_cpu_utilization_efficiency(self, performance_tracker):
        """Test CPU utilization efficiency."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            # Monitor CPU usage during operations
            cpu_samples = []
            
            async def monitor_cpu():
                for _ in range(20):
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    cpu_samples.append(cpu_percent)
                    await asyncio.sleep(0.1)
            
            async def perform_operations():
                for i in range(200):
                    await orchestrator.state_manager.transition_state(
                        system_metrics={"cpu_test": i}
                    )
                    
                    if hasattr(orchestrator, 'telemetry') and orchestrator.telemetry:
                        await orchestrator.telemetry.collect_metrics()
            
            # Run monitoring and operations concurrently
            await asyncio.gather(monitor_cpu(), perform_operations())
            
            # Analyze CPU utilization
            avg_cpu = np.mean(cpu_samples)
            max_cpu = np.max(cpu_samples)
            
            # CPU usage should be reasonable
            assert avg_cpu < 80  # Average CPU under 80%
            assert max_cpu < 95  # Max CPU under 95%
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.performance
    async def test_io_efficiency(self, temp_dir, performance_tracker):
        """Test I/O operation efficiency."""
        state_manager = RSIStateManager()
        
        # Test state persistence performance
        state_files = []
        
        performance_tracker.start()
        
        for i in range(10):
            # Create state with varying amounts of data
            await state_manager.transition_state(
                system_metrics={f"metric_{j}": float(j) for j in range(i * 100)},
                metadata={"save_iteration": i}
            )
            
            # Save state
            state_file = temp_dir / f"state_{i}.pkl"
            await state_manager.save_state(str(state_file))
            state_files.append(state_file)
        
        performance_tracker.stop()
        
        # I/O operations should be efficient
        operations_per_second = 10 / performance_tracker.duration
        assert operations_per_second >= 5  # At least 5 save operations per second
        
        # Test load performance
        load_start = time.time()
        
        for state_file in state_files:
            new_manager = RSIStateManager()
            await new_manager.load_state(str(state_file))
        
        load_duration = time.time() - load_start
        load_ops_per_second = len(state_files) / load_duration
        assert load_ops_per_second >= 10  # At least 10 load operations per second


class TestBenchmarkComparisons:
    """Benchmark tests comparing different approaches."""
    
    @pytest.mark.performance
    async def test_synchronous_vs_asynchronous_performance(self, performance_tracker):
        """Compare synchronous vs asynchronous operation performance."""
        state_manager = RSIStateManager()
        
        # Test synchronous approach (simulated)
        performance_tracker.start()
        
        for i in range(100):
            await state_manager.transition_state(
                system_metrics={"sync_test": i}
            )
            # Simulate synchronous processing time
            await asyncio.sleep(0.001)
        
        performance_tracker.stop()
        sync_duration = performance_tracker.duration
        
        # Test asynchronous approach
        async def async_update(i):
            await state_manager.transition_state(
                system_metrics={"async_test": i}
            )
            await asyncio.sleep(0.001)
        
        performance_tracker.start()
        
        # Run updates concurrently
        tasks = [async_update(i) for i in range(100)]
        await asyncio.gather(*tasks)
        
        performance_tracker.stop()
        async_duration = performance_tracker.duration
        
        # Async should be faster for concurrent operations
        assert async_duration < sync_duration * 0.8  # At least 20% faster
    
    @pytest.mark.performance
    async def test_batch_vs_individual_processing(self, sample_training_data):
        """Compare batch vs individual processing performance."""
        orchestrator = RSIOrchestrator(environment="test")
        await orchestrator.start()
        
        try:
            if hasattr(orchestrator, 'online_learner') and orchestrator.online_learner:
                X_test = sample_training_data["X_test"][:100]  # Use subset for testing
                y_test = sample_training_data["y_test"][:100]
                
                # Test individual processing
                start_time = time.time()
                
                for i, (x, y) in enumerate(zip(X_test, y_test)):
                    features = {f"feature_{j}": x[j] for j in range(len(x))}
                    await orchestrator.online_learner.learn_sample(features, y)
                
                individual_duration = time.time() - start_time
                
                # For online learning, individual processing is typically the norm
                # But we can test the efficiency of the individual approach
                samples_per_second = len(X_test) / individual_duration
                assert samples_per_second >= 20  # Minimum throughput requirement
                
        finally:
            await orchestrator.stop()


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    @pytest.mark.performance
    async def test_baseline_performance_metrics(self, performance_tracker):
        """Establish baseline performance metrics."""
        orchestrator = RSIOrchestrator(environment="test")
        
        # Test startup performance
        performance_tracker.start()
        await orchestrator.start()
        performance_tracker.stop()
        startup_time = performance_tracker.duration
        
        try:
            # Test operation performance
            performance_tracker.start()
            
            for i in range(1000):
                await orchestrator.state_manager.transition_state(
                    system_metrics={"baseline_test": i}
                )
            
            performance_tracker.stop()
            operation_time = performance_tracker.duration
            
            # Record baseline metrics (these would be stored for regression testing)
            baseline_metrics = {
                "startup_time": startup_time,
                "operations_per_second": 1000 / operation_time,
                "memory_usage": psutil.Process().memory_info().rss,
            }
            
            # Basic performance requirements
            assert baseline_metrics["startup_time"] < 30
            assert baseline_metrics["operations_per_second"] >= 100
            assert baseline_metrics["memory_usage"] < 1024 * 1024 * 1024  # 1GB
            
        finally:
            await orchestrator.stop()