"""
Pytest configuration and shared fixtures for Hephaestus RSI tests.

Provides common fixtures, test utilities, and configuration for comprehensive testing.
"""

import asyncio
import pytest
import tempfile
import shutil
import os
from pathlib import Path
from typing import AsyncGenerator, Generator, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pandas as pd

# Import test configuration
from tests import (
    TEST_DATA_DIR, TEST_FIXTURES_DIR, TEST_OUTPUTS_DIR,
    ENABLE_SLOW_TESTS, ENABLE_INTEGRATION_TESTS, 
    ENABLE_PERFORMANCE_TESTS, ENABLE_SECURITY_TESTS
)

# Core imports
from src.core.state import RSIState, RSIStateManager
from src.validation.validators import RSIValidator
from src.safety.circuits import CircuitBreakerManager


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests") 
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "redis: Tests requiring Redis")
    config.addinivalue_line("markers", "docker: Tests requiring Docker")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment settings."""
    skip_slow = pytest.mark.skip(reason="Slow tests disabled (set HEPHAESTUS_ENABLE_SLOW_TESTS=true)")
    skip_integration = pytest.mark.skip(reason="Integration tests disabled")
    skip_performance = pytest.mark.skip(reason="Performance tests disabled")
    skip_security = pytest.mark.skip(reason="Security tests disabled")
    
    for item in items:
        if "slow" in item.keywords and not ENABLE_SLOW_TESTS:
            item.add_marker(skip_slow)
        if "integration" in item.keywords and not ENABLE_INTEGRATION_TESTS:
            item.add_marker(skip_integration)
        if "performance" in item.keywords and not ENABLE_PERFORMANCE_TESTS:
            item.add_marker(skip_performance)
        if "security" in item.keywords and not ENABLE_SECURITY_TESTS:
            item.add_marker(skip_security)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture
def test_data_dir() -> Path:
    """Get test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def test_fixtures_dir() -> Path:
    """Get test fixtures directory."""
    return TEST_FIXTURES_DIR


@pytest.fixture
def test_outputs_dir() -> Path:
    """Get test outputs directory."""
    return TEST_OUTPUTS_DIR


# Core Component Fixtures

@pytest.fixture
def rsi_state() -> RSIState:
    """Create a test RSI state."""
    return RSIState(
        system_metrics={
            "cpu_usage": 0.3,
            "memory_usage": 0.5,
            "disk_usage": 0.2
        },
        learning_metrics={
            "accuracy": 0.85,
            "loss": 0.15,
            "learning_rate": 0.001
        },
        safety_metrics={
            "safety_score": 0.95,
            "circuit_breaker_state": "closed",
            "anomaly_score": 0.1
        },
        metadata={
            "version": "1.0.0",
            "environment": "test",
            "created_at": "2025-01-01T00:00:00Z"
        }
    )


@pytest.fixture
async def state_manager(rsi_state: RSIState) -> AsyncGenerator[RSIStateManager, None]:
    """Create a test state manager."""
    manager = RSIStateManager(initial_state=rsi_state)
    yield manager
    # Cleanup if needed


@pytest.fixture
def validator() -> RSIValidator:
    """Create a test validator."""
    return RSIValidator()


@pytest.fixture
async def circuit_manager() -> AsyncGenerator[CircuitBreakerManager, None]:
    """Create a test circuit breaker manager."""
    manager = CircuitBreakerManager()
    yield manager
    # Cleanup if needed


# Mock Fixtures

@pytest.fixture
def mock_telemetry():
    """Mock telemetry collector."""
    with patch('src.monitoring.telemetry.TelemetryCollector') as mock:
        mock_instance = MagicMock()
        mock_instance.collect_metrics = AsyncMock(return_value={
            "cpu_percent": 30.0,
            "memory_percent": 50.0,
            "inference_rate": 10.0
        })
        mock_instance.record_event = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_behavioral_monitor():
    """Mock behavioral monitor."""
    with patch('src.monitoring.anomaly_detection.BehavioralMonitor') as mock:
        mock_instance = MagicMock()
        mock_instance.start_monitoring = MagicMock()
        mock_instance.stop_monitoring = MagicMock()
        mock_instance.get_anomalies = MagicMock(return_value=[])
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_sandbox():
    """Mock execution sandbox."""
    with patch('src.security.sandbox.RSISandbox') as mock:
        mock_instance = MagicMock()
        mock_instance.execute_code = AsyncMock(return_value={
            "success": True,
            "result": "Test execution successful",
            "duration": 1.0,
            "resource_usage": {"cpu": 10, "memory": 50}
        })
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_model_versioning():
    """Mock model version manager."""
    with patch('src.core.model_versioning.ModelVersionManager') as mock:
        mock_instance = MagicMock()
        mock_instance.save_model = AsyncMock(return_value="test_model_v1.0.0")
        mock_instance.load_model = AsyncMock(return_value=MagicMock())
        mock_instance.list_models = AsyncMock(return_value=[])
        mock.return_value = mock_instance
        yield mock_instance


# Data Fixtures

@pytest.fixture
def sample_training_data() -> Dict[str, np.ndarray]:
    """Generate sample training data for ML tests."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    return {
        "X_train": X[:800],
        "X_test": X[800:],
        "y_train": y[:800], 
        "y_test": y[800:]
    }


@pytest.fixture
def sample_time_series_data() -> pd.DataFrame:
    """Generate sample time series data for testing."""
    dates = pd.date_range('2025-01-01', periods=1000, freq='H')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'cpu_usage': np.random.uniform(0, 100, 1000),
        'memory_usage': np.random.uniform(0, 100, 1000),
        'network_io': np.random.uniform(0, 1000, 1000),
        'disk_io': np.random.uniform(0, 500, 1000),
        'anomaly_score': np.random.uniform(0, 1, 1000)
    })
    
    return data


@pytest.fixture
def sample_hypothesis() -> Dict[str, Any]:
    """Create sample hypothesis for testing."""
    return {
        "id": "test_hypothesis_001",
        "title": "Optimize Memory Allocation",
        "description": "Implement smarter memory pooling to reduce allocation overhead",
        "category": "performance",
        "priority": "high",
        "expected_improvement": 0.15,
        "code_changes": {
            "file": "src/core/memory_manager.py",
            "function": "allocate_memory",
            "changes": "# Optimized memory allocation logic"
        },
        "validation_criteria": {
            "performance_improvement": ">= 10%",
            "memory_usage_reduction": ">= 5%",
            "no_regression": "accuracy >= 95%"
        },
        "safety_constraints": {
            "max_complexity": 0.7,
            "safety_level": "high",
            "timeout_seconds": 300
        }
    }


# Environment Setup Fixtures

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("HEPHAESTUS_ENVIRONMENT", "test")
    monkeypatch.setenv("HEPHAESTUS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("HEPHAESTUS_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("HEPHAESTUS_ENABLE_SAFETY_CHECKS", "true")


@pytest.fixture
def isolated_filesystem(temp_dir):
    """Provide isolated filesystem for tests that modify files."""
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        yield temp_dir
    finally:
        os.chdir(original_cwd)


# Performance Test Utilities

@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests."""
    import time
    import psutil
    
    class PerformanceTracker:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss
            
        def stop(self):
            self.end_time = time.time()
            self.end_memory = psutil.Process().memory_info().rss
            
        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
            
        @property
        def memory_delta(self):
            if self.start_memory and self.end_memory:
                return self.end_memory - self.start_memory
            return None
    
    return PerformanceTracker()


# Async Test Utilities

@pytest.fixture
async def async_test_client():
    """Create async test client for API testing."""
    from httpx import AsyncClient
    from src.main import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# Security Test Utilities

@pytest.fixture
def security_test_cases():
    """Provide security test cases for vulnerability testing."""
    return {
        "code_injection": [
            "__import__('os').system('rm -rf /')",
            "exec('import subprocess; subprocess.run([\"cat\", \"/etc/passwd\"])')",
            "eval('__import__(\"os\").system(\"whoami\")')",
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
        ],
        "resource_exhaustion": [
            "while True: pass",
            "[0] * (10**9)",
            "import threading; [threading.Thread(target=lambda: None).start() for _ in range(1000)]",
        ]
    }


# Cleanup Fixtures

@pytest.fixture(autouse=True, scope="session")
def cleanup_test_artifacts():
    """Cleanup test artifacts after test session."""
    yield
    
    # Cleanup test databases
    test_db_files = [
        "test_mlflow.db",
        "test_model_registry.db", 
        "test_episodic_memory.db"
    ]
    
    for db_file in test_db_files:
        if Path(db_file).exists():
            Path(db_file).unlink()
    
    # Cleanup test log files
    test_log_dirs = ["logs/test", "logs/testing"]
    for log_dir in test_log_dirs:
        log_path = Path(log_dir)
        if log_path.exists():
            shutil.rmtree(log_path)
    
    # Cleanup temporary model files
    temp_model_dirs = ["temp_models", "test_models"]
    for model_dir in temp_model_dirs:
        model_path = Path(model_dir)
        if model_path.exists():
            shutil.rmtree(model_path)