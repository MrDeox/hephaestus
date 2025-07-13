"""
Hephaestus RSI Test Suite

Comprehensive test suite for the Hephaestus Recursive Self-Improvement system.
Includes unit, integration, performance, and security tests with 90%+ coverage.

Test Categories:
- Unit Tests: Individual component testing
- Integration Tests: Multi-component interaction testing  
- Performance Tests: Load, stress, and benchmark testing
- Security Tests: Safety, sandbox, and vulnerability testing
- Regression Tests: Preventing functionality loss during refactoring

Usage:
    pytest                          # Run all tests
    pytest -m unit                  # Run only unit tests
    pytest -m integration           # Run only integration tests
    pytest -m performance           # Run performance tests
    pytest -m security              # Run security tests
    pytest --cov=src --cov-report=html  # Generate coverage report
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path for testing
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Test configuration
ENABLE_SLOW_TESTS = os.getenv("HEPHAESTUS_ENABLE_SLOW_TESTS", "false").lower() == "true"
ENABLE_INTEGRATION_TESTS = os.getenv("HEPHAESTUS_ENABLE_INTEGRATION_TESTS", "true").lower() == "true"
ENABLE_PERFORMANCE_TESTS = os.getenv("HEPHAESTUS_ENABLE_PERFORMANCE_TESTS", "false").lower() == "true"
ENABLE_SECURITY_TESTS = os.getenv("HEPHAESTUS_ENABLE_SECURITY_TESTS", "true").lower() == "true"

# Test data directories
TEST_DATA_DIR = TEST_DIR / "data"
TEST_FIXTURES_DIR = TEST_DIR / "fixtures"
TEST_OUTPUTS_DIR = TEST_DIR / "outputs"

# Ensure test directories exist
for test_dir in [TEST_DATA_DIR, TEST_FIXTURES_DIR, TEST_OUTPUTS_DIR]:
    test_dir.mkdir(exist_ok=True)

__version__ = "1.0.0"
__all__ = [
    "TEST_DIR",
    "PROJECT_ROOT", 
    "SRC_DIR",
    "TEST_DATA_DIR",
    "TEST_FIXTURES_DIR",
    "TEST_OUTPUTS_DIR",
    "ENABLE_SLOW_TESTS",
    "ENABLE_INTEGRATION_TESTS", 
    "ENABLE_PERFORMANCE_TESTS",
    "ENABLE_SECURITY_TESTS",
]