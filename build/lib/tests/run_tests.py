"""
Test runner for Hephaestus RSI test suite.

Provides convenient interface for running different types of tests
with appropriate configurations and reporting.
"""

import sys
import argparse
import subprocess
import time
import toml
from pathlib import Path
from typing import List, Optional


def get_pytest_addopts() -> List[str]:
    """Get addopts from pyproject.toml."""
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "r") as f:
                pyproject_data = toml.load(f)
                return pyproject_data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", [])
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}")
    return []

def run_pytest(
    test_path: Optional[str] = None,
    markers: Optional[List[str]] = None,
    coverage: bool = True,
    parallel: bool = False,
    verbose: bool = False,
    html_report: bool = False,
    additional_args: Optional[List[str]] = None
) -> int:
    """Run pytest with specified options."""
    
    cmd = ["python", "-m", "pytest"]
    
    # Add test path
    if test_path:
        cmd.append(test_path)
    
    # Add markers
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # Get addopts from pyproject.toml
    addopts = get_pytest_addopts()
    
    # Add coverage
    if coverage and not any(opt.startswith("--cov") for opt in addopts):
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=90"
        ])
        
        if html_report:
            cmd.append("--cov-report=html")
    
    # Add parallel execution
    if parallel and "-n" not in addopts:
        cmd.extend(["-n", "auto"])
    
    # Add verbosity
    if verbose:
        cmd.extend(["-v", "-s"])
    elif not any(opt.startswith("--tb") for opt in addopts):
        cmd.append("--tb=short")
    
    # Add additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Hephaestus RSI Test Runner")
    
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "security", "performance", "quick"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--path",
        help="Specific test path to run"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML coverage report"
    )
    
    parser.add_argument(
        "--markers",
        nargs="+",
        help="Additional pytest markers to use"
    )
    
    args = parser.parse_args()
    
    # Determine test configuration based on test type
    test_configs = {
        "all": {
            "test_path": "tests/",
            "markers": None,
            "parallel": True
        },
        "unit": {
            "test_path": "tests/unit/",
            "markers": ["unit"],
            "parallel": True
        },
        "integration": {
            "test_path": "tests/integration/",
            "markers": ["integration"],
            "parallel": False  # Integration tests may interfere with each other
        },
        "security": {
            "test_path": "tests/security/",
            "markers": ["security"],
            "parallel": False  # Security tests may need exclusive access
        },
        "performance": {
            "test_path": "tests/performance/",
            "markers": ["performance"],
            "parallel": False  # Performance tests need consistent environment
        },
        "quick": {
            "test_path": "tests/unit/",
            "markers": ["unit", "not slow"],
            "parallel": True
        }
    }
    
    config = test_configs[args.test_type]
    
    # Override with command line arguments
    test_path = args.path or config["test_path"]
    markers = args.markers or config["markers"]
    parallel = args.parallel or config["parallel"]
    coverage = not args.no_coverage
    
    print(f"🧪 Running {args.test_type} tests...")
    print(f"📁 Test path: {test_path}")
    if markers:
        print(f"🏷️  Markers: {', '.join(markers)}")
    print(f"⚡ Parallel: {parallel}")
    print(f"📊 Coverage: {coverage}")
    print("-" * 50)
    
    start_time = time.time()
    
    # Run tests
    exit_code = run_pytest(
        test_path=test_path,
        markers=markers,
        coverage=coverage,
        parallel=parallel,
        verbose=args.verbose,
        html_report=args.html_report
    )
    
    duration = time.time() - start_time
    
    print("-" * 50)
    print(f"⏱️  Test duration: {duration:.2f}s")
    
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        
    return exit_code


if __name__ == "__main__":
    sys.exit(main())