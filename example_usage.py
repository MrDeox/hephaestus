#!/usr/bin/env python3
"""
Example usage of the Hephaestus RSI System.
Demonstrates key features and safety mechanisms.
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

from src.main import RSIOrchestrator
from src.core.state import RSIState
from src.learning.online_learning import create_classification_learner
from src.validation.validators import create_strict_validator
from src.security.sandbox import create_production_sandbox
from src.monitoring.anomaly_detection import create_behavioral_monitor


async def basic_usage_example():
    """Basic usage example showing core functionality."""
    print("=== Basic Usage Example ===")
    
    # Initialize RSI system
    orchestrator = RSIOrchestrator(environment="development")
    await orchestrator.start()
    
    try:
        # 1. Make predictions
        print("\n1. Making predictions...")
        features = {"temperature": 25.5, "humidity": 60.0, "pressure": 1013.25}
        prediction_result = await orchestrator.predict(features, user_id="demo_user")
        print(f"Prediction: {prediction_result}")
        
        # 2. Learn from new data
        print("\n2. Learning from new data...")
        learning_result = await orchestrator.learn(
            features=features,
            target=1.0,  # Positive class
            user_id="demo_user"
        )
        print(f"Learning result: {learning_result}")
        
        # 3. Execute code safely
        print("\n3. Executing code safely...")
        safe_code = """
import math
import random

# Generate some data
data = [random.uniform(0, 100) for _ in range(10)]
mean = sum(data) / len(data)
std_dev = math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))

print(f"Data: {data}")
print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
"""
        
        code_result = await orchestrator.execute_code(safe_code, user_id="demo_user")
        print(f"Code execution result: {code_result['status']}")
        print(f"Output: {code_result['output']}")
        
        # 4. Check system health
        print("\n4. System health check...")
        health = await orchestrator.get_system_health()
        print(f"System status: {health['overall_status']}")
        
        # 5. Get performance metrics
        print("\n5. Performance analysis...")
        performance = await orchestrator.analyze_performance()
        print(f"Learning accuracy: {performance['metrics']['learning']['accuracy']}")
        
    finally:
        await orchestrator.stop()


async def online_learning_example():
    """Demonstrate online learning with concept drift detection."""
    print("\n=== Online Learning Example ===")
    
    # Create classification learner
    learner = create_classification_learner()
    
    # Simulate data stream with concept drift
    print("\n1. Training initial model...")
    
    # Phase 1: Initial training data
    for i in range(100):
        x = {"feature1": np.random.normal(0, 1), "feature2": np.random.normal(0, 1)}
        y = 1 if x["feature1"] + x["feature2"] > 0 else 0
        
        metrics = await learner.learn_one(x, y)
        
        if i % 20 == 0:
            print(f"  Sample {i}: Accuracy = {metrics.accuracy:.3f}")
    
    print(f"\nInitial model accuracy: {learner.current_metrics.accuracy:.3f}")
    
    # Phase 2: Concept drift - relationship changes
    print("\n2. Simulating concept drift...")
    
    for i in range(100, 200):
        x = {"feature1": np.random.normal(0, 1), "feature2": np.random.normal(0, 1)}
        # Concept drift: now y depends on feature1 - feature2
        y = 1 if x["feature1"] - x["feature2"] > 0 else 0
        
        metrics = await learner.learn_one(x, y)
        
        if i % 20 == 0:
            print(f"  Sample {i}: Accuracy = {metrics.accuracy:.3f}, Drift = {metrics.concept_drift_detected}")
    
    print(f"\nFinal model accuracy: {learner.current_metrics.accuracy:.3f}")
    print(f"Concept drift events: {learner.current_metrics.drift_type}")


async def security_sandbox_example():
    """Demonstrate security sandbox with various code samples."""
    print("\n=== Security Sandbox Example ===")
    
    # Create production sandbox
    sandbox = create_production_sandbox()
    
    # Test cases
    test_cases = [
        {
            "name": "Safe mathematical computation",
            "code": """
import math
result = math.sqrt(16) + math.pi
print(f"Result: {result}")
""",
            "expected": "success"
        },
        {
            "name": "Dangerous import attempt",
            "code": """
import os
os.system('echo "This should not execute"')
""",
            "expected": "security_violation"
        },
        {
            "name": "Infinite loop (should timeout)",
            "code": """
while True:
    pass
""",
            "expected": "timeout"
        },
        {
            "name": "Memory exhaustion attempt",
            "code": """
big_list = [0] * (10**8)  # Try to allocate ~800MB
""",
            "expected": "resource_limit_exceeded"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        result = sandbox.execute(test_case["code"], timeout_seconds=5)
        print(f"  Status: {result.status}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Execution time: {result.execution_time_ms:.2f}ms")
        
        if result.security_violations:
            print(f"  Security violations: {result.security_violations}")
        
        if result.output:
            print(f"  Output: {result.output[:100]}...")


async def anomaly_detection_example():
    """Demonstrate behavioral anomaly detection."""
    print("\n=== Anomaly Detection Example ===")
    
    # Create behavioral monitor
    monitor = create_behavioral_monitor(algorithm="iforest")
    
    # Simulate normal behavior
    print("\n1. Collecting normal behavior data...")
    
    for i in range(150):
        # Normal behavior: consistent patterns
        behavioral_data = {
            "request_count": np.random.poisson(100),
            "response_time": np.random.gamma(2, 50),
            "error_rate": np.random.beta(1, 99),
            "cpu_usage": np.random.normal(30, 5),
            "memory_usage": np.random.normal(40, 8)
        }
        
        monitor.collect_behavioral_data(behavioral_data)
        
        if i % 50 == 0:
            print(f"  Collected {i} normal samples")
    
    # Wait for model training
    await asyncio.sleep(1)
    
    # Simulate anomalous behavior
    print("\n2. Introducing anomalous behavior...")
    
    anomaly_cases = [
        {
            "name": "High error rate",
            "data": {
                "request_count": 95,
                "response_time": 45,
                "error_rate": 0.5,  # Anomalously high
                "cpu_usage": 32,
                "memory_usage": 38
            }
        },
        {
            "name": "Extreme response time",
            "data": {
                "request_count": 98,
                "response_time": 5000,  # Anomalously high
                "error_rate": 0.01,
                "cpu_usage": 35,
                "memory_usage": 42
            }
        },
        {
            "name": "Resource spike",
            "data": {
                "request_count": 102,
                "response_time": 48,
                "error_rate": 0.015,
                "cpu_usage": 95,  # Anomalously high
                "memory_usage": 88   # Anomalously high
            }
        }
    ]
    
    for case in anomaly_cases:
        print(f"\n  Testing: {case['name']}")
        monitor.collect_behavioral_data(case["data"])
        
        # Check for alerts
        await asyncio.sleep(0.1)
        active_alerts = monitor.get_active_alerts()
        
        if active_alerts:
            alert = active_alerts[-1]
            print(f"    Alert triggered: {alert.description}")
            print(f"    Severity: {alert.severity}")
            print(f"    Confidence: {alert.confidence:.3f}")
        else:
            print(f"    No alert triggered")
    
    # Get monitoring statistics
    stats = monitor.get_monitoring_stats()
    print(f"\nMonitoring statistics:")
    print(f"  Total alerts: {stats['total_alerts']}")
    print(f"  Active alerts: {stats['active_alerts']}")
    print(f"  Models trained: {stats['models_trained']}")


async def comprehensive_safety_example():
    """Demonstrate comprehensive safety features."""
    print("\n=== Comprehensive Safety Example ===")
    
    # Create validator
    validator = create_strict_validator()
    
    # Test various validation scenarios
    validation_tests = [
        {
            "name": "Valid model weights",
            "test": lambda: validator.validate_model_weights({
                "layer1_weights": {
                    "weight_name": "layer1_weights",
                    "shape": [784, 128],
                    "dtype": "float32",
                    "min_value": -1.0,
                    "max_value": 1.0,
                    "checksum": "a" * 64
                }
            })
        },
        {
            "name": "Invalid model weights (bad checksum)",
            "test": lambda: validator.validate_model_weights({
                "layer1_weights": {
                    "weight_name": "layer1_weights",
                    "shape": [784, 128],
                    "dtype": "float32",
                    "min_value": -1.0,
                    "max_value": 1.0,
                    "checksum": "invalid"
                }
            })
        },
        {
            "name": "Valid learning config",
            "test": lambda: validator.validate_learning_config({
                "learning_rate": 0.01,
                "batch_size": 32,
                "max_epochs": 100,
                "patience": 10,
                "validation_split": 0.2
            })
        },
        {
            "name": "Invalid learning config (high learning rate)",
            "test": lambda: validator.validate_learning_config({
                "learning_rate": 10.0,  # Too high
                "batch_size": 32,
                "max_epochs": 100,
                "patience": 10,
                "validation_split": 0.2
            })
        },
        {
            "name": "Valid code",
            "test": lambda: validator.validate_code("""
import math
result = math.sqrt(16)
print(result)
""")
        },
        {
            "name": "Invalid code (dangerous import)",
            "test": lambda: validator.validate_code("""
import os
os.system('rm -rf /')
""")
        }
    ]
    
    for test in validation_tests:
        print(f"\n{test['name']}:")
        try:
            result = test["test"]()
            print(f"  Valid: {result.valid}")
            print(f"  Message: {result.message}")
            if result.field_errors:
                print(f"  Errors: {result.field_errors}")
        except Exception as e:
            print(f"  Exception: {e}")
    
    # Get validation summary
    summary = validator.get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"  Total validations: {summary['total_validations']}")
    print(f"  Success rate: {summary['success_rate']:.2%}")
    print(f"  By type: {summary['by_type']}")


async def performance_monitoring_example():
    """Demonstrate performance monitoring and self-improvement."""
    print("\n=== Performance Monitoring Example ===")
    
    orchestrator = RSIOrchestrator(environment="development")
    await orchestrator.start()
    
    try:
        # Simulate some learning to generate performance data
        print("\n1. Generating performance data...")
        
        for i in range(50):
            features = {
                "x": np.random.normal(0, 1),
                "y": np.random.normal(0, 1)
            }
            target = 1 if features["x"] + features["y"] > 0 else 0
            
            await orchestrator.learn(features, target)
            
            if i % 10 == 0:
                print(f"  Completed {i} learning iterations")
        
        # Analyze performance
        print("\n2. Analyzing performance...")
        performance = await orchestrator.analyze_performance()
        
        print(f"  Learning accuracy: {performance['metrics']['learning']['accuracy']:.3f}")
        print(f"  Samples processed: {performance['metrics']['learning']['samples_processed']}")
        print(f"  Needs improvement: {performance['needs_improvement']}")
        
        if performance['recommendations']:
            print(f"  Recommendations: {performance['recommendations']}")
        
        # Trigger self-improvement if needed
        if performance['needs_improvement']:
            print("\n3. Triggering self-improvement...")
            await orchestrator.trigger_self_improvement(performance)
        
        # Get system metrics
        print("\n4. System metrics...")
        health = await orchestrator.get_system_health()
        print(f"  Overall status: {health['overall_status']}")
        print(f"  Component health: {list(health['components'].keys())}")
        
    finally:
        await orchestrator.stop()


async def main():
    """Run all examples."""
    examples = [
        basic_usage_example,
        online_learning_example,
        security_sandbox_example,
        anomaly_detection_example,
        comprehensive_safety_example,
        performance_monitoring_example
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
        
        print("\n" + "="*60)
        await asyncio.sleep(1)


if __name__ == "__main__":
    print("Hephaestus RSI System - Example Usage")
    print("=" * 60)
    asyncio.run(main())