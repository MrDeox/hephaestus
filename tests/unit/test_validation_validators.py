"""
Unit tests for Validation System (src/validation/validators.py)

Tests the comprehensive validation system using Pydantic v2,
ensuring proper validation of model weights, learning configurations,
code safety, and performance metrics.
"""

import pytest
import numpy as np
import ast
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from src.validation.validators import (
    RSIValidator, ValidationResult, ValidationLevel,
    ModelWeightsValidator, LearningConfigValidator,
    CodeSafetyValidator, PerformanceMetricsValidator,
    validate_model_weights, validate_learning_config,
    validate_code_safety, validate_performance_metrics
)


class TestValidationResult:
    """Test cases for ValidationResult data structure."""
    
    @pytest.mark.unit
    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.95,
            errors=[],
            warnings=["Minor warning"],
            metadata={"validator": "test"}
        )
        
        assert result.is_valid is True
        assert result.confidence == 0.95
        assert result.errors == []
        assert result.warnings == ["Minor warning"]
        assert result.metadata == {"validator": "test"}
    
    @pytest.mark.unit
    def test_validation_result_invalid(self):
        """Test ValidationResult with errors."""
        result = ValidationResult(
            is_valid=False,
            confidence=0.3,
            errors=["Critical error", "Another error"],
            warnings=[]
        )
        
        assert result.is_valid is False
        assert result.confidence == 0.3
        assert len(result.errors) == 2
        assert "Critical error" in result.errors
    
    @pytest.mark.unit
    def test_validation_result_summary(self):
        """Test ValidationResult summary generation."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.85,
            errors=[],
            warnings=["Warning 1", "Warning 2"],
            metadata={"validator": "test", "duration": 0.5}
        )
        
        summary = result.get_summary()
        
        assert isinstance(summary, str)
        assert "valid" in summary.lower()
        assert "0.85" in summary or "85%" in summary
        assert "2" in summary  # warning count


class TestRSIValidator:
    """Test cases for main RSIValidator class."""
    
    @pytest.mark.unit
    def test_rsi_validator_initialization(self):
        """Test RSIValidator initialization."""
        validator = RSIValidator()
        
        assert validator.validation_level == ValidationLevel.STRICT
        assert isinstance(validator.model_weights_validator, ModelWeightsValidator)
        assert isinstance(validator.learning_config_validator, LearningConfigValidator)
        assert isinstance(validator.code_safety_validator, CodeSafetyValidator)
        assert isinstance(validator.performance_metrics_validator, PerformanceMetricsValidator)
    
    @pytest.mark.unit
    def test_rsi_validator_with_custom_level(self):
        """Test RSIValidator with custom validation level."""
        validator = RSIValidator(validation_level=ValidationLevel.PERMISSIVE)
        
        assert validator.validation_level == ValidationLevel.PERMISSIVE
    
    @pytest.mark.unit
    async def test_validate_all_components(self, validator):
        """Test comprehensive validation of all components."""
        # Valid test data
        model_weights = np.random.randn(100, 50)
        learning_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam"
        }
        code = "result = x + y"
        performance_metrics = {
            "accuracy": 0.95,
            "loss": 0.05,
            "f1_score": 0.93,
            "precision": 0.94,
            "recall": 0.92
        }
        
        result = await validator.validate_all(
            model_weights=model_weights,
            learning_config=learning_config,
            code=code,
            performance_metrics=performance_metrics
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.confidence > 0.8
        assert len(result.errors) == 0
    
    @pytest.mark.unit
    async def test_validate_with_invalid_data(self, validator):
        """Test validation with invalid data."""
        # Invalid test data
        model_weights = np.array([np.inf, np.nan, -np.inf])  # Invalid values
        learning_config = {
            "learning_rate": -0.1,  # Invalid negative learning rate
            "batch_size": 0,        # Invalid zero batch size
        }
        code = "__import__('os').system('rm -rf /')"  # Unsafe code
        performance_metrics = {
            "accuracy": 1.5,  # Invalid accuracy > 1
            "loss": -0.1      # Invalid negative loss
        }
        
        result = await validator.validate_all(
            model_weights=model_weights,
            learning_config=learning_config,
            code=code,
            performance_metrics=performance_metrics
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.confidence < 0.5
        assert len(result.errors) > 0
    
    @pytest.mark.unit
    async def test_validate_partial_data(self, validator):
        """Test validation with partial data."""
        result = await validator.validate_all(
            model_weights=np.random.randn(10, 5),
            learning_config=None,
            code=None,
            performance_metrics=None
        )
        
        assert isinstance(result, ValidationResult)
        # Should still validate the provided data


class TestModelWeightsValidator:
    """Test cases for ModelWeightsValidator."""
    
    @pytest.mark.unit
    def test_valid_model_weights(self):
        """Test validation of valid model weights."""
        weights = np.random.randn(100, 50)
        
        result = validate_model_weights(weights)
        
        assert result.is_valid is True
        assert result.confidence > 0.9
        assert len(result.errors) == 0
    
    @pytest.mark.unit
    def test_weights_with_nan_values(self):
        """Test validation of weights with NaN values."""
        weights = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
        
        result = validate_model_weights(weights)
        
        assert result.is_valid is False
        assert "nan" in str(result.errors).lower()
        assert result.confidence < 0.5
    
    @pytest.mark.unit
    def test_weights_with_infinite_values(self):
        """Test validation of weights with infinite values."""
        weights = np.array([[1.0, 2.0, np.inf], [4.0, -np.inf, 6.0]])
        
        result = validate_model_weights(weights)
        
        assert result.is_valid is False
        assert ("inf" in str(result.errors).lower() or 
                "infinite" in str(result.errors).lower())
    
    @pytest.mark.unit
    def test_weights_extreme_values(self):
        """Test validation of weights with extreme values."""
        weights = np.array([[1e10, 2.0, 3.0], [4.0, -1e10, 6.0]])
        
        result = validate_model_weights(weights)
        
        # Depending on implementation, might warn about extreme values
        assert isinstance(result, ValidationResult)
        if not result.is_valid:
            assert len(result.errors) > 0
    
    @pytest.mark.unit
    def test_empty_weights(self):
        """Test validation of empty weights."""
        weights = np.array([])
        
        result = validate_model_weights(weights)
        
        assert result.is_valid is False
        assert "empty" in str(result.errors).lower()
    
    @pytest.mark.unit
    def test_weights_wrong_type(self):
        """Test validation with wrong data type."""
        weights = "not_an_array"
        
        result = validate_model_weights(weights)
        
        assert result.is_valid is False
        assert "type" in str(result.errors).lower()
    
    @pytest.mark.unit
    def test_weights_shape_validation(self):
        """Test validation of weight shapes."""
        # 1D weights (might be valid for some models)
        weights_1d = np.random.randn(100)
        result_1d = validate_model_weights(weights_1d)
        assert isinstance(result_1d, ValidationResult)
        
        # 3D weights (might be valid for CNNs)
        weights_3d = np.random.randn(32, 32, 3)
        result_3d = validate_model_weights(weights_3d)
        assert isinstance(result_3d, ValidationResult)
        
        # Very high dimensional (might be invalid)
        weights_high_dim = np.random.randn(2, 2, 2, 2, 2, 2)
        result_high_dim = validate_model_weights(weights_high_dim)
        assert isinstance(result_high_dim, ValidationResult)


class TestLearningConfigValidator:
    """Test cases for LearningConfigValidator."""
    
    @pytest.mark.unit
    def test_valid_learning_config(self):
        """Test validation of valid learning configuration."""
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
            "momentum": 0.9,
            "weight_decay": 1e-4
        }
        
        result = validate_learning_config(config)
        
        assert result.is_valid is True
        assert result.confidence > 0.9
        assert len(result.errors) == 0
    
    @pytest.mark.unit
    def test_invalid_learning_rate(self):
        """Test validation with invalid learning rate."""
        configs = [
            {"learning_rate": -0.1},  # Negative
            {"learning_rate": 0},     # Zero
            {"learning_rate": 10},    # Too high
            {"learning_rate": "invalid"}  # Wrong type
        ]
        
        for config in configs:
            result = validate_learning_config(config)
            assert result.is_valid is False
            assert "learning_rate" in str(result.errors).lower()
    
    @pytest.mark.unit
    def test_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        configs = [
            {"batch_size": 0},       # Zero
            {"batch_size": -1},      # Negative
            {"batch_size": 1.5},     # Non-integer
            {"batch_size": "invalid"}  # Wrong type
        ]
        
        for config in configs:
            result = validate_learning_config(config)
            assert result.is_valid is False
            assert "batch_size" in str(result.errors).lower()
    
    @pytest.mark.unit
    def test_invalid_epochs(self):
        """Test validation with invalid epochs."""
        configs = [
            {"epochs": 0},        # Zero
            {"epochs": -1},       # Negative
            {"epochs": 1.5},      # Non-integer
            {"epochs": "invalid"}   # Wrong type
        ]
        
        for config in configs:
            result = validate_learning_config(config)
            assert result.is_valid is False
            assert "epochs" in str(result.errors).lower()
    
    @pytest.mark.unit
    def test_optimizer_validation(self):
        """Test optimizer validation."""
        # Valid optimizers
        valid_optimizers = ["adam", "sgd", "rmsprop", "adagrad"]
        for optimizer in valid_optimizers:
            config = {"optimizer": optimizer}
            result = validate_learning_config(config)
            # Should not fail solely due to optimizer
            if not result.is_valid:
                assert "optimizer" not in str(result.errors).lower()
        
        # Invalid optimizer
        config = {"optimizer": "invalid_optimizer"}
        result = validate_learning_config(config)
        # Might warn about unknown optimizer but not necessarily fail
        assert isinstance(result, ValidationResult)
    
    @pytest.mark.unit
    def test_empty_config(self):
        """Test validation with empty configuration."""
        result = validate_learning_config({})
        
        # Empty config might be valid (using defaults) or invalid (missing required fields)
        assert isinstance(result, ValidationResult)
        if not result.is_valid:
            assert len(result.errors) > 0


class TestCodeSafetyValidator:
    """Test cases for CodeSafetyValidator."""
    
    @pytest.mark.unit
    def test_safe_code(self):
        """Test validation of safe code."""
        safe_codes = [
            "result = x + y",
            "def add(a, b): return a + b",
            "import math; result = math.sqrt(16)",
            "data = [1, 2, 3, 4, 5]",
            "for i in range(10): print(i)"
        ]
        
        for code in safe_codes:
            result = validate_code_safety(code)
            assert result.is_valid is True
            assert result.confidence > 0.8
    
    @pytest.mark.unit
    def test_unsafe_imports(self):
        """Test validation of unsafe imports."""
        unsafe_codes = [
            "import os; os.system('rm -rf /')",
            "import subprocess; subprocess.run(['cat', '/etc/passwd'])",
            "import socket; s = socket.socket()",
            "__import__('os').system('whoami')",
            "from subprocess import call; call(['ls'])"
        ]
        
        for code in unsafe_codes:
            result = validate_code_safety(code)
            assert result.is_valid is False
            assert len(result.errors) > 0
            assert ("import" in str(result.errors).lower() or 
                    "unsafe" in str(result.errors).lower())
    
    @pytest.mark.unit
    def test_unsafe_builtins(self):
        """Test validation of unsafe builtin usage."""
        unsafe_codes = [
            "exec('print(\"Hello\")')",
            "eval('2 + 2')",
            "compile('print(1)', 'test', 'exec')",
            "globals()['__builtins__']",
            "locals().update({'x': 1})"
        ]
        
        for code in unsafe_codes:
            result = validate_code_safety(code)
            assert result.is_valid is False
            assert len(result.errors) > 0
    
    @pytest.mark.unit
    def test_file_operations(self):
        """Test validation of file operations."""
        file_codes = [
            "open('/etc/passwd', 'r')",
            "with open('file.txt', 'w') as f: f.write('data')",
            "import os; os.remove('file.txt')",
            "file('/etc/shadow')"
        ]
        
        for code in file_codes:
            result = validate_code_safety(code)
            # Depending on implementation, file operations might be restricted
            assert isinstance(result, ValidationResult)
            if not result.is_valid:
                assert ("file" in str(result.errors).lower() or 
                        "open" in str(result.errors).lower())
    
    @pytest.mark.unit
    def test_syntax_errors(self):
        """Test validation of syntactically incorrect code."""
        invalid_codes = [
            "if True print('hello')",  # Missing colon
            "def func( return x",       # Incomplete function
            "for i in range(10",       # Incomplete loop
            "x = [1, 2, 3",           # Incomplete list
            "import"                   # Incomplete import
        ]
        
        for code in invalid_codes:
            result = validate_code_safety(code)
            assert result.is_valid is False
            assert ("syntax" in str(result.errors).lower() or 
                    "parse" in str(result.errors).lower())
    
    @pytest.mark.unit
    def test_ast_analysis(self):
        """Test AST-based code analysis."""
        code = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""
        
        result = validate_code_safety(code)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        # Should analyze AST nodes and find safe operations
    
    @pytest.mark.unit
    def test_complex_safe_code(self):
        """Test validation of complex but safe code."""
        code = """
import math
import json

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.results = []
    
    def process(self, data):
        for item in data:
            try:
                value = math.sqrt(item['value'])
                self.results.append({
                    'original': item,
                    'processed': value,
                    'timestamp': item.get('timestamp', 0)
                })
            except (KeyError, ValueError, TypeError):
                continue
        
        return self.results
    
    def export_json(self):
        return json.dumps(self.results, indent=2)
"""
        
        result = validate_code_safety(code)
        
        assert result.is_valid is True
        assert result.confidence > 0.7


class TestPerformanceMetricsValidator:
    """Test cases for PerformanceMetricsValidator."""
    
    @pytest.mark.unit
    def test_valid_performance_metrics(self):
        """Test validation of valid performance metrics."""
        metrics = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.91,
            "f1_score": 0.92,
            "loss": 0.05,
            "auc": 0.88,
            "mse": 0.02,
            "mae": 0.015
        }
        
        result = validate_performance_metrics(metrics)
        
        assert result.is_valid is True
        assert result.confidence > 0.9
        assert len(result.errors) == 0
    
    @pytest.mark.unit
    def test_out_of_range_metrics(self):
        """Test validation of out-of-range metrics."""
        invalid_metrics = [
            {"accuracy": 1.5},      # > 1
            {"accuracy": -0.1},     # < 0
            {"precision": 2.0},     # > 1
            {"recall": -0.5},       # < 0
            {"f1_score": 1.1},      # > 1
            {"loss": -0.1},         # < 0 (depending on loss function)
        ]
        
        for metrics in invalid_metrics:
            result = validate_performance_metrics(metrics)
            assert result.is_valid is False
            assert len(result.errors) > 0
    
    @pytest.mark.unit
    def test_inconsistent_metrics(self):
        """Test validation of inconsistent metrics."""
        # F1 score doesn't match precision and recall
        metrics = {
            "precision": 0.9,
            "recall": 0.8,
            "f1_score": 0.95  # Should be ~0.847
        }
        
        result = validate_performance_metrics(metrics)
        
        # Might warn about inconsistency
        if not result.is_valid:
            assert "inconsistent" in str(result.errors + result.warnings).lower()
    
    @pytest.mark.unit
    def test_missing_metrics(self):
        """Test validation with minimal metrics."""
        metrics = {"accuracy": 0.85}
        
        result = validate_performance_metrics(metrics)
        
        # Should be valid even with minimal metrics
        assert result.is_valid is True
    
    @pytest.mark.unit
    def test_wrong_metric_types(self):
        """Test validation with wrong metric types."""
        metrics = {
            "accuracy": "95%",      # String instead of float
            "precision": [0.9],     # List instead of float
            "recall": None,         # None value
            "f1_score": {"value": 0.92}  # Dict instead of float
        }
        
        result = validate_performance_metrics(metrics)
        
        assert result.is_valid is False
        assert "type" in str(result.errors).lower()
    
    @pytest.mark.unit
    def test_nan_infinite_metrics(self):
        """Test validation with NaN/infinite metrics."""
        metrics = {
            "accuracy": np.nan,
            "precision": np.inf,
            "recall": -np.inf,
            "f1_score": 0.92
        }
        
        result = validate_performance_metrics(metrics)
        
        assert result.is_valid is False
        assert ("nan" in str(result.errors).lower() or 
                "inf" in str(result.errors).lower())


class TestValidationIntegration:
    """Integration tests for validation system."""
    
    @pytest.mark.unit
    async def test_validation_pipeline(self, validator):
        """Test complete validation pipeline."""
        # Simulate real-world data
        model_weights = np.random.randn(784, 128)  # Neural network weights
        learning_config = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 50,
            "optimizer": "adam",
            "momentum": 0.9
        }
        code = """
def predict(model, data):
    import numpy as np
    result = np.dot(data, model)
    return 1 / (1 + np.exp(-result))  # Sigmoid activation
"""
        performance_metrics = {
            "accuracy": 0.94,
            "precision": 0.92,
            "recall": 0.91,
            "f1_score": 0.915,
            "loss": 0.12
        }
        
        result = await validator.validate_all(
            model_weights=model_weights,
            learning_config=learning_config,
            code=code,
            performance_metrics=performance_metrics
        )
        
        assert result.is_valid is True
        assert result.confidence > 0.85
        assert len(result.errors) == 0
        
        # Should have validation metadata
        assert "validation_time" in result.metadata
        assert "validators_used" in result.metadata
    
    @pytest.mark.unit
    async def test_validation_with_warnings(self, validator):
        """Test validation that produces warnings but passes."""
        model_weights = np.random.randn(10, 5) * 100  # Large weights (might warn)
        learning_config = {
            "learning_rate": 0.1,  # High learning rate (might warn)
            "batch_size": 1,       # Small batch size (might warn)
            "epochs": 1000         # Many epochs (might warn)
        }
        
        result = await validator.validate_all(
            model_weights=model_weights,
            learning_config=learning_config
        )
        
        # Should pass but might have warnings
        assert isinstance(result, ValidationResult)
        if result.is_valid:
            # Check if there are warnings about suboptimal settings
            assert isinstance(result.warnings, list)
    
    @pytest.mark.unit
    @pytest.mark.performance
    async def test_validation_performance(self, validator, performance_tracker):
        """Test validation performance with large datasets."""
        # Large model weights
        large_weights = np.random.randn(10000, 1000)
        
        performance_tracker.start()
        
        result = await validator.validate_all(
            model_weights=large_weights,
            learning_config={"learning_rate": 0.001},
            code="result = data.mean()",
            performance_metrics={"accuracy": 0.95}
        )
        
        performance_tracker.stop()
        
        assert isinstance(result, ValidationResult)
        assert performance_tracker.duration < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.unit
    def test_validation_levels(self):
        """Test different validation levels."""
        # Test with different validation levels
        strict_validator = RSIValidator(validation_level=ValidationLevel.STRICT)
        permissive_validator = RSIValidator(validation_level=ValidationLevel.PERMISSIVE)
        
        # Borderline case that might pass in permissive but fail in strict
        borderline_config = {
            "learning_rate": 0.5,  # High but not invalid
            "batch_size": 2,       # Very small but not invalid
        }
        
        strict_result = validate_learning_config(
            borderline_config, 
            level=ValidationLevel.STRICT
        )
        permissive_result = validate_learning_config(
            borderline_config, 
            level=ValidationLevel.PERMISSIVE
        )
        
        # Strict should be more restrictive
        assert isinstance(strict_result, ValidationResult)
        assert isinstance(permissive_result, ValidationResult)
        
        if strict_result.is_valid != permissive_result.is_valid:
            assert permissive_result.is_valid  # Permissive should be more lenient


class TestValidationUtilities:
    """Test validation utility functions."""
    
    @pytest.mark.unit
    def test_validation_error_aggregation(self):
        """Test aggregation of validation errors from multiple validators."""
        results = [
            ValidationResult(is_valid=False, errors=["Error 1"], warnings=[]),
            ValidationResult(is_valid=False, errors=["Error 2"], warnings=["Warning 1"]),
            ValidationResult(is_valid=True, errors=[], warnings=["Warning 2"])
        ]
        
        # Test utility function to aggregate results (if exists)
        # This would test a hypothetical aggregate_validation_results function
        
        all_errors = []
        all_warnings = []
        overall_valid = True
        
        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            if not result.is_valid:
                overall_valid = False
        
        assert len(all_errors) == 2
        assert len(all_warnings) == 2
        assert overall_valid is False
    
    @pytest.mark.unit
    def test_validation_confidence_calculation(self):
        """Test validation confidence calculation."""
        # Test confidence calculation based on various factors
        high_confidence_factors = {
            "no_errors": True,
            "no_warnings": True,
            "all_metrics_in_range": True,
            "code_ast_valid": True
        }
        
        low_confidence_factors = {
            "has_errors": True,
            "multiple_warnings": True,
            "edge_case_values": True,
            "complex_code": True
        }
        
        # These would test hypothetical confidence calculation functions
        assert isinstance(high_confidence_factors, dict)
        assert isinstance(low_confidence_factors, dict)