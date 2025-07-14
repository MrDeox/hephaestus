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
    ModelWeightValidation, LearningConfigValidation,
    CodeValidation, SafetyConstraints,
    create_strict_validator, create_development_validator
)


class TestValidationResult:
    """Test cases for ValidationResult data structure."""
    
    @pytest.mark.unit
    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            valid=True,
            confidence=0.95,
            errors=[],
            warnings=["Minor warning"],
            metadata={"validator": "test"}
        )
        
        assert result.valid is True
        assert result.confidence == 0.95
        assert result.errors == []
        assert result.warnings == ["Minor warning"]
        assert result.metadata == {"validator": "test"}
    
    @pytest.mark.unit
    def test_validation_result_invalid(self):
        """Test ValidationResult with errors."""
        result = ValidationResult(
            valid=False,
            confidence=0.3,
            errors=["Critical error", "Another error"],
            warnings=[],
            metadata={}
        )
        
        assert result.valid is False
        assert result.confidence == 0.3
        assert len(result.errors) == 2
        assert result.warnings == []


@pytest.fixture
def validator():
    """Fixture for RSIValidator instance."""
    return RSIValidator()


@pytest.fixture
def safety_constraints():
    """Fixture for SafetyConstraints instance."""
    return SafetyConstraints()


class TestRSIValidator:
    """Test cases for main RSIValidator class."""
    
    @pytest.mark.unit
    def test_rsi_validator_initialization(self):
        """Test RSIValidator initialization."""
        validator = RSIValidator()
        
        assert hasattr(validator, 'safety_constraints')
        assert hasattr(validator, 'cerberus_validator')
        assert hasattr(validator, 'validation_history')
        assert isinstance(validator.validation_history, list)
    
    @pytest.mark.unit
    def test_rsi_validator_with_custom_constraints(self):
        """Test RSIValidator with custom safety constraints."""
        custom_constraints = SafetyConstraints()
        validator = RSIValidator(safety_constraints=custom_constraints)
        
        assert validator.safety_constraints == custom_constraints


class TestModelWeightsValidation:
    """Test cases for model weights validation function."""
    
    @pytest.mark.unit
    def test_valid_model_weights(self, validator):
        """Test validation of valid model weights."""
        weights = {"layer1": np.random.randn(10, 5).tolist()}
        
        result = validator.validate_model_weights(weights)
        
        assert result.valid is True
        assert result.confidence > 0.0
        assert len(result.errors) == 0
    
    @pytest.mark.unit
    def test_empty_weights(self, validator):
        """Test validation of empty weights."""
        weights = {}
        
        result = validator.validate_model_weights(weights)
        
        # Should handle empty weights gracefully
        assert isinstance(result, ValidationResult)


class TestLearningConfigValidation:
    """Test cases for learning configuration validation function."""
    
    @pytest.mark.unit
    def test_valid_learning_config(self, validator):
        """Test validation of valid learning configuration."""
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
        }
        
        result = validator.validate_learning_config(config)
        
        assert result.valid is True
        assert result.confidence > 0.0


class TestPerformanceMetricsValidation:
    """Test cases for performance metrics validation function."""
    
    @pytest.mark.unit
    def test_valid_performance_metrics(self, validator):
        """Test validation of valid performance metrics."""
        metrics = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.91,
            "f1_score": 0.92,
            "loss": 0.05,
        }
        
        result = validator.validate_performance_metrics(metrics)
        
        assert result.valid is True
        assert result.confidence > 0.0
        assert len(result.errors) == 0


class TestValidationIntegration:
    """Integration tests for validation system."""
    
    @pytest.mark.unit
    def test_validation_pipeline(self, validator):
        """Test complete validation pipeline."""
        # Test that validator methods work
        weights = {"layer1": [[1, 2], [3, 4]]}
        result = validator.validate_model_weights(weights)
        
        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'valid')
        assert hasattr(result, 'confidence')