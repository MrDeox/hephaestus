"""
Comprehensive validation layer for RSI system.
Provides type-safe validation with high performance using Pydantic v2.
"""

import re
import ast
import inspect
from typing import Any, Dict, List, Optional, Union, Callable, Type, get_type_hints
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import hashlib
import json

from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic import Field
from cerberus import Validator
from cryptography.fernet import Fernet
import numpy as np


class ValidationType(str, Enum):
    """Types of validation performed."""
    SCHEMA = "schema"
    SAFETY = "safety"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS_LOGIC = "business_logic"


class ValidationSeverity(str, Enum):
    """Severity levels for validation failures."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationResult(BaseModel):
    """Result of a validation operation."""
    
    valid: bool
    validation_type: ValidationType
    severity: ValidationSeverity
    message: str
    field_errors: Dict[str, List[str]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = {"use_enum_values": True}


from typing import Annotated

# Pydantic v2 constrained types
SafeFloat = Annotated[float, Field(ge=-1e6, le=1e6)]
SafeInt = Annotated[int, Field(ge=-1000000, le=1000000)]
SafeStr = Annotated[str, Field(min_length=1, max_length=10000, pattern=r'^[a-zA-Z0-9_\-\.\s]+$')]


class ModelWeightValidation(BaseModel):
    """Validation schema for model weights."""
    
    weight_name: SafeStr
    shape: List[SafeInt]
    dtype: str = Field(pattern=r'^(float32|float64|int32|int64)$')
    min_value: SafeFloat
    max_value: SafeFloat
    checksum: str = Field(min_length=64, max_length=64)
    
    @field_validator('shape')
    def validate_shape(cls, v):
        """Validate tensor shape."""
        if not v or len(v) > 8:  # Reasonable limit on dimensions
            raise ValueError("Shape must be non-empty and have at most 8 dimensions")
        return v


class LearningConfigValidation(BaseModel):
    """Validation schema for learning configuration."""
    
    learning_rate: SafeFloat = Field(gt=0, le=1.0)
    batch_size: SafeInt = Field(ge=1, le=10000)
    max_epochs: SafeInt = Field(ge=1, le=10000)
    patience: SafeInt = Field(ge=1, le=1000)
    validation_split: float = Field(ge=0.0, le=0.5)
    
    # Safety constraints
    max_gradient_norm: SafeFloat = Field(default=1.0, ge=0.1, le=10.0)
    weight_decay: SafeFloat = Field(default=0.0, ge=0.0, le=0.1)
    
    @field_validator('validation_split')
    def validate_split(cls, v):
        """Validate validation split."""
        if not 0.0 <= v <= 0.5:
            raise ValueError("validation_split must be between 0.0 and 0.5")
        return v


class SafetyConstraints(BaseModel):
    """Safety constraints for RSI operations."""
    
    max_memory_mb: SafeInt = Field(default=1024, ge=100, le=16384)
    max_cpu_percent: SafeFloat = Field(default=80.0, ge=10.0, le=100.0)
    max_execution_time_seconds: SafeInt = Field(default=300, ge=1, le=3600)
    allowed_modules: List[SafeStr] = Field(default_factory=list)
    forbidden_functions: List[SafeStr] = Field(default_factory=list)
    
    @field_validator('allowed_modules')
    def validate_modules(cls, v):
        """Validate allowed modules."""
        safe_modules = {
            'math', 'random', 'datetime', 'json', 'collections',
            'itertools', 'functools', 'operator', 'numpy', 'pandas',
            'scikit-learn', 'torch', 'tensorflow'
        }
        for module in v:
            if module not in safe_modules:
                raise ValueError(f"Module '{module}' is not in the safe list")
        return v


class CodeValidation(BaseModel):
    """Validation for code that will be executed."""
    
    code: str = Field(min_length=1, max_length=50000)
    language: str = Field(default="python", pattern=r'^(python|sql)$')
    safety_level: str = Field(default="strict", pattern=r'^(strict|moderate|permissive)$')
    
    @field_validator('code')
    def validate_code_safety(cls, v):
        """Validate code for safety."""
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'import\s+subprocess',
            r'__import__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Code contains potentially dangerous pattern: {pattern}")
        
        # Parse AST to detect dangerous constructs
        try:
            tree = ast.parse(v)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if hasattr(node, 'module') and node.module:
                        if node.module in ['os', 'sys', 'subprocess', 'socket']:
                            raise ValueError(f"Import of dangerous module: {node.module}")
        except SyntaxError:
            raise ValueError("Code contains syntax errors")
        
        return v


class RSIValidator:
    """
    Comprehensive validator for RSI system operations.
    Combines Pydantic and Cerberus for multi-layer validation.
    """
    
    def __init__(self, safety_constraints: Optional[SafetyConstraints] = None):
        self.safety_constraints = safety_constraints or SafetyConstraints()
        self.cerberus_validator = Validator()
        self.validation_history: List[ValidationResult] = []
    
    def validate_model_weights(
        self, 
        weights: Dict[str, Any], 
        strict: bool = True
    ) -> ValidationResult:
        """Validate model weights."""
        try:
            errors = {}
            
            for name, weight_data in weights.items():
                try:
                    # Validate weight structure
                    weight_validation = ModelWeightValidation(**weight_data)
                    
                    # Additional safety checks
                    if 'values' in weight_data:
                        values = np.array(weight_data['values'])
                        
                        # Check for NaN/Inf values
                        if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                            errors[name] = ["Weight contains NaN or Inf values"]
                            continue
                        
                        # Verify checksum
                        computed_checksum = hashlib.sha256(values.tobytes()).hexdigest()
                        if computed_checksum != weight_validation.checksum:
                            errors[name] = ["Weight checksum mismatch"]
                            continue
                        
                        # Validate bounds
                        if np.any(values < weight_validation.min_value) or \
                           np.any(values > weight_validation.max_value):
                            errors[name] = ["Weight values outside allowed bounds"]
                    
                except ValidationError as e:
                    errors[name] = [str(error) for error in e.errors()]
            
            result = ValidationResult(
                valid=len(errors) == 0,
                validation_type=ValidationType.SAFETY,
                severity=ValidationSeverity.CRITICAL if errors else ValidationSeverity.INFO,
                message=f"Model weight validation {'passed' if not errors else 'failed'}",
                field_errors=errors,
                metadata={"weight_count": len(weights)}
            )
            
            self.validation_history.append(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                valid=False,
                validation_type=ValidationType.SAFETY,
                severity=ValidationSeverity.CRITICAL,
                message=f"Model weight validation error: {str(e)}",
                metadata={"exception": str(e)}
            )
            self.validation_history.append(result)
            return result
    
    def validate_learning_config(
        self, 
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate learning configuration."""
        try:
            learning_config = LearningConfigValidation(**config)
            
            # Additional business logic validation
            warnings = []
            
            if learning_config.learning_rate > 0.1:
                warnings.append("Learning rate is very high, may cause instability")
            
            if learning_config.batch_size < 32:
                warnings.append("Small batch size may lead to noisy gradients")
            
            severity = ValidationSeverity.WARNING if warnings else ValidationSeverity.INFO
            
            result = ValidationResult(
                valid=True,
                validation_type=ValidationType.BUSINESS_LOGIC,
                severity=severity,
                message="Learning configuration validation passed",
                metadata={
                    "warnings": warnings,
                    "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
                }
            )
            
            self.validation_history.append(result)
            return result
            
        except ValidationError as e:
            result = ValidationResult(
                valid=False,
                validation_type=ValidationType.SCHEMA,
                severity=ValidationSeverity.ERROR,
                message="Learning configuration validation failed",
                field_errors={"config": [str(error) for error in e.errors()]}
            )
            self.validation_history.append(result)
            return result
    
    def validate_code(
        self, 
        code: str, 
        language: str = "python",
        safety_level: str = "strict"
    ) -> ValidationResult:
        """Validate code for safe execution."""
        try:
            code_validation = CodeValidation(
                code=code,
                language=language,
                safety_level=safety_level
            )
            
            # Additional static analysis
            warnings = []
            
            # Check for complex operations that might be slow
            if len(code.splitlines()) > 100:
                warnings.append("Code is very long, may impact performance")
            
            # Check for loops that might be infinite
            if re.search(r'while\s+True', code):
                warnings.append("Potential infinite loop detected")
            
            result = ValidationResult(
                valid=True,
                validation_type=ValidationType.SECURITY,
                severity=ValidationSeverity.WARNING if warnings else ValidationSeverity.INFO,
                message="Code validation passed",
                metadata={
                    "warnings": warnings,
                    "code_lines": len(code.splitlines()),
                    "code_hash": hashlib.sha256(code.encode()).hexdigest()
                }
            )
            
            self.validation_history.append(result)
            return result
            
        except ValidationError as e:
            result = ValidationResult(
                valid=False,
                validation_type=ValidationType.SECURITY,
                severity=ValidationSeverity.CRITICAL,
                message="Code validation failed",
                field_errors={"code": [str(error) for error in e.errors()]}
            )
            self.validation_history.append(result)
            return result
    
    def validate_performance_metrics(
        self, 
        metrics: Dict[str, Any]
    ) -> ValidationResult:
        """Validate performance metrics."""
        schema = {
            'accuracy': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'loss': {'type': 'float', 'min': 0.0},
            'training_time': {'type': 'float', 'min': 0.0},
            'memory_usage': {'type': 'float', 'min': 0.0, 'max': self.safety_constraints.max_memory_mb},
            'cpu_usage': {'type': 'float', 'min': 0.0, 'max': 100.0}
        }
        
        if self.cerberus_validator.validate(metrics, schema):
            result = ValidationResult(
                valid=True,
                validation_type=ValidationType.PERFORMANCE,
                severity=ValidationSeverity.INFO,
                message="Performance metrics validation passed",
                metadata={"metrics_count": len(metrics)}
            )
        else:
            errors = []
            for field, error_list in self.cerberus_validator.errors.items():
                if isinstance(error_list, list):
                    errors.extend(error_list)
                else:
                    errors.append(str(error_list))
            
            result = ValidationResult(
                valid=False,
                validation_type=ValidationType.PERFORMANCE,
                severity=ValidationSeverity.ERROR,
                message="Performance metrics validation failed",
                field_errors={"metrics": errors}
            )
        
        self.validation_history.append(result)
        return result
    
    def validate_state_transition(
        self, 
        old_state: Dict[str, Any], 
        new_state: Dict[str, Any]
    ) -> ValidationResult:
        """Validate state transitions for safety."""
        warnings = []
        errors = []
        
        # Check for dramatic changes
        if 'model_weights' in old_state and 'model_weights' in new_state:
            # Simple check for weight change magnitude
            old_keys = set(old_state['model_weights'].keys())
            new_keys = set(new_state['model_weights'].keys())
            
            if old_keys != new_keys:
                warnings.append("Model architecture changed")
        
        # Check version progression
        if 'version' in old_state and 'version' in new_state:
            if new_state['version'] <= old_state['version']:
                errors.append("Version did not increase")
        
        # Check timestamp progression
        if 'last_modified' in old_state and 'last_modified' in new_state:
            try:
                old_time = datetime.fromisoformat(old_state['last_modified'])
                new_time = datetime.fromisoformat(new_state['last_modified'])
                if new_time <= old_time:
                    errors.append("Timestamp did not progress")
            except ValueError:
                warnings.append("Invalid timestamp format")
        
        result = ValidationResult(
            valid=len(errors) == 0,
            validation_type=ValidationType.SAFETY,
            severity=ValidationSeverity.ERROR if errors else (
                ValidationSeverity.WARNING if warnings else ValidationSeverity.INFO
            ),
            message=f"State transition validation {'passed' if not errors else 'failed'}",
            field_errors={"transition": errors},
            metadata={"warnings": warnings}
        )
        
        self.validation_history.append(result)
        return result
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed."""
        total = len(self.validation_history)
        if total == 0:
            return {"total_validations": 0}
        
        passed = sum(1 for v in self.validation_history if v.valid)
        failed = total - passed
        
        by_type = {}
        by_severity = {}
        
        for validation in self.validation_history:
            by_type[validation.validation_type] = by_type.get(validation.validation_type, 0) + 1
            by_severity[validation.severity] = by_severity.get(validation.severity, 0) + 1
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
            "by_type": by_type,
            "by_severity": by_severity,
            "recent_failures": [
                v for v in self.validation_history[-10:] if not v.valid
            ]
        }


# Factory functions for common validators
def create_strict_validator() -> RSIValidator:
    """Create validator with strict safety constraints."""
    constraints = SafetyConstraints(
        max_memory_mb=512,
        max_cpu_percent=50.0,
        max_execution_time_seconds=60,
        allowed_modules=['math', 'random', 'datetime', 'json', 'numpy'],
        forbidden_functions=['exec', 'eval', 'compile', 'open', 'input']
    )
    return RSIValidator(constraints)


def create_development_validator() -> RSIValidator:
    """Create validator with more permissive constraints for development."""
    constraints = SafetyConstraints(
        max_memory_mb=2048,
        max_cpu_percent=80.0,
        max_execution_time_seconds=300,
        allowed_modules=['math', 'random', 'datetime', 'json', 'numpy', 'pandas', 'scikit-learn'],
        forbidden_functions=['exec', 'eval', 'compile']
    )
    return RSIValidator(constraints)