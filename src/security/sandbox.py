"""
Secure execution sandbox for RSI system.
Provides safe code execution with multiple layers of protection.
"""

import ast
import sys
import time
import signal
import subprocess
import tempfile
import os
import resource
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import hashlib
import uuid

from RestrictedPython import compile_restricted, safe_globals, safe_builtins
from RestrictedPython.Guards import guarded_iter_unpack_sequence
from RestrictedPython.transformer import RestrictingNodeTransformer
import docker

from ..validation.validators import RSIValidator, CodeValidation, SafetyConstraints
from ..monitoring.telemetry import trace_operation, record_safety_event
from loguru import logger


class SandboxType(str, Enum):
    """Types of sandbox execution."""
    RESTRICTED_PYTHON = "restricted_python"
    DOCKER_CONTAINER = "docker_container"
    SUBPROCESS = "subprocess"
    CODEJAIL = "codejail"


class ExecutionStatus(str, Enum):
    """Execution status codes."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    SECURITY_VIOLATION = "security_violation"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"


@dataclass
class ExecutionResult:
    """Result of sandbox execution."""
    
    status: ExecutionStatus
    output: str
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    cpu_time_ms: float = 0.0
    security_violations: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.security_violations is None:
            self.security_violations = []
        if self.metadata is None:
            self.metadata = {}


class SafeExecutionEnvironment:
    """
    Safe execution environment with restricted Python.
    Provides basic sandboxing without external dependencies.
    """
    
    def __init__(self, safety_constraints: Optional[SafetyConstraints] = None):
        self.safety_constraints = safety_constraints or SafetyConstraints()
        self.allowed_modules = set(self.safety_constraints.allowed_modules)
        self.forbidden_functions = set(self.safety_constraints.forbidden_functions)
        
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create safe global namespace."""
        safe_globals_dict = safe_globals.copy()
        safe_globals_dict.update(safe_builtins)
        
        # Add safe built-ins
        safe_globals_dict.update({
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
                'enumerate': enumerate,
                'zip': zip,
                'range': range,
                'map': map,
                'filter': filter,
                'any': any,
                'all': all,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'print': lambda *args, **kwargs: None,  # Disable print
                '_iter_unpack_sequence': guarded_iter_unpack_sequence,
                '_getiter': iter,
                '_getattr': getattr,
            }
        })
        
        # Add allowed modules
        import math
        import random
        import datetime
        import json
        
        allowed_module_map = {
            'math': math,
            'random': random,
            'datetime': datetime,
            'json': json,
        }
        
        for module_name in self.allowed_modules:
            if module_name in allowed_module_map:
                safe_globals_dict[module_name] = allowed_module_map[module_name]
        
        return safe_globals_dict
    
    def _validate_code_ast(self, code: str) -> List[str]:
        """Validate code using AST analysis."""
        violations = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.allowed_modules:
                            violations.append(f"Forbidden import: {alias.name}")
                
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.allowed_modules:
                        violations.append(f"Forbidden import from: {node.module}")
                
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.forbidden_functions:
                            violations.append(f"Forbidden function call: {node.func.id}")
                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr in self.forbidden_functions:
                            violations.append(f"Forbidden method call: {node.func.attr}")
                
                # Check for dangerous attribute access
                if isinstance(node, ast.Attribute):
                    dangerous_attrs = ['__import__', '__builtins__', '__globals__', '__locals__']
                    if node.attr in dangerous_attrs:
                        violations.append(f"Forbidden attribute access: {node.attr}")
                
                # Check for exec/eval usage
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', 'compile']:
                        violations.append(f"Forbidden dynamic execution: {node.func.id}")
        
        except SyntaxError as e:
            violations.append(f"Syntax error: {e}")
        
        return violations
    
    def execute(self, code: str, timeout_seconds: Optional[int] = None) -> ExecutionResult:
        """Execute code in safe environment."""
        start_time = time.time()
        timeout_seconds = timeout_seconds or self.safety_constraints.max_execution_time_seconds
        
        # Validate code
        violations = self._validate_code_ast(code)
        if violations:
            return ExecutionResult(
                status=ExecutionStatus.SECURITY_VIOLATION,
                output="",
                error="Security violations detected",
                security_violations=violations,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Compile restricted code
        try:
            compiled_code = compile_restricted(code, '<string>', 'exec')
            if compiled_code is None:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    output="",
                    error="Failed to compile restricted code",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Compilation error: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Create safe execution environment
        safe_globals = self._create_safe_globals()
        safe_locals = {}
        
        # Capture output
        output_buffer = []
        
        def safe_print(*args, **kwargs):
            output_buffer.append(' '.join(str(arg) for arg in args))
        
        safe_globals['print'] = safe_print
        
        # Execute with timeout
        def execute_with_timeout():
            try:
                exec(compiled_code, safe_globals, safe_locals)
                return True
            except Exception as e:
                output_buffer.append(f"Execution error: {e}")
                return False
        
        # Run with timeout
        execution_thread = threading.Thread(target=execute_with_timeout)
        execution_thread.daemon = True
        execution_thread.start()
        execution_thread.join(timeout_seconds)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        if execution_thread.is_alive():
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                output='\n'.join(output_buffer),
                error="Execution timed out",
                execution_time_ms=execution_time_ms
            )
        
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output='\n'.join(output_buffer),
            execution_time_ms=execution_time_ms,
            metadata={'globals_count': len(safe_globals), 'locals_count': len(safe_locals)}
        )


class DockerSandbox:
    """
    Docker-based sandbox for maximum isolation.
    Requires Docker to be installed and running.
    """
    
    def __init__(
        self,
        image: str = "python:3.9-slim",
        memory_limit: str = "512m",
        cpu_limit: str = "0.5",
        timeout_seconds: int = 300,
        network_disabled: bool = True
    ):
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout_seconds = timeout_seconds
        self.network_disabled = network_disabled
        
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Docker sandbox is available."""
        return self.client is not None
    
    def execute(self, code: str, timeout_seconds: Optional[int] = None) -> ExecutionResult:
        """Execute code in Docker container."""
        if not self.is_available():
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error="Docker sandbox not available"
            )
        
        start_time = time.time()
        timeout_seconds = timeout_seconds or self.timeout_seconds
        
        try:
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                code_file = f.name
            
            # Run container
            container = self.client.containers.run(
                self.image,
                command=f"python {os.path.basename(code_file)}",
                volumes={os.path.dirname(code_file): {'bind': '/code', 'mode': 'ro'}},
                working_dir='/code',
                mem_limit=self.memory_limit,
                cpu_quota=int(float(self.cpu_limit) * 100000),
                cpu_period=100000,
                network_disabled=self.network_disabled,
                detach=True,
                stdout=True,
                stderr=True,
                remove=True
            )
            
            # Wait for completion
            try:
                result = container.wait(timeout=timeout_seconds)
                logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                if result['StatusCode'] == 0:
                    return ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        output=logs,
                        execution_time_ms=execution_time_ms,
                        metadata={'container_id': container.id}
                    )
                else:
                    return ExecutionResult(
                        status=ExecutionStatus.ERROR,
                        output=logs,
                        error=f"Container exited with code {result['StatusCode']}",
                        execution_time_ms=execution_time_ms
                    )
            
            except docker.errors.ContainerError as e:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    output="",
                    error=f"Container error: {e}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            finally:
                # Clean up
                try:
                    container.stop()
                    container.remove()
                except:
                    pass
                
                # Remove temporary file
                try:
                    os.unlink(code_file)
                except:
                    pass
        
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Docker execution error: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            )


class SubprocessSandbox:
    """
    Subprocess-based sandbox with resource limits.
    Provides moderate isolation using OS-level process controls.
    """
    
    def __init__(
        self,
        python_executable: str = sys.executable,
        memory_limit_mb: int = 512,
        cpu_time_limit_seconds: int = 30,
        timeout_seconds: int = 300
    ):
        self.python_executable = python_executable
        self.memory_limit_mb = memory_limit_mb
        self.cpu_time_limit_seconds = cpu_time_limit_seconds
        self.timeout_seconds = timeout_seconds
    
    def _set_resource_limits(self):
        """Set resource limits for the subprocess."""
        # Memory limit
        memory_limit_bytes = self.memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        
        # CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_time_limit_seconds, self.cpu_time_limit_seconds))
        
        # Disable core dumps
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        
        # Limit number of processes
        resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
    
    def execute(self, code: str, timeout_seconds: Optional[int] = None) -> ExecutionResult:
        """Execute code in subprocess with resource limits."""
        start_time = time.time()
        timeout_seconds = timeout_seconds or self.timeout_seconds
        
        try:
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                code_file = f.name
            
            try:
                # Run subprocess with resource limits
                process = subprocess.Popen(
                    [self.python_executable, code_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=self._set_resource_limits,
                    text=True
                )
                
                # Wait for completion
                try:
                    stdout, stderr = process.communicate(timeout=timeout_seconds)
                    execution_time_ms = (time.time() - start_time) * 1000
                    
                    if process.returncode == 0:
                        return ExecutionResult(
                            status=ExecutionStatus.SUCCESS,
                            output=stdout,
                            execution_time_ms=execution_time_ms,
                            metadata={'return_code': process.returncode}
                        )
                    else:
                        return ExecutionResult(
                            status=ExecutionStatus.ERROR,
                            output=stdout,
                            error=stderr,
                            execution_time_ms=execution_time_ms,
                            metadata={'return_code': process.returncode}
                        )
                
                except subprocess.TimeoutExpired:
                    process.kill()
                    return ExecutionResult(
                        status=ExecutionStatus.TIMEOUT,
                        output="",
                        error="Execution timed out",
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(code_file)
                except:
                    pass
        
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Subprocess execution error: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            )


class RSISandbox:
    """
    Main RSI sandbox that orchestrates different sandbox types.
    Provides fallback mechanisms and comprehensive safety.
    """
    
    def __init__(
        self,
        primary_sandbox: SandboxType = SandboxType.RESTRICTED_PYTHON,
        fallback_sandbox: Optional[SandboxType] = None,
        safety_constraints: Optional[SafetyConstraints] = None,
        validator: Optional[RSIValidator] = None
    ):
        self.primary_sandbox = primary_sandbox
        self.fallback_sandbox = fallback_sandbox
        self.safety_constraints = safety_constraints or SafetyConstraints()
        self.validator = validator
        
        # Initialize sandbox implementations
        self.restricted_python = SafeExecutionEnvironment(safety_constraints)
        self.docker_sandbox = DockerSandbox(
            memory_limit=f"{self.safety_constraints.max_memory_mb}m",
            timeout_seconds=self.safety_constraints.max_execution_time_seconds
        )
        self.subprocess_sandbox = SubprocessSandbox(
            memory_limit_mb=self.safety_constraints.max_memory_mb,
            timeout_seconds=self.safety_constraints.max_execution_time_seconds
        )
        
        # Execution history
        self.execution_history: List[ExecutionResult] = []
        
        logger.info(f"RSI Sandbox initialized with primary: {primary_sandbox}")
    
    @trace_operation("sandbox_execute")
    def execute(
        self,
        code: str,
        timeout_seconds: Optional[int] = None,
        sandbox_type: Optional[SandboxType] = None
    ) -> ExecutionResult:
        """
        Execute code in the specified sandbox.
        
        Args:
            code: Python code to execute
            timeout_seconds: Execution timeout
            sandbox_type: Override sandbox type
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        sandbox_type = sandbox_type or self.primary_sandbox
        
        # Validate code first
        if self.validator:
            validation_result = self.validator.validate_code(code)
            if not validation_result.valid:
                record_safety_event(
                    "code_validation_failed",
                    "error",
                    {"validation_errors": validation_result.field_errors}
                )
                return ExecutionResult(
                    status=ExecutionStatus.SECURITY_VIOLATION,
                    output="",
                    error=f"Code validation failed: {validation_result.message}",
                    security_violations=validation_result.field_errors.get("code", []),
                    execution_time_ms=(time.time() - start_time) * 1000
                )
        
        # Execute in primary sandbox
        result = self._execute_in_sandbox(code, sandbox_type, timeout_seconds)
        
        # Try fallback if primary fails
        if (result.status in [ExecutionStatus.ERROR, ExecutionStatus.SECURITY_VIOLATION] and
            self.fallback_sandbox and
            self.fallback_sandbox != sandbox_type):
            
            logger.warning(f"Primary sandbox failed, trying fallback: {self.fallback_sandbox}")
            fallback_result = self._execute_in_sandbox(code, self.fallback_sandbox, timeout_seconds)
            
            if fallback_result.status == ExecutionStatus.SUCCESS:
                result = fallback_result
                result.metadata = result.metadata or {}
                result.metadata["used_fallback"] = True
        
        # Record execution
        self.execution_history.append(result)
        
        # Keep only recent executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
        
        # Record safety events
        if result.status == ExecutionStatus.SECURITY_VIOLATION:
            record_safety_event(
                "security_violation",
                "critical",
                {"violations": result.security_violations}
            )
        elif result.status == ExecutionStatus.TIMEOUT:
            record_safety_event(
                "execution_timeout",
                "warning",
                {"timeout_seconds": timeout_seconds}
            )
        
        return result
    
    def _execute_in_sandbox(
        self,
        code: str,
        sandbox_type: SandboxType,
        timeout_seconds: Optional[int]
    ) -> ExecutionResult:
        """Execute code in specific sandbox type."""
        try:
            if sandbox_type == SandboxType.RESTRICTED_PYTHON:
                return self.restricted_python.execute(code, timeout_seconds)
            
            elif sandbox_type == SandboxType.DOCKER_CONTAINER:
                if self.docker_sandbox.is_available():
                    return self.docker_sandbox.execute(code, timeout_seconds)
                else:
                    return ExecutionResult(
                        status=ExecutionStatus.ERROR,
                        output="",
                        error="Docker sandbox not available"
                    )
            
            elif sandbox_type == SandboxType.SUBPROCESS:
                return self.subprocess_sandbox.execute(code, timeout_seconds)
            
            else:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    output="",
                    error=f"Unknown sandbox type: {sandbox_type}"
                )
        
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output="",
                error=f"Sandbox execution error: {e}"
            )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total = len(self.execution_history)
        success = sum(1 for r in self.execution_history if r.status == ExecutionStatus.SUCCESS)
        timeouts = sum(1 for r in self.execution_history if r.status == ExecutionStatus.TIMEOUT)
        errors = sum(1 for r in self.execution_history if r.status == ExecutionStatus.ERROR)
        security_violations = sum(1 for r in self.execution_history if r.status == ExecutionStatus.SECURITY_VIOLATION)
        
        execution_times = [r.execution_time_ms for r in self.execution_history if r.execution_time_ms > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "total_executions": total,
            "success_count": success,
            "timeout_count": timeouts,
            "error_count": errors,
            "security_violation_count": security_violations,
            "success_rate": success / total if total > 0 else 0,
            "avg_execution_time_ms": avg_execution_time,
            "recent_violations": [
                r.security_violations for r in self.execution_history[-10:]
                if r.status == ExecutionStatus.SECURITY_VIOLATION
            ]
        }


# Factory functions for common sandbox configurations
def create_development_sandbox() -> RSISandbox:
    """Create sandbox for development with relaxed constraints."""
    constraints = SafetyConstraints(
        max_memory_mb=1024,
        max_execution_time_seconds=120,
        allowed_modules=['math', 'random', 'datetime', 'json', 'collections']
    )
    
    return RSISandbox(
        primary_sandbox=SandboxType.RESTRICTED_PYTHON,
        fallback_sandbox=SandboxType.SUBPROCESS,
        safety_constraints=constraints
    )


def create_production_sandbox() -> RSISandbox:
    """Create sandbox for production with strict constraints."""
    constraints = SafetyConstraints(
        max_memory_mb=512,
        max_execution_time_seconds=60,
        allowed_modules=['math', 'random', 'datetime', 'json']
    )
    
    return RSISandbox(
        primary_sandbox=SandboxType.DOCKER_CONTAINER,
        fallback_sandbox=SandboxType.RESTRICTED_PYTHON,
        safety_constraints=constraints
    )


def create_strict_sandbox() -> RSISandbox:
    """Create sandbox with maximum security constraints."""
    constraints = SafetyConstraints(
        max_memory_mb=256,
        max_execution_time_seconds=30,
        allowed_modules=['math', 'random']
    )
    
    return RSISandbox(
        primary_sandbox=SandboxType.RESTRICTED_PYTHON,
        safety_constraints=constraints
    )