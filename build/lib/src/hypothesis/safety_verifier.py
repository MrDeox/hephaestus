"""
Advanced RSI Safety Verification and Secure Execution System.
Implements CodeJail, Docker isolation, and comprehensive safety measures.
"""

import asyncio
import time
import json
import tempfile
import subprocess
import os
import shutil
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import uuid
from loguru import logger

try:
    import docker
    from docker.types import Mount, Resources, Ulimit
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    logger.warning("Docker not available - using fallback isolation")

try:
    import RestrictedPython
    from RestrictedPython import compile_restricted, safe_globals
    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    logger.warning("RestrictedPython not available - using basic restrictions")

from .hypothesis_generator import RSIHypothesis, HypothesisType
from .hypothesis_validator import HypothesisValidationResult
from ..security.sandbox import RSISandbox
from ..safety.circuits import RSICircuitBreaker


class IsolationLevel(str, Enum):
    BASIC = "basic"                    # Basic Python restrictions
    RESTRICTED_PYTHON = "restricted"   # RestrictedPython sandbox
    DOCKER_CONTAINER = "docker"        # Full Docker isolation
    HYBRID = "hybrid"                  # RestrictedPython + Docker


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    KILLED = "killed"


@dataclass
class SafetyConstraints:
    """Comprehensive safety constraints for hypothesis execution"""
    max_execution_time_seconds: int = 300
    max_memory_mb: int = 1024
    max_cpu_percent: float = 50.0
    max_disk_io_mb: int = 100
    max_network_requests: int = 0
    allowed_modules: List[str] = None
    forbidden_functions: List[str] = None
    require_approval: bool = False
    isolation_level: IsolationLevel = IsolationLevel.DOCKER_CONTAINER
    
    def __post_init__(self):
        if self.allowed_modules is None:
            self.allowed_modules = [
                'math', 'random', 'datetime', 'json', 'collections',
                'itertools', 'functools', 'operator', 'statistics',
                'numpy', 'pandas', 'sklearn', 'scipy'
            ]
        if self.forbidden_functions is None:
            self.forbidden_functions = [
                'exec', 'eval', 'compile', 'open', '__import__',
                'globals', 'locals', 'vars', 'dir', 'getattr',
                'setattr', 'delattr', 'hasattr', 'input', 'raw_input'
            ]


@dataclass
class ExecutionResult:
    """Result of safe hypothesis execution"""
    execution_id: str
    hypothesis_id: str
    status: ExecutionStatus
    isolation_level: IsolationLevel
    
    # Results
    output: Optional[Any] = None
    logs: List[str] = None
    metrics: Dict[str, Any] = None
    
    # Execution metadata
    start_time: float = None
    end_time: float = None
    execution_time_ms: float = None
    memory_usage_mb: float = None
    cpu_usage_percent: float = None
    
    # Safety information
    safety_violations: List[str] = None
    resource_violations: List[str] = None
    security_issues: List[str] = None
    
    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.safety_violations is None:
            self.safety_violations = []
        if self.resource_violations is None:
            self.resource_violations = []
        if self.security_issues is None:
            self.security_issues = []


class RSISafetyVerifier:
    """
    Comprehensive safety verification and secure execution system.
    Implements multiple isolation layers and comprehensive monitoring.
    """
    
    def __init__(self, 
                 default_constraints: Optional[SafetyConstraints] = None,
                 sandbox: Optional[RSISandbox] = None,
                 circuit_breaker: Optional[RSICircuitBreaker] = None):
        
        self.default_constraints = default_constraints or SafetyConstraints()
        self.sandbox = sandbox
        self.circuit_breaker = circuit_breaker
        
        # Docker client
        self.docker_client = None
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize Docker client: {}", str(e))
                DOCKER_AVAILABLE = False
        
        # Execution tracking
        self.active_executions: Dict[str, ExecutionResult] = {}
        self.execution_history: List[ExecutionResult] = []
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'security_violations': 0,
            'resource_violations': 0
        }
        
        # Safety monitoring
        self.monitoring_active = False
        self.monitor_task = None
        
        logger.info("RSI Safety Verifier initialized with {} isolation", 
                   self.default_constraints.isolation_level.value)
    
    async def execute_hypothesis_safely(self, 
                                      hypothesis: RSIHypothesis,
                                      validation_result: HypothesisValidationResult,
                                      constraints: Optional[SafetyConstraints] = None,
                                      context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute RSI hypothesis with comprehensive safety measures.
        
        Args:
            hypothesis: The hypothesis to execute
            validation_result: Previous validation results
            constraints: Safety constraints to apply
            context: Additional execution context
            
        Returns:
            Comprehensive execution result
        """
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        constraints = constraints or self.default_constraints
        
        logger.info("Starting safe execution {} for hypothesis {} with {} isolation", 
                   execution_id, hypothesis.hypothesis_id, constraints.isolation_level.value)
        
        # Create execution result
        result = ExecutionResult(
            execution_id=execution_id,
            hypothesis_id=hypothesis.hypothesis_id,
            status=ExecutionStatus.PENDING,
            isolation_level=constraints.isolation_level,
            start_time=time.time()
        )
        
        self.active_executions[execution_id] = result
        
        try:
            # Pre-execution safety checks
            await self._perform_pre_execution_checks(hypothesis, validation_result, constraints)
            
            # Start monitoring
            monitor_task = asyncio.create_task(self._monitor_execution(execution_id, constraints))
            
            result.status = ExecutionStatus.RUNNING
            
            # Execute based on isolation level
            if constraints.isolation_level == IsolationLevel.DOCKER_CONTAINER:
                execution_result = await self._execute_in_docker(hypothesis, constraints, context)
            elif constraints.isolation_level == IsolationLevel.RESTRICTED_PYTHON:
                execution_result = await self._execute_in_restricted_python(hypothesis, constraints, context)
            elif constraints.isolation_level == IsolationLevel.HYBRID:
                execution_result = await self._execute_hybrid(hypothesis, constraints, context)
            else:  # BASIC
                execution_result = await self._execute_basic(hypothesis, constraints, context)
            
            # Stop monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            
            # Update result with execution data
            result.output = execution_result.get('output')
            result.metrics = execution_result.get('metrics', {})
            result.logs.extend(execution_result.get('logs', []))
            result.memory_usage_mb = execution_result.get('memory_usage_mb', 0)
            result.cpu_usage_percent = execution_result.get('cpu_usage_percent', 0)
            
            # Post-execution safety validation
            await self._perform_post_execution_checks(result, constraints)
            
            result.status = ExecutionStatus.COMPLETED
            result.end_time = time.time()
            result.execution_time_ms = (result.end_time - result.start_time) * 1000
            
            logger.info("Execution {} completed successfully in {:.2f}ms", 
                       execution_id, result.execution_time_ms)
            
        except asyncio.TimeoutError:
            result.status = ExecutionStatus.TIMEOUT
            result.error_message = f"Execution exceeded time limit of {constraints.max_execution_time_seconds}s"
            logger.warning("Execution {} timed out", execution_id)
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error_message = str(e)
            result.error_type = type(e).__name__
            logger.error("Execution {} failed: {}", execution_id, str(e))
            
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            # Add to history
            self.execution_history.append(result)
            if len(self.execution_history) > 1000:
                self.execution_history.pop(0)
            
            # Update statistics
            self._update_execution_stats(result)
        
        return result
    
    async def _execute_in_docker(self, 
                                hypothesis: RSIHypothesis,
                                constraints: SafetyConstraints,
                                context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute hypothesis in secure Docker container"""
        
        if not DOCKER_AVAILABLE or not self.docker_client:
            raise RuntimeError("Docker not available for secure execution")
        
        # Create temporary directory for code and results
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write hypothesis code to file
            code_file = temp_path / "hypothesis_code.py"
            hypothesis_code = self._generate_hypothesis_code(hypothesis, context)
            
            with open(code_file, 'w') as f:
                f.write(hypothesis_code)
            
            # Create results file
            results_file = temp_path / "results.json"
            
            # Configure container resources
            resources = Resources(
                mem_limit=f"{constraints.max_memory_mb}m",
                cpu_period=100000,
                cpu_quota=int(constraints.max_cpu_percent * 1000),
                ulimits=[
                    Ulimit(name='fsize', soft=constraints.max_disk_io_mb * 1024 * 1024),
                    Ulimit(name='nproc', soft=100)
                ]
            )
            
            # Configure mounts
            mounts = [
                Mount('/workspace', str(temp_path), type='bind', read_only=False),
                Mount('/tmp', None, type='tmpfs', tmpfs_size=f"{constraints.max_memory_mb // 4}m")
            ]
            
            # Container configuration
            container_config = {
                'image': 'python:3.11-slim',
                'command': [
                    'python', '-c', f'''
import sys
sys.path.insert(0, "/workspace")
import json
import time
import traceback
import resource
import os

try:
    # Set resource limits
    resource.setrlimit(resource.RLIMIT_CPU, ({constraints.max_execution_time_seconds}, {constraints.max_execution_time_seconds}))
    resource.setrlimit(resource.RLIMIT_AS, ({constraints.max_memory_mb * 1024 * 1024}, {constraints.max_memory_mb * 1024 * 1024}))
    
    # Import and execute hypothesis code
    import hypothesis_code
    
    # Execute hypothesis
    start_time = time.time()
    result = hypothesis_code.execute_hypothesis()
    end_time = time.time()
    
    # Collect metrics
    usage = resource.getrusage(resource.RUSAGE_SELF)
    
    output_data = {{
        "success": True,
        "result": result,
        "execution_time": end_time - start_time,
        "memory_usage_kb": usage.ru_maxrss,
        "cpu_time": usage.ru_utime + usage.ru_stime,
        "logs": []
    }}
    
    with open("/workspace/results.json", "w") as f:
        json.dump(output_data, f, default=str)
        
except Exception as e:
    error_data = {{
        "success": False,
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc(),
        "logs": []
    }}
    
    with open("/workspace/results.json", "w") as f:
        json.dump(error_data, f)
'''
                ],
                'working_dir': '/workspace',
                'user': 'nobody',
                'network_disabled': constraints.max_network_requests == 0,
                'cap_drop': ['ALL'],
                'security_opt': ['no-new-privileges'],
                'read_only': False,
                'remove': True,
                'stdout': True,
                'stderr': True,
                'resources': resources,
                'mounts': mounts
            }
            
            # Run container with timeout
            try:
                container = self.docker_client.containers.run(
                    **container_config,
                    timeout=constraints.max_execution_time_seconds + 10,
                    detach=False
                )
                
                # Read results
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    if results.get('success'):
                        return {
                            'output': results.get('result'),
                            'metrics': {
                                'execution_time': results.get('execution_time', 0),
                                'memory_usage_kb': results.get('memory_usage_kb', 0),
                                'cpu_time': results.get('cpu_time', 0)
                            },
                            'logs': results.get('logs', []),
                            'memory_usage_mb': results.get('memory_usage_kb', 0) / 1024,
                            'cpu_usage_percent': min(100, results.get('cpu_time', 0) / results.get('execution_time', 1) * 100)
                        }
                    else:
                        raise RuntimeError(f"Container execution failed: {results.get('error', 'Unknown error')}")
                else:
                    raise RuntimeError("No results file generated by container")
                    
            except docker.errors.ContainerError as e:
                raise RuntimeError(f"Container execution error: {e}")
            except Exception as e:
                raise RuntimeError(f"Docker execution failed: {e}")
    
    async def _execute_in_restricted_python(self, 
                                          hypothesis: RSIHypothesis,
                                          constraints: SafetyConstraints,
                                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute hypothesis using RestrictedPython"""
        
        if not RESTRICTED_PYTHON_AVAILABLE:
            raise RuntimeError("RestrictedPython not available")
        
        # Generate safe execution code
        hypothesis_code = self._generate_hypothesis_code(hypothesis, context)
        
        # Compile with RestrictedPython
        try:
            compiled_code = compile_restricted(hypothesis_code, '<hypothesis>', 'exec')
            if compiled_code is None:
                raise RuntimeError("Failed to compile restricted code")
        except Exception as e:
            raise RuntimeError(f"RestrictedPython compilation failed: {e}")
        
        # Create safe execution environment
        safe_builtins = safe_globals.copy()
        
        # Add allowed modules
        for module_name in constraints.allowed_modules:
            try:
                module = __import__(module_name)
                safe_builtins[module_name] = module
            except ImportError:
                pass  # Skip unavailable modules
        
        # Remove forbidden functions
        for func_name in constraints.forbidden_functions:
            safe_builtins.pop(func_name, None)
        
        # Execute with monitoring
        start_time = time.time()
        execution_globals = safe_builtins.copy()
        
        try:
            # Execute in subprocess for better isolation
            process = await asyncio.create_subprocess_exec(
                'python', '-c', f'''
import json
import time
import traceback
import sys
import os

# Set up restricted environment
restricted_globals = {repr(safe_builtins)}

try:
    # Execute code
    start_time = time.time()
    exec(compile({repr(hypothesis_code)}, "<hypothesis>", "exec"), restricted_globals)
    
    # Get result
    result = restricted_globals.get("result", None)
    end_time = time.time()
    
    output = {{
        "success": True,
        "result": result,
        "execution_time": end_time - start_time,
        "logs": []
    }}
    
    print(json.dumps(output, default=str))
    
except Exception as e:
    output = {{
        "success": False,
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc()
    }}
    
    print(json.dumps(output))
''',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=constraints.max_execution_time_seconds
            )
            
            # Parse results
            results = json.loads(stdout.decode())
            end_time = time.time()
            
            if results.get('success'):
                return {
                    'output': results.get('result'),
                    'metrics': {
                        'execution_time': results.get('execution_time', 0)
                    },
                    'logs': results.get('logs', []),
                    'memory_usage_mb': 0,  # Not available in RestrictedPython
                    'cpu_usage_percent': 0
                }
            else:
                raise RuntimeError(f"RestrictedPython execution failed: {results.get('error')}")
                
        except asyncio.TimeoutError:
            if process:
                process.terminate()
                await process.wait()
            raise RuntimeError("RestrictedPython execution timed out")
        except Exception as e:
            raise RuntimeError(f"RestrictedPython execution error: {e}")
    
    async def _execute_hybrid(self, 
                            hypothesis: RSIHypothesis,
                            constraints: SafetyConstraints,
                            context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute using hybrid RestrictedPython + Docker approach"""
        
        # First try RestrictedPython for speed
        restricted_constraints = SafetyConstraints(
            max_execution_time_seconds=min(30, constraints.max_execution_time_seconds),
            max_memory_mb=min(256, constraints.max_memory_mb),
            isolation_level=IsolationLevel.RESTRICTED_PYTHON,
            allowed_modules=constraints.allowed_modules,
            forbidden_functions=constraints.forbidden_functions
        )
        
        try:
            return await self._execute_in_restricted_python(hypothesis, restricted_constraints, context)
        except Exception as e:
            logger.warning("RestrictedPython execution failed, falling back to Docker: {}", str(e))
            
            # Fallback to Docker for more complex cases
            return await self._execute_in_docker(hypothesis, constraints, context)
    
    async def _execute_basic(self, 
                           hypothesis: RSIHypothesis,
                           constraints: SafetyConstraints,
                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Basic execution with minimal restrictions"""
        
        hypothesis_code = self._generate_hypothesis_code(hypothesis, context)
        
        # Execute in subprocess for basic isolation
        process = await asyncio.create_subprocess_exec(
            'python', '-c', hypothesis_code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=constraints.max_execution_time_seconds
            )
            
            # Simple result parsing
            result = stdout.decode().strip()
            
            return {
                'output': result,
                'metrics': {},
                'logs': [stderr.decode()] if stderr else [],
                'memory_usage_mb': 0,
                'cpu_usage_percent': 0
            }
            
        except asyncio.TimeoutError:
            process.terminate()
            await process.wait()
            raise RuntimeError("Basic execution timed out")
    
    def _generate_hypothesis_code(self, 
                                 hypothesis: RSIHypothesis, 
                                 context: Optional[Dict[str, Any]]) -> str:
        """Generate executable code for hypothesis"""
        
        # This is a simplified code generation
        # In practice, this would be much more sophisticated
        
        if hypothesis.hypothesis_type == HypothesisType.HYPERPARAMETER_OPTIMIZATION:
            return f'''
import numpy as np
import json

def execute_hypothesis():
    # Hyperparameter optimization simulation
    params = {json.dumps(hypothesis.parameters)}
    
    # Simulate model training with new hyperparameters
    # This would be replaced with actual model training
    
    baseline_accuracy = 0.85
    improvement = np.random.normal(0.02, 0.01)
    new_accuracy = baseline_accuracy + improvement
    
    result = {{
        "hypothesis_id": "{hypothesis.hypothesis_id}",
        "hypothesis_type": "{hypothesis.hypothesis_type.value}",
        "parameters": params,
        "baseline_accuracy": baseline_accuracy,
        "new_accuracy": new_accuracy,
        "improvement": improvement,
        "success": True
    }}
    
    return result

if __name__ == "__main__":
    result = execute_hypothesis()
    print(json.dumps(result, default=str))
'''
        
        elif hypothesis.hypothesis_type == HypothesisType.ARCHITECTURE_CHANGE:
            return f'''
import numpy as np
import json

def execute_hypothesis():
    # Architecture change simulation
    params = {json.dumps(hypothesis.parameters)}
    
    # Simulate architecture evaluation
    layer_count = params.get("layer_count", 3)
    hidden_units = params.get("hidden_units", 128)
    
    # Estimate performance based on architecture
    complexity_score = layer_count * hidden_units / 1000
    baseline_accuracy = 0.85
    
    # More complex architectures might perform better but cost more
    improvement = min(0.1, complexity_score * 0.02)
    new_accuracy = baseline_accuracy + improvement
    
    result = {{
        "hypothesis_id": "{hypothesis.hypothesis_id}",
        "hypothesis_type": "{hypothesis.hypothesis_type.value}",
        "parameters": params,
        "baseline_accuracy": baseline_accuracy,
        "new_accuracy": new_accuracy,
        "improvement": improvement,
        "complexity_score": complexity_score,
        "success": True
    }}
    
    return result

if __name__ == "__main__":
    result = execute_hypothesis()
    print(json.dumps(result, default=str))
'''
        
        else:
            return f'''
import json

def execute_hypothesis():
    # Generic hypothesis execution
    params = {json.dumps(hypothesis.parameters)}
    
    result = {{
        "hypothesis_id": "{hypothesis.hypothesis_id}",
        "hypothesis_type": "{hypothesis.hypothesis_type.value}",
        "parameters": params,
        "simulated_result": "success",
        "message": "Hypothesis executed successfully",
        "success": True
    }}
    
    return result

if __name__ == "__main__":
    result = execute_hypothesis()
    print(json.dumps(result, default=str))
'''
    
    async def _perform_pre_execution_checks(self, 
                                          hypothesis: RSIHypothesis,
                                          validation_result: HypothesisValidationResult,
                                          constraints: SafetyConstraints):
        """Perform safety checks before execution"""
        
        # Check validation results
        if not validation_result.is_valid:
            raise RuntimeError(f"Hypothesis validation failed: {validation_result.overall_status}")
        
        # Check if approval is required
        if constraints.require_approval and not validation_result.requires_human_review:
            raise RuntimeError("Human approval required but not obtained")
        
        # Check safety scores
        if validation_result.safety_score < 0.6:
            raise RuntimeError(f"Safety score too low: {validation_result.safety_score}")
        
        # Check resource availability
        # This would check actual system resources in production
        logger.debug("Pre-execution safety checks passed")
    
    async def _perform_post_execution_checks(self, 
                                           result: ExecutionResult,
                                           constraints: SafetyConstraints):
        """Perform safety checks after execution"""
        
        # Check resource usage
        if result.memory_usage_mb and result.memory_usage_mb > constraints.max_memory_mb:
            result.resource_violations.append(f"Memory usage exceeded: {result.memory_usage_mb}MB > {constraints.max_memory_mb}MB")
        
        if result.cpu_usage_percent and result.cpu_usage_percent > constraints.max_cpu_percent:
            result.resource_violations.append(f"CPU usage exceeded: {result.cpu_usage_percent}% > {constraints.max_cpu_percent}%")
        
        if result.execution_time_ms and result.execution_time_ms > constraints.max_execution_time_seconds * 1000:
            result.resource_violations.append(f"Execution time exceeded: {result.execution_time_ms}ms > {constraints.max_execution_time_seconds * 1000}ms")
        
        # Check for security issues in output
        if result.output and isinstance(result.output, str):
            if any(keyword in result.output.lower() for keyword in ['password', 'secret', 'key', 'token']):
                result.security_issues.append("Potentially sensitive information in output")
        
        logger.debug("Post-execution safety checks completed with {} violations", 
                    len(result.resource_violations) + len(result.security_issues))
    
    async def _monitor_execution(self, execution_id: str, constraints: SafetyConstraints):
        """Monitor execution for safety violations"""
        
        while execution_id in self.active_executions:
            try:
                result = self.active_executions[execution_id]
                
                # Check execution time
                if result.start_time and time.time() - result.start_time > constraints.max_execution_time_seconds:
                    logger.warning("Execution {} exceeded time limit", execution_id)
                    result.status = ExecutionStatus.TIMEOUT
                    break
                
                # Monitor system resources (simplified)
                # In production, this would monitor actual container/process resources
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error("Error monitoring execution {}: {}", execution_id, str(e))
                break
    
    def _update_execution_stats(self, result: ExecutionResult):
        """Update execution statistics"""
        self.execution_stats['total_executions'] += 1
        
        if result.status == ExecutionStatus.COMPLETED:
            self.execution_stats['successful_executions'] += 1
        else:
            self.execution_stats['failed_executions'] += 1
        
        if result.security_issues:
            self.execution_stats['security_violations'] += 1
        
        if result.resource_violations:
            self.execution_stats['resource_violations'] += 1
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get comprehensive safety and execution statistics"""
        
        total_executions = self.execution_stats['total_executions']
        
        return {
            'execution_stats': self.execution_stats.copy(),
            'active_executions': len(self.active_executions),
            'execution_history_size': len(self.execution_history),
            'success_rate': (
                self.execution_stats['successful_executions'] / max(1, total_executions)
            ),
            'security_violation_rate': (
                self.execution_stats['security_violations'] / max(1, total_executions)
            ),
            'resource_violation_rate': (
                self.execution_stats['resource_violations'] / max(1, total_executions)
            ),
            'average_execution_time_ms': np.mean([
                r.execution_time_ms for r in self.execution_history[-100:]
                if r.execution_time_ms is not None
            ]) if self.execution_history else 0.0,
            'isolation_capabilities': {
                'docker_available': DOCKER_AVAILABLE,
                'restricted_python_available': RESTRICTED_PYTHON_AVAILABLE,
                'default_isolation_level': self.default_constraints.isolation_level.value
            },
            'safety_features': {
                'resource_monitoring': True,
                'security_scanning': True,
                'circuit_breaker_protection': self.circuit_breaker is not None,
                'sandbox_integration': self.sandbox is not None
            }
        }
    
    async def cleanup(self):
        """Cleanup resources and stop monitoring"""
        
        # Stop monitoring
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup active executions
        for execution_id in list(self.active_executions.keys()):
            logger.warning("Cleaning up active execution: {}", execution_id)
            # In production, this would properly terminate containers/processes
        
        # Close Docker client
        if self.docker_client:
            try:
                self.docker_client.close()
                logger.info("Docker client closed")
            except Exception as e:
                logger.warning("Error closing Docker client: {}", str(e))
        
        logger.info("Safety verifier cleanup completed")


# Factory functions for different safety levels
def create_strict_safety_constraints() -> SafetyConstraints:
    """Create strict safety constraints for production"""
    return SafetyConstraints(
        max_execution_time_seconds=60,
        max_memory_mb=512,
        max_cpu_percent=25.0,
        max_disk_io_mb=50,
        max_network_requests=0,
        require_approval=True,
        isolation_level=IsolationLevel.RESTRICTED_PYTHON,
        allowed_modules=['math', 'random', 'datetime', 'json'],
        forbidden_functions=[
            'exec', 'eval', 'compile', 'open', '__import__',
            'globals', 'locals', 'vars', 'dir', 'getattr',
            'setattr', 'delattr', 'hasattr', 'input'
        ]
    )


def create_development_safety_constraints() -> SafetyConstraints:
    """Create relaxed safety constraints for development"""
    return SafetyConstraints(
        max_execution_time_seconds=300,
        max_memory_mb=1024,
        max_cpu_percent=50.0,
        max_disk_io_mb=100,
        max_network_requests=0,
        require_approval=False,
        isolation_level=IsolationLevel.HYBRID,
        allowed_modules=[
            'math', 'random', 'datetime', 'json', 'collections',
            'itertools', 'functools', 'operator', 'statistics',
            'numpy', 'pandas', 'sklearn'
        ]
    )


def create_experimental_safety_constraints() -> SafetyConstraints:
    """Create constraints for experimental hypothesis testing"""
    return SafetyConstraints(
        max_execution_time_seconds=600,
        max_memory_mb=2048,
        max_cpu_percent=75.0,
        max_disk_io_mb=500,
        max_network_requests=0,
        require_approval=True,
        isolation_level=IsolationLevel.RESTRICTED_PYTHON
    )