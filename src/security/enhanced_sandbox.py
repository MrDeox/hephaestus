"""
Enhanced Security Sandbox for Hephaestus RSI.

Implements multiple layers of security including network isolation,
filesystem restrictions, resource limits, and code analysis.
"""

import ast
import asyncio
import contextlib
import docker
import hashlib
import inspect
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
import signal
import psutil

from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_globals
from RestrictedPython.transformer import RestrictingNodeTransformer

from ..common.exceptions import (
    SandboxError, SecurityViolationError, TimeoutError,
    create_error_context
)
from ..common.resource_manager import get_resource_manager, ResourceType
from config.base_config import get_config


class SecurityLevel(str, Enum):
    """Security levels for sandbox execution."""
    MINIMAL = "minimal"      # Basic restrictions
    STANDARD = "standard"    # Default security 
    HIGH = "high"           # Strict security
    MAXIMUM = "maximum"     # Paranoid security


class SandboxType(str, Enum):
    """Types of sandbox implementations."""
    RESTRICTED_PYTHON = "restricted_python"
    SUBPROCESS = "subprocess"
    DOCKER = "docker"
    VIRTUAL_MACHINE = "virtual_machine"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    allowed_imports: Set[str] = field(default_factory=set)
    blocked_imports: Set[str] = field(default_factory=set)
    allowed_builtins: Set[str] = field(default_factory=set)
    blocked_builtins: Set[str] = field(default_factory=set)
    max_execution_time: int = 30
    max_memory_mb: int = 256
    max_cpu_percent: float = 50.0
    allow_file_access: bool = False
    allowed_file_paths: Set[str] = field(default_factory=set)
    allow_network_access: bool = False
    allowed_network_hosts: Set[str] = field(default_factory=set)
    max_subprocess_count: int = 0
    max_thread_count: int = 1
    enable_ast_validation: bool = True
    enable_bytecode_analysis: bool = True
    enable_runtime_monitoring: bool = True


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""
    success: bool
    result: Any = None
    output: str = ""
    error_message: str = ""
    execution_time: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    security_violations: List[str] = field(default_factory=list)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeAnalyzer:
    """Analyzes code for security vulnerabilities."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.dangerous_patterns = {
            'eval': ['eval', 'exec', 'compile'],
            'imports': ['__import__', 'importlib'],
            'file_operations': ['open', 'file', 'input', 'raw_input'],
            'system_calls': ['system', 'popen', 'subprocess'],
            'network': ['socket', 'urllib', 'requests', 'http'],
            'introspection': ['globals', 'locals', 'vars', 'dir', 'getattr', 'setattr'],
            'dangerous_modules': ['os', 'sys', 'subprocess', 'socket', 'pickle', 'marshal']
        }
    
    def analyze_code(self, code: str) -> List[str]:
        """Analyze code and return list of security violations."""
        violations = []
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Analyze AST nodes
            violations.extend(self._analyze_ast(tree))
            
            # Analyze string patterns
            violations.extend(self._analyze_string_patterns(code))
            
            # Check imports
            violations.extend(self._analyze_imports(tree))
            
        except SyntaxError as e:
            violations.append(f"Syntax error: {e}")
        except Exception as e:
            violations.append(f"Code analysis error: {e}")
        
        return violations
    
    def _analyze_ast(self, tree: ast.AST) -> List[str]:
        """Analyze AST for dangerous constructs."""
        violations = []
        
        for node in ast.walk(tree):
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    # Check dangerous built-ins
                    for category, functions in self.dangerous_patterns.items():
                        if func_name in functions:
                            violations.append(f"Dangerous function call: {func_name} ({category})")
                
                # Check attribute access on dangerous modules
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        module_name = node.func.value.id
                        if module_name in self.dangerous_patterns['dangerous_modules']:
                            violations.append(f"Dangerous module method call: {module_name}.{node.func.attr}")
            
            # Check imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.policy.blocked_imports:
                        violations.append(f"Blocked import: {alias.name}")
                    if (self.policy.allowed_imports and 
                        alias.name not in self.policy.allowed_imports):
                        violations.append(f"Unauthorized import: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name in self.policy.blocked_imports:
                    violations.append(f"Blocked import from: {module_name}")
                if (self.policy.allowed_imports and 
                    module_name not in self.policy.allowed_imports):
                    violations.append(f"Unauthorized import from: {module_name}")
            
            # Check attribute access
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    # Check access to dangerous attributes
                    if (node.value.id == '__builtins__' or 
                        node.attr in ['__globals__', '__locals__', '__dict__']):
                        violations.append(f"Dangerous attribute access: {node.value.id}.{node.attr}")
        
        return violations
    
    def _analyze_string_patterns(self, code: str) -> List[str]:
        """Analyze code for dangerous string patterns."""
        violations = []
        
        # Check for suspicious patterns
        suspicious_patterns = [
            ('rm -rf', 'Potential file deletion command'),
            ('subprocess.', 'Subprocess usage'),
            ('os.system', 'System command execution'),
            ('__import__', 'Dynamic import'),
            ('exec(', 'Dynamic execution'),
            ('eval(', 'Dynamic evaluation'),
        ]
        
        for pattern, description in suspicious_patterns:
            if pattern in code:
                violations.append(f"Suspicious pattern detected: {description}")
        
        return violations
    
    def _analyze_imports(self, tree: ast.AST) -> List[str]:
        """Analyze import statements."""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Track all imports
                pass  # Could implement import tracking here
        
        return violations


class RestrictedPythonSandbox:
    """RestrictedPython-based sandbox implementation."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.analyzer = CodeAnalyzer(policy)
    
    async def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute code in RestrictedPython sandbox."""
        start_time = time.time()
        result = ExecutionResult(success=False)
        
        try:
            # Analyze code first
            violations = self.analyzer.analyze_code(code)
            if violations:
                result.security_violations = violations
                result.error_message = f"Security violations: {'; '.join(violations)}"
                return result
            
            # Compile restricted code
            try:
                compiled_code = compile_restricted(code, '<sandbox>', 'exec')
                if compiled_code is None:
                    result.error_message = "Failed to compile restricted code"
                    return result
            except SyntaxError as e:
                result.error_message = f"Syntax error: {e}"
                return result
            
            # Prepare execution environment
            restricted_globals = self._create_restricted_globals(context)
            restricted_locals = {}
            
            # Execute with timeout
            execution_task = asyncio.create_task(
                self._execute_with_monitoring(compiled_code, restricted_globals, restricted_locals)
            )
            
            try:
                exec_result = await asyncio.wait_for(
                    execution_task,
                    timeout=self.policy.max_execution_time
                )
                result.success = True
                result.result = exec_result
                
            except asyncio.TimeoutError:
                execution_task.cancel()
                result.error_message = f"Execution timeout ({self.policy.max_execution_time}s)"
                
        except Exception as e:
            result.error_message = f"Execution error: {e}"
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    def _create_restricted_globals(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create restricted global environment."""
        # Start with safe globals
        restricted_globals = safe_globals.copy()
        
        # Add allowed builtins
        if self.policy.allowed_builtins:
            for builtin_name in self.policy.allowed_builtins:
                if builtin_name in __builtins__:
                    restricted_globals[builtin_name] = __builtins__[builtin_name]
        
        # Remove blocked builtins
        for builtin_name in self.policy.blocked_builtins:
            restricted_globals.pop(builtin_name, None)
        
        # Add safe imports
        safe_modules = {
            'math': __import__('math'),
            'random': __import__('random'),
            'datetime': __import__('datetime'),
            'json': __import__('json'),
            're': __import__('re'),
            'collections': __import__('collections'),
            'itertools': __import__('itertools'),
            'functools': __import__('functools'),
        }
        
        for module_name in self.policy.allowed_imports:
            if module_name in safe_modules:
                restricted_globals[module_name] = safe_modules[module_name]
        
        # Add context variables
        if context:
            for key, value in context.items():
                if not key.startswith('_'):  # Don't allow private variables
                    restricted_globals[key] = value
        
        return restricted_globals
    
    async def _execute_with_monitoring(
        self,
        compiled_code,
        restricted_globals: Dict[str, Any],
        restricted_locals: Dict[str, Any]
    ) -> Any:
        """Execute code with resource monitoring."""
        # Monitor resource usage during execution
        resource_manager = get_resource_manager()
        
        async with resource_manager.allocate_async_resource(
            ResourceType.MEMORY,
            self.policy.max_memory_mb
        ):
            # Execute in thread to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: exec(compiled_code, restricted_globals, restricted_locals)
            )


class SubprocessSandbox:
    """Subprocess-based sandbox implementation."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.analyzer = CodeAnalyzer(policy)
    
    async def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute code in subprocess sandbox."""
        start_time = time.time()
        result = ExecutionResult(success=False)
        
        try:
            # Analyze code first
            violations = self.analyzer.analyze_code(code)
            if violations:
                result.security_violations = violations
                result.error_message = f"Security violations: {'; '.join(violations)}"
                return result
            
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                script_path = f.name
                
                # Write code to file
                f.write(code)
                f.flush()
            
            try:
                # Execute in subprocess with limits
                process = await self._create_limited_subprocess(script_path)
                
                # Wait for completion with timeout
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.policy.max_execution_time
                    )
                    
                    if process.returncode == 0:
                        result.success = True
                        result.output = stdout.decode('utf-8')
                    else:
                        result.error_message = stderr.decode('utf-8')
                        
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    result.error_message = f"Execution timeout ({self.policy.max_execution_time}s)"
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(script_path)
                except OSError:
                    pass
                
        except Exception as e:
            result.error_message = f"Subprocess execution error: {e}"
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    async def _create_limited_subprocess(self, script_path: str) -> asyncio.subprocess.Process:
        """Create subprocess with resource limits."""
        # Prepare environment
        env = os.environ.copy()
        
        # Remove potentially dangerous environment variables
        dangerous_env_vars = ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PATH']
        for var in dangerous_env_vars:
            env.pop(var, None)
        
        # Set safe PATH
        env['PATH'] = '/usr/bin:/bin'
        
        # Create subprocess
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            preexec_fn=self._setup_process_limits
        )
        
        return process
    
    def _setup_process_limits(self):
        """Setup process resource limits."""
        import resource
        
        # Set memory limit
        if self.policy.max_memory_mb:
            memory_limit = self.policy.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        
        # Set CPU limit
        if self.policy.max_execution_time:
            resource.setrlimit(resource.RLIMIT_CPU, (self.policy.max_execution_time, self.policy.max_execution_time))
        
        # Limit number of processes
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
        
        # Limit file operations if not allowed
        if not self.policy.allow_file_access:
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


class DockerSandbox:
    """Docker-based sandbox implementation."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.analyzer = CodeAnalyzer(policy)
        self.docker_client = None
        self._initialize_docker()
    
    def _initialize_docker(self):
        """Initialize Docker client."""
        try:
            self.docker_client = docker.from_env()
            # Test Docker connectivity
            self.docker_client.ping()
        except Exception as e:
            print(f"Docker not available: {e}")
            self.docker_client = None
    
    async def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute code in Docker sandbox."""
        if not self.docker_client:
            return ExecutionResult(
                success=False,
                error_message="Docker not available"
            )
        
        start_time = time.time()
        result = ExecutionResult(success=False)
        
        try:
            # Analyze code first
            violations = self.analyzer.analyze_code(code)
            if violations:
                result.security_violations = violations
                result.error_message = f"Security violations: {'; '.join(violations)}"
                return result
            
            # Create container and execute
            container = None
            try:
                container = await self._create_container(code)
                output = await self._run_container(container)
                
                result.success = True
                result.output = output
                
            except Exception as e:
                result.error_message = f"Docker execution error: {e}"
            
            finally:
                if container:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass
                
        except Exception as e:
            result.error_message = f"Docker sandbox error: {e}"
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    async def _create_container(self, code: str):
        """Create Docker container for code execution."""
        # Create Dockerfile content
        dockerfile_content = f"""
FROM python:3.11-alpine
RUN adduser -D -s /bin/sh sandbox
USER sandbox
WORKDIR /home/sandbox
COPY code.py /home/sandbox/
CMD ["python", "code.py"]
"""
        
        # Create temporary directory for build context
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write Dockerfile
            (temp_path / "Dockerfile").write_text(dockerfile_content)
            
            # Write code file
            (temp_path / "code.py").write_text(code)
            
            # Build image
            image, _ = self.docker_client.images.build(
                path=str(temp_path),
                tag=f"hephaestus-sandbox-{uuid.uuid4().hex[:8]}",
                rm=True
            )
            
            # Create container with limits
            container = self.docker_client.containers.create(
                image.id,
                mem_limit=f"{self.policy.max_memory_mb}m",
                memswap_limit=f"{self.policy.max_memory_mb}m",
                cpu_quota=int(self.policy.max_cpu_percent * 1000),
                cpu_period=100000,
                network_disabled=not self.policy.allow_network_access,
                read_only=True,
                tmpfs={'/tmp': 'noexec,nosuid,size=10m'},
                security_opt=['no-new-privileges'],
                cap_drop=['ALL']
            )
            
            return container
    
    async def _run_container(self, container) -> str:
        """Run container and return output."""
        # Start container
        container.start()
        
        # Wait for completion with timeout
        try:
            exit_code = container.wait(timeout=self.policy.max_execution_time)
            
            # Get output
            output = container.logs(stdout=True, stderr=True).decode('utf-8')
            
            if exit_code['StatusCode'] != 0:
                raise Exception(f"Container exited with code {exit_code['StatusCode']}: {output}")
            
            return output
            
        except Exception as e:
            # Kill container if still running
            try:
                container.kill()
            except Exception:
                pass
            raise e


class EnhancedSandbox:
    """Enhanced multi-layer sandbox system."""
    
    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
        preferred_type: SandboxType = SandboxType.RESTRICTED_PYTHON
    ):
        self.security_level = security_level
        self.preferred_type = preferred_type
        self.policy = self._create_security_policy()
        
        # Initialize available sandbox implementations
        self.sandboxes = {
            SandboxType.RESTRICTED_PYTHON: RestrictedPythonSandbox(self.policy),
            SandboxType.SUBPROCESS: SubprocessSandbox(self.policy),
        }
        
        # Initialize Docker sandbox if available
        try:
            docker_sandbox = DockerSandbox(self.policy)
            if docker_sandbox.docker_client:
                self.sandboxes[SandboxType.DOCKER] = docker_sandbox
        except Exception:
            pass
    
    def _create_security_policy(self) -> SecurityPolicy:
        """Create security policy based on security level."""
        if self.security_level == SecurityLevel.MINIMAL:
            return SecurityPolicy(
                allowed_imports={'math', 'random', 'datetime', 'json'},
                max_execution_time=60,
                max_memory_mb=512,
                max_cpu_percent=80.0,
                enable_ast_validation=False,
                enable_bytecode_analysis=False
            )
        
        elif self.security_level == SecurityLevel.STANDARD:
            return SecurityPolicy(
                allowed_imports={'math', 'random', 'datetime', 'json', 're'},
                blocked_imports={'os', 'sys', 'subprocess', 'socket'},
                blocked_builtins={'eval', 'exec', 'compile', '__import__'},
                max_execution_time=30,
                max_memory_mb=256,
                max_cpu_percent=50.0,
                enable_ast_validation=True
            )
        
        elif self.security_level == SecurityLevel.HIGH:
            return SecurityPolicy(
                allowed_imports={'math', 'datetime', 'json'},
                blocked_imports={'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests'},
                blocked_builtins={'eval', 'exec', 'compile', '__import__', 'open', 'input'},
                max_execution_time=15,
                max_memory_mb=128,
                max_cpu_percent=25.0,
                max_subprocess_count=0,
                enable_ast_validation=True,
                enable_bytecode_analysis=True,
                enable_runtime_monitoring=True
            )
        
        else:  # MAXIMUM
            return SecurityPolicy(
                allowed_imports={'math'},
                blocked_imports=set(__builtins__.keys()) - {'math'},
                blocked_builtins={'eval', 'exec', 'compile', '__import__', 'open', 'input', 'print'},
                max_execution_time=10,
                max_memory_mb=64,
                max_cpu_percent=10.0,
                max_subprocess_count=0,
                allow_file_access=False,
                allow_network_access=False,
                enable_ast_validation=True,
                enable_bytecode_analysis=True,
                enable_runtime_monitoring=True
            )
    
    async def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        sandbox_type: Optional[SandboxType] = None
    ) -> ExecutionResult:
        """Execute code with enhanced security."""
        target_type = sandbox_type or self.preferred_type
        
        # Try preferred sandbox first
        if target_type in self.sandboxes:
            try:
                result = await self.sandboxes[target_type].execute(code, context)
                result.metadata['sandbox_type'] = target_type.value
                return result
            except Exception as e:
                # Log error and try fallback
                print(f"Primary sandbox {target_type} failed: {e}")
        
        # Try fallback sandboxes
        fallback_order = [
            SandboxType.RESTRICTED_PYTHON,
            SandboxType.SUBPROCESS,
            SandboxType.DOCKER
        ]
        
        for fallback_type in fallback_order:
            if fallback_type != target_type and fallback_type in self.sandboxes:
                try:
                    result = await self.sandboxes[fallback_type].execute(code, context)
                    result.metadata['sandbox_type'] = fallback_type.value
                    result.metadata['fallback_used'] = True
                    return result
                except Exception as e:
                    print(f"Fallback sandbox {fallback_type} failed: {e}")
        
        # All sandboxes failed
        return ExecutionResult(
            success=False,
            error_message="All sandbox implementations failed",
            metadata={'sandbox_type': 'none'}
        )
    
    def get_available_sandboxes(self) -> List[SandboxType]:
        """Get list of available sandbox types."""
        return list(self.sandboxes.keys())
    
    def update_policy(self, **policy_updates):
        """Update security policy."""
        for key, value in policy_updates.items():
            if hasattr(self.policy, key):
                setattr(self.policy, key, value)
        
        # Update all sandbox instances
        for sandbox in self.sandboxes.values():
            sandbox.policy = self.policy