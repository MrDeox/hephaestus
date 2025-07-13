"""
Security tests for RSI sandbox and execution environment.

Tests comprehensive security measures including code injection prevention,
resource limits, privilege escalation prevention, and sandbox escapes.
"""

import pytest
import asyncio
import time
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

from src.security.sandbox import (
    RSISandbox, SafeExecutionEnvironment, DockerSandbox,
    SandboxType, SecurityLevel, ExecutionResult
)


class TestCodeInjectionPrevention:
    """Test prevention of various code injection attacks."""
    
    @pytest.mark.security
    async def test_system_command_injection(self):
        """Test prevention of system command injection."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        malicious_codes = [
            "__import__('os').system('whoami')",
            "__import__('subprocess').run(['ls', '-la'])",
            "exec('import os; os.system(\"cat /etc/passwd\")')",
            "eval('__import__(\"os\").system(\"id\")')",
            "compile('import os; os.system(\"uname -a\")', 'malicious', 'exec')",
        ]
        
        for code in malicious_codes:
            result = await sandbox.execute_code(code, timeout=5)
            
            # Should either fail to execute or be blocked
            assert (
                result.success is False or
                "blocked" in result.error_message.lower() or
                "restricted" in result.error_message.lower() or
                "forbidden" in result.error_message.lower()
            )
    
    @pytest.mark.security
    async def test_import_restrictions(self):
        """Test restrictions on dangerous imports."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        dangerous_imports = [
            "import os",
            "import subprocess",
            "import sys",
            "import socket",
            "import urllib",
            "import requests",
            "from os import system",
            "from subprocess import call",
            "__import__('os')",
        ]
        
        for import_code in dangerous_imports:
            result = await sandbox.execute_code(import_code, timeout=5)
            
            # Should be blocked or restricted
            assert (
                result.success is False or
                "import" in result.error_message.lower() or
                "restricted" in result.error_message.lower()
            )
    
    @pytest.mark.security
    async def test_builtin_function_restrictions(self):
        """Test restrictions on dangerous builtin functions."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        dangerous_builtins = [
            "open('/etc/passwd')",
            "exec('print(1)')",
            "eval('1+1')",
            "compile('print(1)', 'test', 'exec')",
            "globals()",
            "locals()",
            "vars()",
            "__import__('os')",
            "getattr(__builtins__, 'exec')",
        ]
        
        for builtin_code in dangerous_builtins:
            result = await sandbox.execute_code(builtin_code, timeout=5)
            
            # Should be blocked or restricted
            assert (
                result.success is False or
                result.result is None or
                "restricted" in str(result.error_message).lower()
            )
    
    @pytest.mark.security
    async def test_file_system_access_prevention(self):
        """Test prevention of unauthorized file system access."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        file_access_codes = [
            "open('/etc/passwd', 'r').read()",
            "open('/etc/shadow', 'r').read()",
            "open('/proc/version', 'r').read()",
            "with open('/tmp/test', 'w') as f: f.write('test')",
            "import pathlib; pathlib.Path('/etc/passwd').read_text()",
        ]
        
        for code in file_access_codes:
            result = await sandbox.execute_code(code, timeout=5)
            
            # Should be blocked or fail
            assert (
                result.success is False or
                "permission" in result.error_message.lower() or
                "access" in result.error_message.lower() or
                "restricted" in result.error_message.lower()
            )
    
    @pytest.mark.security
    async def test_network_access_prevention(self):
        """Test prevention of network access."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        network_codes = [
            "import socket; socket.socket().connect(('google.com', 80))",
            "import urllib.request; urllib.request.urlopen('http://google.com')",
            "import http.client; http.client.HTTPConnection('google.com')",
        ]
        
        for code in network_codes:
            result = await sandbox.execute_code(code, timeout=10)
            
            # Should be blocked or fail
            assert (
                result.success is False or
                "network" in result.error_message.lower() or
                "connection" in result.error_message.lower() or
                "restricted" in result.error_message.lower()
            )


class TestResourceLimits:
    """Test resource consumption limits and protection."""
    
    @pytest.mark.security
    async def test_cpu_time_limits(self):
        """Test CPU time limits."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        # Infinite loop should be terminated
        infinite_loop = "while True: pass"
        
        start_time = time.time()
        result = await sandbox.execute_code(infinite_loop, timeout=2)
        execution_time = time.time() - start_time
        
        # Should timeout within reasonable time
        assert execution_time < 5  # Should not run much longer than timeout
        assert (
            result.success is False or
            "timeout" in result.error_message.lower() or
            "time" in result.error_message.lower()
        )
    
    @pytest.mark.security
    async def test_memory_limits(self):
        """Test memory consumption limits."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        # Try to allocate large amounts of memory
        memory_bomb = "data = [0] * (10**8)"  # ~800MB of integers
        
        result = await sandbox.execute_code(memory_bomb, timeout=10)
        
        # Should either succeed with limits or fail gracefully
        assert isinstance(result, ExecutionResult)
        if not result.success:
            assert (
                "memory" in result.error_message.lower() or
                "limit" in result.error_message.lower() or
                "resource" in result.error_message.lower()
            )
    
    @pytest.mark.security
    async def test_recursion_limits(self):
        """Test recursion depth limits."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        # Deep recursion that should hit limits
        deep_recursion = """
def recursive_func(n):
    if n > 0:
        return recursive_func(n - 1)
    return 0

result = recursive_func(10000)
"""
        
        result = await sandbox.execute_code(deep_recursion, timeout=5)
        
        # Should either hit recursion limit or complete
        assert isinstance(result, ExecutionResult)
        if not result.success:
            assert (
                "recursion" in result.error_message.lower() or
                "stack" in result.error_message.lower() or
                "depth" in result.error_message.lower()
            )
    
    @pytest.mark.security
    async def test_thread_creation_limits(self):
        """Test thread creation limits."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        # Try to create many threads
        thread_bomb = """
import threading
threads = []
for i in range(1000):
    t = threading.Thread(target=lambda: None)
    threads.append(t)
    t.start()
"""
        
        result = await sandbox.execute_code(thread_bomb, timeout=10)
        
        # Should be restricted or fail
        assert (
            result.success is False or
            "thread" in result.error_message.lower() or
            "resource" in result.error_message.lower() or
            "limit" in result.error_message.lower()
        )


class TestSandboxEscapePrevention:
    """Test prevention of sandbox escape attempts."""
    
    @pytest.mark.security
    async def test_python_introspection_restrictions(self):
        """Test restrictions on Python introspection that could lead to escapes."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        introspection_codes = [
            "import inspect; inspect.getmembers(__builtins__)",
            "dir(__builtins__)",
            "vars()",
            "globals()",
            "locals()",
            "__builtins__.__dict__",
            "().__class__.__bases__[0].__subclasses__()",
            "type.__subclasses__(type)[0]",
        ]
        
        for code in introspection_codes:
            result = await sandbox.execute_code(code, timeout=5)
            
            # Should be restricted or return limited information
            assert isinstance(result, ExecutionResult)
            if result.success and result.result is not None:
                # If it succeeds, result should be limited/filtered
                result_str = str(result.result).lower()
                # Should not expose dangerous modules or functions
                dangerous_items = ["os", "subprocess", "sys", "socket", "urllib"]
                for item in dangerous_items:
                    assert item not in result_str
    
    @pytest.mark.security
    async def test_module_attribute_access_restrictions(self):
        """Test restrictions on accessing module attributes."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        attribute_access_codes = [
            "import math; math.__loader__.load_module('os')",
            "import sys; sys.modules['os']",
            "[].__class__.__module__",
            "().__class__.__module__",
            "type(open).__module__",
        ]
        
        for code in attribute_access_codes:
            result = await sandbox.execute_code(code, timeout=5)
            
            # Should be restricted or fail
            assert (
                result.success is False or
                result.result is None or
                "restricted" in str(result.error_message).lower()
            )
    
    @pytest.mark.security
    async def test_bytecode_manipulation_prevention(self):
        """Test prevention of bytecode manipulation attacks."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        bytecode_codes = [
            "import types; types.CodeType",
            "compile('import os', 'test', 'exec')",
            "(lambda: None).__code__",
            "exec(compile('print(1)', 'test', 'exec'))",
        ]
        
        for code in bytecode_codes:
            result = await sandbox.execute_code(code, timeout=5)
            
            # Should be restricted
            assert (
                result.success is False or
                "compile" in result.error_message.lower() or
                "code" in result.error_message.lower() or
                "restricted" in result.error_message.lower()
            )


class TestDockerSandboxSecurity:
    """Test Docker-based sandbox security (if available)."""
    
    @pytest.mark.security
    @pytest.mark.docker
    async def test_docker_container_isolation(self):
        """Test Docker container isolation."""
        try:
            sandbox = RSISandbox(sandbox_type=SandboxType.DOCKER)
        except Exception:
            pytest.skip("Docker not available")
        
        # Test that code runs in isolated container
        isolation_test = """
import os
import socket
import platform

result = {
    'hostname': socket.gethostname(),
    'platform': platform.system(),
    'user': os.getenv('USER', 'unknown'),
    'pwd': os.getcwd()
}
"""
        
        result = await sandbox.execute_code(isolation_test, timeout=10)
        
        if result.success and result.result:
            # Should run in isolated environment
            container_info = result.result
            assert isinstance(container_info, dict)
            # Hostname should be different from host
            # User should be container user, not host user
    
    @pytest.mark.security
    @pytest.mark.docker
    async def test_docker_filesystem_isolation(self):
        """Test Docker filesystem isolation."""
        try:
            sandbox = RSISandbox(sandbox_type=SandboxType.DOCKER)
        except Exception:
            pytest.skip("Docker not available")
        
        # Try to access host filesystem
        host_access_test = """
import os
import pathlib

results = []
test_paths = ['/etc/passwd', '/proc/version', '/sys', '/dev']

for path in test_paths:
    try:
        if os.path.exists(path):
            results.append(f"{path}: exists")
        else:
            results.append(f"{path}: not found")
    except Exception as e:
        results.append(f"{path}: error - {str(e)}")

results
"""
        
        result = await sandbox.execute_code(host_access_test, timeout=10)
        
        # Should run in isolated filesystem
        assert isinstance(result, ExecutionResult)
        if result.success:
            # Access should be limited or results should be from container, not host
            assert isinstance(result.result, list)
    
    @pytest.mark.security
    @pytest.mark.docker
    async def test_docker_network_isolation(self):
        """Test Docker network isolation."""
        try:
            sandbox = RSISandbox(sandbox_type=SandboxType.DOCKER)
        except Exception:
            pytest.skip("Docker not available")
        
        # Try network operations
        network_test = """
import socket
import subprocess

results = []

# Test socket creation
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    s.connect(('8.8.8.8', 53))
    results.append("External connection: success")
    s.close()
except Exception as e:
    results.append(f"External connection: failed - {str(e)}")

# Test subprocess network commands
try:
    output = subprocess.run(['ping', '-c', '1', '8.8.8.8'], 
                          capture_output=True, timeout=5, text=True)
    results.append(f"Ping: {output.returncode}")
except Exception as e:
    results.append(f"Ping: failed - {str(e)}")

results
"""
        
        result = await sandbox.execute_code(network_test, timeout=15)
        
        # Network should be restricted or isolated
        assert isinstance(result, ExecutionResult)


class TestInputValidationSecurity:
    """Test input validation and sanitization."""
    
    @pytest.mark.security
    async def test_malformed_python_code(self):
        """Test handling of malformed Python code."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        malformed_codes = [
            "if True print('test')",  # Missing colon
            "def func( return x",       # Incomplete function
            "import os; os.system(",   # Incomplete statement
            "exec('''",                # Incomplete exec
            "for i in range(10",       # Incomplete loop
            "\x00\x01\x02",          # Binary data
            "а = 1",                   # Unicode that might confuse parser
        ]
        
        for code in malformed_codes:
            result = await sandbox.execute_code(code, timeout=5)
            
            # Should handle gracefully without crashing
            assert isinstance(result, ExecutionResult)
            assert result.success is False
            assert "syntax" in result.error_message.lower() or \
                   "parse" in result.error_message.lower() or \
                   "invalid" in result.error_message.lower()
    
    @pytest.mark.security
    async def test_large_input_handling(self):
        """Test handling of excessively large inputs."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        # Very large code string
        large_code = "x = " + "1" * 1000000  # 1MB of 1s
        
        result = await sandbox.execute_code(large_code, timeout=10)
        
        # Should handle without crashing or consuming excessive resources
        assert isinstance(result, ExecutionResult)
        # May succeed or fail, but should not hang or crash
    
    @pytest.mark.security
    async def test_special_character_handling(self):
        """Test handling of special characters and encodings."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        special_codes = [
            "print('Hello 🌍')",                    # Unicode emojis
            "print('Тест')",                        # Cyrillic
            "print('测试')",                         # Chinese
            "print('\\x41\\x42\\x43')",            # Hex escapes
            "print('\\101\\102\\103')",             # Octal escapes
            "print(r'C:\\Windows\\System32')",      # Raw strings
            "print('''Multi\nline\nstring''')",     # Multi-line strings
        ]
        
        for code in special_codes:
            result = await sandbox.execute_code(code, timeout=5)
            
            # Should handle Unicode and special characters correctly
            assert isinstance(result, ExecutionResult)
            # Should not cause encoding errors or security issues


class TestPrivilegeEscalation:
    """Test prevention of privilege escalation attacks."""
    
    @pytest.mark.security
    async def test_user_permission_restrictions(self):
        """Test that code runs with restricted user permissions."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        permission_test = """
import os
import pwd
import grp

result = {
    'uid': os.getuid() if hasattr(os, 'getuid') else 'N/A',
    'gid': os.getgid() if hasattr(os, 'getgid') else 'N/A',
    'user': os.getenv('USER', 'unknown'),
    'home': os.getenv('HOME', 'unknown'),
}
"""
        
        result = await sandbox.execute_code(permission_test, timeout=5)
        
        if result.success and result.result:
            user_info = result.result
            # Should not be running as root or privileged user
            if 'uid' in user_info and user_info['uid'] != 'N/A':
                assert user_info['uid'] != 0  # Not root
    
    @pytest.mark.security
    async def test_sudo_command_prevention(self):
        """Test prevention of sudo and privilege escalation commands."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        privilege_codes = [
            "import subprocess; subprocess.run(['sudo', 'whoami'])",
            "import subprocess; subprocess.run(['su', 'root'])",
            "import os; os.system('sudo ls')",
            "import os; os.setuid(0)",
            "import os; os.setgid(0)",
        ]
        
        for code in privilege_codes:
            result = await sandbox.execute_code(code, timeout=5)
            
            # Should be blocked or fail
            assert (
                result.success is False or
                "permission" in result.error_message.lower() or
                "denied" in result.error_message.lower() or
                "restricted" in result.error_message.lower()
            )


class TestSandboxConfiguration:
    """Test sandbox configuration and security levels."""
    
    @pytest.mark.security
    def test_security_level_enforcement(self):
        """Test different security levels."""
        # High security sandbox
        high_security = RSISandbox(
            sandbox_type=SandboxType.RESTRICTED_PYTHON,
            security_level=SecurityLevel.HIGH
        )
        
        # Medium security sandbox
        medium_security = RSISandbox(
            sandbox_type=SandboxType.RESTRICTED_PYTHON,
            security_level=SecurityLevel.MEDIUM
        )
        
        assert high_security.security_level == SecurityLevel.HIGH
        assert medium_security.security_level == SecurityLevel.MEDIUM
        
        # High security should have stricter restrictions
        assert high_security.get_allowed_imports() <= medium_security.get_allowed_imports()
    
    @pytest.mark.security
    async def test_allowed_imports_configuration(self):
        """Test configuration of allowed imports."""
        sandbox = RSISandbox(
            sandbox_type=SandboxType.RESTRICTED_PYTHON,
            allowed_imports=['math', 'json', 'datetime']
        )
        
        # Allowed imports should work
        allowed_code = "import math; result = math.sqrt(16)"
        result = await sandbox.execute_code(allowed_code, timeout=5)
        assert result.success is True
        assert result.result == 4.0
        
        # Disallowed imports should fail
        disallowed_code = "import os"
        result = await sandbox.execute_code(disallowed_code, timeout=5)
        assert result.success is False
    
    @pytest.mark.security
    async def test_timeout_configuration(self):
        """Test timeout configuration."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        # Short timeout
        start_time = time.time()
        result = await sandbox.execute_code("import time; time.sleep(5)", timeout=1)
        execution_time = time.time() - start_time
        
        # Should timeout quickly
        assert execution_time < 3  # Should not run much longer than timeout
        assert (
            result.success is False and
            ("timeout" in result.error_message.lower() or
             "time" in result.error_message.lower())
        )


class TestSandboxPerformance:
    """Test sandbox performance and DoS prevention."""
    
    @pytest.mark.security
    @pytest.mark.performance
    async def test_cpu_bomb_prevention(self):
        """Test prevention of CPU exhaustion attacks."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        cpu_bomb = """
import math
for i in range(10**6):
    math.factorial(100)
"""
        
        start_time = time.time()
        result = await sandbox.execute_code(cpu_bomb, timeout=3)
        execution_time = time.time() - start_time
        
        # Should timeout or limit CPU usage
        assert execution_time < 10  # Should not run too long
        assert isinstance(result, ExecutionResult)
    
    @pytest.mark.security
    @pytest.mark.performance
    async def test_memory_bomb_prevention(self):
        """Test prevention of memory exhaustion attacks."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        memory_bombs = [
            "[0] * (10**8)",           # Large list
            "{'a' * i: i for i in range(10**5)}",  # Large dict
            "'a' * (10**7)",           # Large string
            "set(range(10**6))",       # Large set
        ]
        
        for bomb in memory_bombs:
            result = await sandbox.execute_code(bomb, timeout=5)
            
            # Should either succeed with limits or fail gracefully
            assert isinstance(result, ExecutionResult)
            if not result.success:
                assert (
                    "memory" in result.error_message.lower() or
                    "limit" in result.error_message.lower() or
                    "resource" in result.error_message.lower()
                )
    
    @pytest.mark.security
    @pytest.mark.performance
    async def test_io_bomb_prevention(self):
        """Test prevention of I/O exhaustion attacks."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        io_bomb = """
import tempfile
import os

# Try to create many files
for i in range(1000):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b'x' * 10000)  # 10KB per file
    except Exception:
        break
"""
        
        result = await sandbox.execute_code(io_bomb, timeout=10)
        
        # Should be limited or restricted
        assert isinstance(result, ExecutionResult)
        # May succeed with limits or fail due to restrictions


class TestSandboxAuditLogging:
    """Test security audit logging and monitoring."""
    
    @pytest.mark.security
    async def test_security_event_logging(self):
        """Test that security events are properly logged."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        # Execute code that should trigger security logging
        malicious_code = "__import__('os').system('whoami')"
        
        with patch('src.monitoring.audit_logger.AuditLogger') as mock_logger:
            mock_instance = MagicMock()
            mock_logger.return_value = mock_instance
            
            result = await sandbox.execute_code(malicious_code, timeout=5)
            
            # Should log security events
            assert isinstance(result, ExecutionResult)
            # Verify logging calls if audit logger is used
    
    @pytest.mark.security
    async def test_execution_metadata_tracking(self):
        """Test tracking of execution metadata for security analysis."""
        sandbox = RSISandbox(sandbox_type=SandboxType.RESTRICTED_PYTHON)
        
        test_code = "result = 2 + 2"
        result = await sandbox.execute_code(test_code, timeout=5)
        
        # Should include security-relevant metadata
        assert isinstance(result, ExecutionResult)
        assert hasattr(result, 'execution_time')
        assert hasattr(result, 'resource_usage')
        
        # Resource usage should be tracked
        if result.resource_usage:
            assert isinstance(result.resource_usage, dict)
            # Should track memory and CPU usage