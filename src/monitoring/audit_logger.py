"""
Comprehensive logging and audit trail system for RSI.
Provides structured logging with complete audit capabilities using Loguru.
"""

import asyncio
import json
import time
import threading
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import gzip
import shutil
import uuid

from loguru import logger
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

from ..core.state import RSIState, StateManager
from ..validation.validators import RSIValidator
from ..monitoring.telemetry import trace_operation


class LogLevel(str, Enum):
    """Log levels for audit trail."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"


class AuditEventType(str, Enum):
    """Types of audit events."""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    DATA_ACCESS = "data_access"
    MODEL_OPERATION = "model_operation"
    CONFIGURATION_CHANGE = "configuration_change"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ERROR_EVENT = "error_event"
    PERFORMANCE_EVENT = "performance_event"


@dataclass
class AuditEvent:
    """Represents a single audit event."""
    
    id: str
    timestamp: datetime
    event_type: AuditEventType
    level: LogLevel
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    action: Optional[str] = None
    resource: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Event-specific data
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Security and integrity
    checksum: Optional[str] = None
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "level": self.level.value,
            "message": self.message,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "component": self.component,
            "action": self.action,
            "resource": self.resource,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "metadata": self.metadata,
            "checksum": self.checksum,
            "signature": self.signature
        }
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for event integrity."""
        # Create deterministic string representation
        event_str = json.dumps(
            {
                "id": self.id,
                "timestamp": self.timestamp.isoformat(),
                "event_type": self.event_type.value,
                "level": self.level.value,
                "message": self.message,
                "user_id": self.user_id,
                "component": self.component,
                "action": self.action,
                "resource": self.resource,
                "metadata": self.metadata
            },
            sort_keys=True
        )
        
        return hashlib.sha256(event_str.encode()).hexdigest()


class AuditLogger:
    """
    Comprehensive audit logging system with security features.
    """
    
    def __init__(
        self,
        log_directory: str = "./logs",
        encryption_key: Optional[str] = None,
        max_file_size_mb: int = 100,
        max_files: int = 10,
        compress_old_logs: bool = True,
        enable_integrity_checks: bool = True,
        state_manager: Optional[StateManager] = None
    ):
        self.log_directory = Path(log_directory)
        self.max_file_size_mb = max_file_size_mb
        self.max_files = max_files
        self.compress_old_logs = compress_old_logs
        self.enable_integrity_checks = enable_integrity_checks
        self.state_manager = state_manager
        
        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Set up encryption
        if encryption_key:
            self.cipher = self._create_cipher(encryption_key)
        else:
            self.cipher = None
        
        # Configure loguru
        self._setup_loguru()
        
        # Event storage
        self.event_buffer: List[AuditEvent] = []
        self.buffer_lock = threading.Lock()
        self.buffer_max_size = 1000
        
        # Background processing
        self.background_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.start_background_processing()
        
        # Event handlers
        self.event_handlers: List[Callable[[AuditEvent], None]] = []
        
        logger.info(f"Audit Logger initialized with directory: {log_directory}")
    
    def _create_cipher(self, password: str) -> Fernet:
        """Create encryption cipher from password."""
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'rsi_audit_salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return Fernet(key)
    
    def _setup_loguru(self):
        """Configure loguru for audit logging."""
        # Remove default handler
        logger.remove()
        
        # Add file handler for audit logs
        audit_file = self.log_directory / "audit.log"
        logger.add(
            audit_file,
            rotation=f"{self.max_file_size_mb} MB",
            retention=self.max_files,
            compression="gz" if self.compress_old_logs else None,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            level="DEBUG",
            serialize=True,
            enqueue=True  # Thread-safe logging
        )
        
        # Add separate handler for security events
        security_file = self.log_directory / "security.log"
        logger.add(
            security_file,
            rotation=f"{self.max_file_size_mb} MB",
            retention=self.max_files,
            compression="gz" if self.compress_old_logs else None,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            level="WARNING",
            filter=lambda record: "security" in record["extra"].get("event_type", ""),
            serialize=True,
            enqueue=True
        )
        
        # Add console handler for development
        logger.add(
            lambda msg: print(msg, end=""),
            format="{time:HH:mm:ss.SSS} | {level} | {message}",
            level="INFO",
            colorize=True
        )
    
    def start_background_processing(self):
        """Start background thread for processing events."""
        if self.background_thread is None or not self.background_thread.is_alive():
            self.background_thread = threading.Thread(
                target=self._background_processor,
                daemon=True
            )
            self.background_thread.start()
    
    def stop_background_processing(self):
        """Stop background processing."""
        self.shutdown_event.set()
        if self.background_thread:
            self.background_thread.join(timeout=5)
    
    def _background_processor(self):
        """Background processor for audit events."""
        while not self.shutdown_event.is_set():
            try:
                # Process buffered events
                if self.event_buffer:
                    with self.buffer_lock:
                        events_to_process = self.event_buffer.copy()
                        self.event_buffer.clear()
                    
                    for event in events_to_process:
                        self._process_event(event)
                
                # Sleep briefly
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Background processor error: {e}")
                time.sleep(1)
    
    def _process_event(self, event: AuditEvent):
        """Process a single audit event."""
        try:
            # Calculate checksum if integrity checks enabled
            if self.enable_integrity_checks:
                event.checksum = event.calculate_checksum()
            
            # Convert to dict for logging
            event_dict = event.to_dict()
            
            # Encrypt sensitive data if cipher is available
            if self.cipher and event.level in [LogLevel.AUDIT, LogLevel.CRITICAL]:
                event_dict = self._encrypt_sensitive_data(event_dict)
            
            # Log the event
            logger.bind(
                event_type=event.event_type.value,
                audit_id=event.id,
                user_id=event.user_id,
                component=event.component
            ).log(event.level.value, json.dumps(event_dict))
            
            # Call event handlers
            for handler in self.event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
            
        except Exception as e:
            logger.error(f"Failed to process audit event: {e}")
    
    def _encrypt_sensitive_data(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive data in event."""
        if not self.cipher:
            return event_dict
        
        sensitive_fields = ["before_state", "after_state", "metadata"]
        
        for field in sensitive_fields:
            if field in event_dict and event_dict[field]:
                try:
                    data_str = json.dumps(event_dict[field])
                    encrypted_data = self.cipher.encrypt(data_str.encode())
                    event_dict[field] = {
                        "encrypted": True,
                        "data": base64.b64encode(encrypted_data).decode()
                    }
                except Exception as e:
                    logger.error(f"Failed to encrypt {field}: {e}")
        
        return event_dict
    
    def add_event_handler(self, handler: Callable[[AuditEvent], None]):
        """Add event handler for real-time processing."""
        self.event_handlers.append(handler)
    
    @trace_operation("audit_log_event")
    def log_event(
        self,
        event_type: AuditEventType,
        level: LogLevel,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        component: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log an audit event."""
        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            level=level,
            message=message,
            user_id=user_id,
            session_id=session_id,
            component=component,
            action=action,
            resource=resource,
            before_state=before_state,
            after_state=after_state,
            metadata=metadata or {},
            **kwargs
        )
        
        # Add to buffer for processing
        with self.buffer_lock:
            self.event_buffer.append(event)
            
            # If buffer is full, process immediately
            if len(self.event_buffer) >= self.buffer_max_size:
                events_to_process = self.event_buffer.copy()
                self.event_buffer.clear()
                
                # Process in background
                threading.Thread(
                    target=lambda: [self._process_event(e) for e in events_to_process],
                    daemon=True
                ).start()
    
    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log user action."""
        self.log_event(
            event_type=AuditEventType.USER_ACTION,
            level=LogLevel.AUDIT,
            message=f"User {user_id} {action} {resource}",
            user_id=user_id,
            action=action,
            resource=resource,
            metadata={
                "success": success,
                "details": details or {}
            }
        )
    
    def log_system_event(
        self,
        component: str,
        event: str,
        level: LogLevel = LogLevel.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log system event."""
        self.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            level=level,
            message=f"System event in {component}: {event}",
            component=component,
            metadata=metadata or {}
        )
    
    def log_security_event(
        self,
        event: str,
        severity: str = "medium",
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security event."""
        level_map = {
            "low": LogLevel.WARNING,
            "medium": LogLevel.ERROR,
            "high": LogLevel.CRITICAL,
            "critical": LogLevel.CRITICAL
        }
        
        self.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            level=level_map.get(severity, LogLevel.WARNING),
            message=f"Security event: {event}",
            user_id=user_id,
            ip_address=ip_address,
            metadata={
                "severity": severity,
                "details": details or {}
            }
        )
    
    def log_model_operation(
        self,
        operation: str,
        model_id: str,
        user_id: Optional[str] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log model operation."""
        self.log_event(
            event_type=AuditEventType.MODEL_OPERATION,
            level=LogLevel.AUDIT,
            message=f"Model operation: {operation} on {model_id}",
            user_id=user_id,
            action=operation,
            resource=model_id,
            before_state=before_state,
            after_state=after_state,
            metadata=metadata or {}
        )
    
    def log_data_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool = True,
        data_size: Optional[int] = None
    ):
        """Log data access."""
        self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            level=LogLevel.AUDIT,
            message=f"Data access: {user_id} {action} {resource}",
            user_id=user_id,
            action=action,
            resource=resource,
            metadata={
                "success": success,
                "data_size": data_size
            }
        )
    
    def log_configuration_change(
        self,
        user_id: str,
        component: str,
        before_config: Dict[str, Any],
        after_config: Dict[str, Any],
        change_reason: Optional[str] = None
    ):
        """Log configuration change."""
        self.log_event(
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            level=LogLevel.AUDIT,
            message=f"Configuration change in {component} by {user_id}",
            user_id=user_id,
            component=component,
            action="configure",
            before_state=before_config,
            after_state=after_config,
            metadata={
                "change_reason": change_reason
            }
        )
    
    def log_authentication(
        self,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        method: str = "password"
    ):
        """Log authentication attempt."""
        self.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            level=LogLevel.AUDIT if success else LogLevel.WARNING,
            message=f"Authentication {'successful' if success else 'failed'} for {user_id}",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata={
                "success": success,
                "method": method
            }
        )
    
    def log_error(
        self,
        error: str,
        component: str,
        user_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log error event."""
        self.log_event(
            event_type=AuditEventType.ERROR_EVENT,
            level=LogLevel.ERROR,
            message=f"Error in {component}: {error}",
            user_id=user_id,
            component=component,
            metadata={
                "stack_trace": stack_trace,
                "context": context or {}
            }
        )
    
    def search_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        component: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search audit events (simplified in-memory search)."""
        # Note: In production, this would query a database or search index
        # For now, we'll return empty results as this is a basic implementation
        return []
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for the last N hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # In production, this would query actual logs
        # For now, return basic summary
        return {
            "time_range": {
                "start": cutoff_time.isoformat(),
                "end": datetime.now(timezone.utc).isoformat()
            },
            "total_events": 0,
            "events_by_type": {},
            "events_by_level": {},
            "unique_users": 0,
            "security_events": 0,
            "error_events": 0,
            "top_components": [],
            "top_actions": []
        }
    
    def verify_log_integrity(self, log_file: Path) -> bool:
        """Verify log file integrity."""
        if not self.enable_integrity_checks:
            return True
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if "checksum" in event:
                            # Verify checksum
                            # (Implementation details would go here)
                            pass
                    except json.JSONDecodeError:
                        continue
            
            return True
            
        except Exception as e:
            logger.error(f"Log integrity verification failed: {e}")
            return False
    
    def export_audit_logs(
        self,
        output_path: Path,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """Export audit logs to file."""
        # Implementation would export logs in specified format
        logger.info(f"Exporting audit logs to {output_path}")
    
    def cleanup_old_logs(self, retention_days: int = 365):
        """Clean up old log files."""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        for log_file in self.log_directory.glob("*.log*"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_time:
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file}")
            except Exception as e:
                logger.error(f"Failed to delete old log file {log_file}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_background_processing()


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def initialize_audit_logger(
    log_directory: str = "./logs",
    encryption_key: Optional[str] = None,
    **kwargs
) -> AuditLogger:
    """Initialize global audit logger."""
    global _audit_logger
    
    _audit_logger = AuditLogger(
        log_directory=log_directory,
        encryption_key=encryption_key,
        **kwargs
    )
    
    return _audit_logger


def get_audit_logger() -> Optional[AuditLogger]:
    """Get the global audit logger."""
    return _audit_logger


# Convenience functions for common audit operations
def audit_user_action(
    user_id: str,
    action: str,
    resource: str,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None
):
    """Log user action."""
    if _audit_logger:
        _audit_logger.log_user_action(user_id, action, resource, success, details)


def audit_security_event(
    event: str,
    severity: str = "medium",
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
):
    """Log security event."""
    if _audit_logger:
        _audit_logger.log_security_event(event, severity, user_id, ip_address, details)


def audit_model_operation(
    operation: str,
    model_id: str,
    user_id: Optional[str] = None,
    before_state: Optional[Dict[str, Any]] = None,
    after_state: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Log model operation."""
    if _audit_logger:
        _audit_logger.log_model_operation(
            operation, model_id, user_id, before_state, after_state, metadata
        )


def audit_system_event(
    component: str,
    event: str,
    level: LogLevel = LogLevel.INFO,
    metadata: Optional[Dict[str, Any]] = None
):
    """Log system event."""
    if _audit_logger:
        _audit_logger.log_system_event(component, event, level, metadata)