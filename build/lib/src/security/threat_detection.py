"""
Advanced Threat Detection System for Hephaestus RSI.

Implements behavioral analysis, anomaly detection, and real-time
threat monitoring with automated response capabilities.
"""

import asyncio
import hashlib
import json
import re
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Pattern
import numpy as np

from ..common.exceptions import SecurityViolationError, create_error_context
from ..monitoring.anomaly_detection import BehavioralMonitor, AnomalyType
from config.base_config import get_config


class ThreatLevel(str, Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(str, Enum):
    """Types of security threats."""
    CODE_INJECTION = "code_injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_EXFILTRATION = "data_exfiltration"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_PAYLOAD = "malicious_payload"
    RECONNAISSANCE = "reconnaissance"
    LATERAL_MOVEMENT = "lateral_movement"
    PERSISTENCE = "persistence"
    EVASION = "evasion"


class ResponseAction(str, Enum):
    """Automated response actions."""
    LOG = "log"
    ALERT = "alert"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    SHUTDOWN = "shutdown"
    ROLLBACK = "rollback"


@dataclass
class ThreatIndicator:
    """Represents a threat indicator."""
    indicator_type: str
    pattern: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    description: str
    regex_pattern: Optional[Pattern] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compile regex pattern if provided."""
        if self.pattern and not self.regex_pattern:
            try:
                self.regex_pattern = re.compile(self.pattern, re.IGNORECASE)
            except re.error:
                pass


@dataclass
class ThreatEvent:
    """Represents a detected threat event."""
    event_id: str
    timestamp: datetime
    threat_type: ThreatType
    threat_level: ThreatLevel
    source: str
    description: str
    indicators: List[ThreatIndicator]
    context: Dict[str, Any] = field(default_factory=dict)
    response_actions: List[ResponseAction] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class ThreatSignatures:
    """Database of threat signatures and indicators."""
    
    def __init__(self):
        self.signatures = self._load_default_signatures()
    
    def _load_default_signatures(self) -> List[ThreatIndicator]:
        """Load default threat signatures."""
        signatures = []
        
        # Code injection patterns
        code_injection_patterns = [
            (r"__import__\s*\(\s*['\"]os['\"]", "Dynamic OS module import"),
            (r"exec\s*\(", "Dynamic code execution"),
            (r"eval\s*\(", "Dynamic code evaluation"),
            (r"compile\s*\(", "Dynamic code compilation"),
            (r"subprocess\.(run|call|Popen)", "Subprocess execution"),
            (r"os\.(system|popen|execv?)", "System command execution"),
            (r"pickle\.(loads?|dumps?)", "Dangerous serialization"),
            (r"marshal\.(loads?|dumps?)", "Dangerous marshal operations"),
        ]
        
        for pattern, description in code_injection_patterns:
            signatures.append(ThreatIndicator(
                indicator_type="code_pattern",
                pattern=pattern,
                threat_type=ThreatType.CODE_INJECTION,
                threat_level=ThreatLevel.HIGH,
                description=description
            ))
        
        # Privilege escalation patterns
        privilege_escalation_patterns = [
            (r"setuid\s*\(", "UID manipulation"),
            (r"setgid\s*\(", "GID manipulation"),
            (r"sudo\s+", "Sudo usage"),
            (r"su\s+", "User switching"),
            (r"chmod\s+.*\+[sx]", "SUID/SGID setting"),
            (r"/etc/passwd", "Password file access"),
            (r"/etc/shadow", "Shadow file access"),
        ]
        
        for pattern, description in privilege_escalation_patterns:
            signatures.append(ThreatIndicator(
                indicator_type="code_pattern",
                pattern=pattern,
                threat_type=ThreatType.PRIVILEGE_ESCALATION,
                threat_level=ThreatLevel.CRITICAL,
                description=description
            ))
        
        # Data exfiltration patterns
        data_exfiltration_patterns = [
            (r"socket\.socket", "Network socket creation"),
            (r"urllib\.request", "HTTP request"),
            (r"requests\.(get|post)", "HTTP library usage"),
            (r"ftp\.", "FTP operations"),
            (r"smtp\.", "Email operations"),
            (r"base64\.(encode|decode)", "Base64 encoding"),
        ]
        
        for pattern, description in data_exfiltration_patterns:
            signatures.append(ThreatIndicator(
                indicator_type="code_pattern",
                pattern=pattern,
                threat_type=ThreatType.DATA_EXFILTRATION,
                threat_level=ThreatLevel.MEDIUM,
                description=description
            ))
        
        # Resource exhaustion patterns
        resource_exhaustion_patterns = [
            (r"while\s+True\s*:", "Infinite loop"),
            (r"for\s+.*in\s+range\s*\(\s*\d{6,}", "Large range iteration"),
            (r"\[\s*0\s*\]\s*\*\s*\d{6,}", "Large list creation"),
            (r"threading\.Thread", "Thread creation"),
            (r"multiprocessing\.", "Process creation"),
        ]
        
        for pattern, description in resource_exhaustion_patterns:
            signatures.append(ThreatIndicator(
                indicator_type="code_pattern",
                pattern=pattern,
                threat_type=ThreatType.RESOURCE_EXHAUSTION,
                threat_level=ThreatLevel.MEDIUM,
                description=description
            ))
        
        # Reconnaissance patterns
        reconnaissance_patterns = [
            (r"platform\.", "System information gathering"),
            (r"sys\.version", "Python version detection"),
            (r"os\.environ", "Environment variable access"),
            (r"pwd\.getpwuid", "User information gathering"),
            (r"socket\.gethostname", "Hostname detection"),
            (r"psutil\.", "System monitoring"),
        ]
        
        for pattern, description in reconnaissance_patterns:
            signatures.append(ThreatIndicator(
                indicator_type="code_pattern",
                pattern=pattern,
                threat_type=ThreatType.RECONNAISSANCE,
                threat_level=ThreatLevel.LOW,
                description=description
            ))
        
        return signatures
    
    def add_signature(self, signature: ThreatIndicator) -> None:
        """Add a new threat signature."""
        self.signatures.append(signature)
    
    def get_signatures_by_type(self, threat_type: ThreatType) -> List[ThreatIndicator]:
        """Get signatures by threat type."""
        return [sig for sig in self.signatures if sig.threat_type == threat_type]
    
    def scan_content(self, content: str) -> List[ThreatIndicator]:
        """Scan content for threat indicators."""
        detected_indicators = []
        
        for signature in self.signatures:
            if signature.regex_pattern and signature.regex_pattern.search(content):
                detected_indicators.append(signature)
        
        return detected_indicators


class BehavioralAnalyzer:
    """Analyzes behavioral patterns for threat detection."""
    
    def __init__(self):
        self.execution_history = deque(maxlen=1000)
        self.access_patterns = defaultdict(list)
        self.frequency_counters = defaultdict(int)
        self.time_windows = defaultdict(lambda: deque(maxlen=100))
        
    def record_execution(
        self,
        source: str,
        code_hash: str,
        execution_time: float,
        resource_usage: Dict[str, Any],
        success: bool
    ) -> None:
        """Record code execution for behavioral analysis."""
        execution_record = {
            'timestamp': datetime.now(timezone.utc),
            'source': source,
            'code_hash': code_hash,
            'execution_time': execution_time,
            'resource_usage': resource_usage,
            'success': success
        }
        
        self.execution_history.append(execution_record)
        self.frequency_counters[code_hash] += 1
        self.time_windows[source].append(execution_record['timestamp'])
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies."""
        anomalies = []
        
        # Detect high-frequency execution
        anomalies.extend(self._detect_frequency_anomalies())
        
        # Detect unusual resource usage
        anomalies.extend(self._detect_resource_anomalies())
        
        # Detect time-based patterns
        anomalies.extend(self._detect_temporal_anomalies())
        
        # Detect code similarity patterns
        anomalies.extend(self._detect_similarity_anomalies())
        
        return anomalies
    
    def _detect_frequency_anomalies(self) -> List[Dict[str, Any]]:
        """Detect high-frequency execution anomalies."""
        anomalies = []
        
        # Check for codes executed too frequently
        for code_hash, count in self.frequency_counters.items():
            if count > 100:  # Threshold for suspicious frequency
                anomalies.append({
                    'type': 'high_frequency_execution',
                    'threat_type': ThreatType.RESOURCE_EXHAUSTION,
                    'threat_level': ThreatLevel.MEDIUM,
                    'description': f'Code executed {count} times',
                    'code_hash': code_hash
                })
        
        return anomalies
    
    def _detect_resource_anomalies(self) -> List[Dict[str, Any]]:
        """Detect unusual resource usage patterns."""
        anomalies = []
        
        if len(self.execution_history) < 10:
            return anomalies
        
        # Analyze memory usage patterns
        memory_usage = [
            record['resource_usage'].get('memory_mb', 0)
            for record in self.execution_history
            if record['resource_usage']
        ]
        
        if memory_usage:
            mean_memory = np.mean(memory_usage)
            std_memory = np.std(memory_usage)
            
            # Check latest executions for anomalies
            recent_executions = list(self.execution_history)[-5:]
            for record in recent_executions:
                memory = record['resource_usage'].get('memory_mb', 0)
                if memory > mean_memory + 3 * std_memory:
                    anomalies.append({
                        'type': 'memory_anomaly',
                        'threat_type': ThreatType.RESOURCE_EXHAUSTION,
                        'threat_level': ThreatLevel.HIGH,
                        'description': f'Unusual memory usage: {memory:.1f} MB',
                        'record': record
                    })
        
        return anomalies
    
    def _detect_temporal_anomalies(self) -> List[Dict[str, Any]]:
        """Detect time-based execution anomalies."""
        anomalies = []
        
        current_time = datetime.now(timezone.utc)
        
        for source, timestamps in self.time_windows.items():
            if len(timestamps) < 5:
                continue
            
            # Check for burst activity (many executions in short time)
            recent_window = current_time - timedelta(minutes=1)
            recent_count = sum(1 for ts in timestamps if ts >= recent_window)
            
            if recent_count > 20:  # Threshold for burst activity
                anomalies.append({
                    'type': 'burst_activity',
                    'threat_type': ThreatType.RESOURCE_EXHAUSTION,
                    'threat_level': ThreatLevel.MEDIUM,
                    'description': f'Burst activity: {recent_count} executions in 1 minute',
                    'source': source
                })
        
        return anomalies
    
    def _detect_similarity_anomalies(self) -> List[Dict[str, Any]]:
        """Detect code similarity anomalies."""
        anomalies = []
        
        # Group recent executions by similar code hashes
        recent_executions = list(self.execution_history)[-50:]
        hash_groups = defaultdict(list)
        
        for record in recent_executions:
            # Group by first 8 characters of hash (similarity detection)
            hash_prefix = record['code_hash'][:8]
            hash_groups[hash_prefix].append(record)
        
        # Detect groups with many similar executions
        for hash_prefix, records in hash_groups.items():
            if len(records) > 10:
                anomalies.append({
                    'type': 'similar_code_pattern',
                    'threat_type': ThreatType.PERSISTENCE,
                    'threat_level': ThreatLevel.LOW,
                    'description': f'Similar code executed {len(records)} times',
                    'hash_prefix': hash_prefix
                })
        
        return anomalies


class ThreatDetectionEngine:
    """Main threat detection engine."""
    
    def __init__(self):
        self.signatures = ThreatSignatures()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.threat_events: List[ThreatEvent] = []
        self.response_handlers: Dict[ThreatType, List[Callable]] = defaultdict(list)
        self.monitoring_enabled = True
        self.auto_response_enabled = True
        
        # Load configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """Load threat detection configuration."""
        try:
            config = get_config()
            
            if hasattr(config.security, 'threat_detection'):
                threat_config = config.security.threat_detection
                self.monitoring_enabled = getattr(threat_config, 'enabled', True)
                self.auto_response_enabled = getattr(threat_config, 'auto_response', True)
        except Exception:
            pass  # Use defaults
    
    async def scan_code(self, code: str, source: str = "unknown") -> List[ThreatEvent]:
        """Scan code for threats and return detected events."""
        threat_events = []
        
        if not self.monitoring_enabled:
            return threat_events
        
        # Signature-based detection
        detected_indicators = self.signatures.scan_content(code)
        
        if detected_indicators:
            # Create threat event
            event_id = self._generate_event_id()
            
            # Determine overall threat level
            max_level = max(
                (indicator.threat_level for indicator in detected_indicators),
                default=ThreatLevel.LOW,
                key=lambda x: ['low', 'medium', 'high', 'critical'].index(x.value)
            )
            
            # Determine primary threat type
            threat_types = [indicator.threat_type for indicator in detected_indicators]
            primary_threat_type = max(set(threat_types), key=threat_types.count)
            
            threat_event = ThreatEvent(
                event_id=event_id,
                timestamp=datetime.now(timezone.utc),
                threat_type=primary_threat_type,
                threat_level=max_level,
                source=source,
                description=f"Detected {len(detected_indicators)} threat indicators",
                indicators=detected_indicators,
                context={
                    'code_hash': hashlib.sha256(code.encode()).hexdigest(),
                    'code_length': len(code),
                    'indicator_count': len(detected_indicators)
                }
            )
            
            threat_events.append(threat_event)
            self.threat_events.append(threat_event)
            
            # Determine response actions
            response_actions = self._determine_response_actions(threat_event)
            threat_event.response_actions = response_actions
            
            # Execute automatic responses
            if self.auto_response_enabled:
                await self._execute_response_actions(threat_event)
        
        # Record execution for behavioral analysis
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        self.behavioral_analyzer.record_execution(
            source=source,
            code_hash=code_hash,
            execution_time=0.0,  # Will be updated after execution
            resource_usage={},   # Will be updated after execution
            success=len(threat_events) == 0  # Consider threats as failed executions
        )
        
        return threat_events
    
    async def analyze_execution_result(
        self,
        source: str,
        code: str,
        execution_result: Dict[str, Any]
    ) -> List[ThreatEvent]:
        """Analyze execution result for behavioral threats."""
        threat_events = []
        
        if not self.monitoring_enabled:
            return threat_events
        
        # Update behavioral analyzer
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        self.behavioral_analyzer.record_execution(
            source=source,
            code_hash=code_hash,
            execution_time=execution_result.get('execution_time', 0.0),
            resource_usage=execution_result.get('resource_usage', {}),
            success=execution_result.get('success', False)
        )
        
        # Detect behavioral anomalies
        anomalies = self.behavioral_analyzer.detect_anomalies()
        
        for anomaly in anomalies:
            event_id = self._generate_event_id()
            
            threat_event = ThreatEvent(
                event_id=event_id,
                timestamp=datetime.now(timezone.utc),
                threat_type=anomaly['threat_type'],
                threat_level=anomaly['threat_level'],
                source=source,
                description=anomaly['description'],
                indicators=[],
                context={
                    'anomaly_type': anomaly['type'],
                    'code_hash': code_hash,
                    **{k: v for k, v in anomaly.items() if k not in ['type', 'threat_type', 'threat_level', 'description']}
                }
            )
            
            threat_events.append(threat_event)
            self.threat_events.append(threat_event)
            
            # Determine and execute response actions
            response_actions = self._determine_response_actions(threat_event)
            threat_event.response_actions = response_actions
            
            if self.auto_response_enabled:
                await self._execute_response_actions(threat_event)
        
        return threat_events
    
    def _determine_response_actions(self, threat_event: ThreatEvent) -> List[ResponseAction]:
        """Determine appropriate response actions for a threat event."""
        actions = [ResponseAction.LOG]  # Always log
        
        # Determine actions based on threat level
        if threat_event.threat_level == ThreatLevel.LOW:
            actions.append(ResponseAction.ALERT)
        
        elif threat_event.threat_level == ThreatLevel.MEDIUM:
            actions.extend([ResponseAction.ALERT, ResponseAction.BLOCK])
        
        elif threat_event.threat_level == ThreatLevel.HIGH:
            actions.extend([ResponseAction.ALERT, ResponseAction.BLOCK, ResponseAction.QUARANTINE])
        
        elif threat_event.threat_level == ThreatLevel.CRITICAL:
            actions.extend([
                ResponseAction.ALERT,
                ResponseAction.BLOCK,
                ResponseAction.QUARANTINE,
                ResponseAction.SHUTDOWN
            ])
        
        # Specific actions based on threat type
        if threat_event.threat_type == ThreatType.CODE_INJECTION:
            actions.append(ResponseAction.BLOCK)
        
        elif threat_event.threat_type == ThreatType.PRIVILEGE_ESCALATION:
            actions.extend([ResponseAction.BLOCK, ResponseAction.SHUTDOWN])
        
        elif threat_event.threat_type == ThreatType.DATA_EXFILTRATION:
            actions.extend([ResponseAction.BLOCK, ResponseAction.QUARANTINE])
        
        return list(set(actions))  # Remove duplicates
    
    async def _execute_response_actions(self, threat_event: ThreatEvent) -> None:
        """Execute response actions for a threat event."""
        for action in threat_event.response_actions:
            try:
                if action == ResponseAction.LOG:
                    self._log_threat_event(threat_event)
                
                elif action == ResponseAction.ALERT:
                    await self._send_alert(threat_event)
                
                elif action == ResponseAction.BLOCK:
                    await self._block_execution(threat_event)
                
                elif action == ResponseAction.QUARANTINE:
                    await self._quarantine_source(threat_event)
                
                elif action == ResponseAction.SHUTDOWN:
                    await self._emergency_shutdown(threat_event)
                
                elif action == ResponseAction.ROLLBACK:
                    await self._rollback_changes(threat_event)
                
            except Exception as e:
                print(f"Error executing response action {action}: {e}")
    
    def _log_threat_event(self, threat_event: ThreatEvent) -> None:
        """Log threat event."""
        log_entry = {
            'event_id': threat_event.event_id,
            'timestamp': threat_event.timestamp.isoformat(),
            'threat_type': threat_event.threat_type.value,
            'threat_level': threat_event.threat_level.value,
            'source': threat_event.source,
            'description': threat_event.description,
            'indicators': [
                {
                    'type': ind.indicator_type,
                    'pattern': ind.pattern,
                    'description': ind.description
                }
                for ind in threat_event.indicators
            ],
            'context': threat_event.context
        }
        
        print(f"[THREAT DETECTED] {json.dumps(log_entry, indent=2)}")
    
    async def _send_alert(self, threat_event: ThreatEvent) -> None:
        """Send threat alert."""
        # Could integrate with alerting systems (email, Slack, etc.)
        alert_message = (
            f"SECURITY ALERT: {threat_event.threat_type.value.upper()} "
            f"({threat_event.threat_level.value.upper()}) detected from {threat_event.source}"
        )
        print(f"[ALERT] {alert_message}")
    
    async def _block_execution(self, threat_event: ThreatEvent) -> None:
        """Block execution from threat source."""
        # Could implement source blocking logic
        print(f"[BLOCKED] Execution blocked for source: {threat_event.source}")
    
    async def _quarantine_source(self, threat_event: ThreatEvent) -> None:
        """Quarantine threat source."""
        # Could implement quarantine logic
        print(f"[QUARANTINED] Source quarantined: {threat_event.source}")
    
    async def _emergency_shutdown(self, threat_event: ThreatEvent) -> None:
        """Perform emergency shutdown for critical threats."""
        print(f"[EMERGENCY SHUTDOWN] Critical threat detected: {threat_event.event_id}")
        # Could trigger system shutdown or circuit breaker
    
    async def _rollback_changes(self, threat_event: ThreatEvent) -> None:
        """Rollback changes made by threat."""
        # Could implement state rollback logic
        print(f"[ROLLBACK] Rolling back changes from: {threat_event.source}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def register_response_handler(
        self,
        threat_type: ThreatType,
        handler: Callable[[ThreatEvent], None]
    ) -> None:
        """Register custom response handler for threat type."""
        self.response_handlers[threat_type].append(handler)
    
    def get_threat_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get threat detection statistics."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.threat_events
            if event.timestamp >= cutoff_time
        ]
        
        # Count by threat type
        threat_type_counts = defaultdict(int)
        for event in recent_events:
            threat_type_counts[event.threat_type.value] += 1
        
        # Count by threat level
        threat_level_counts = defaultdict(int)
        for event in recent_events:
            threat_level_counts[event.threat_level.value] += 1
        
        return {
            'time_period_hours': hours,
            'total_threats': len(recent_events),
            'by_type': dict(threat_type_counts),
            'by_level': dict(threat_level_counts),
            'resolved_count': sum(1 for event in recent_events if event.resolved),
            'active_count': sum(1 for event in recent_events if not event.resolved)
        }


# Global threat detection instance
_threat_detector: Optional[ThreatDetectionEngine] = None


def get_threat_detector() -> ThreatDetectionEngine:
    """Get global threat detector instance."""
    global _threat_detector
    if _threat_detector is None:
        _threat_detector = ThreatDetectionEngine()
    return _threat_detector


def set_threat_detector(detector: ThreatDetectionEngine) -> None:
    """Set global threat detector instance."""
    global _threat_detector
    _threat_detector = detector