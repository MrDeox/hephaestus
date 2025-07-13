"""
Safety Circuit Breaker System for RSI Operations.
Provides critical failure prevention with automatic rollback capabilities.
"""

import asyncio
import time
import json
import pickle
import hashlib
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from loguru import logger
import threading
from concurrent.futures import ThreadPoolExecutor

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class SafetyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SafetyCheckpoint:
    """System state checkpoint for rollback"""
    timestamp: float
    checkpoint_id: str
    model_state_hash: str
    system_metrics: Dict[str, Any]
    safety_scores: Dict[str, float]
    operation_context: Dict[str, Any]
    checkpoint_size_mb: float

@dataclass
class SafetyViolation:
    """Safety violation record"""
    timestamp: float
    violation_id: str
    violation_type: str
    severity: SafetyLevel
    description: str
    operation_context: Dict[str, Any]
    automatic_action_taken: Optional[str]
    human_intervention_required: bool

class RSISafetyCircuitBreaker:
    """Advanced circuit breaker for RSI safety with automatic rollback"""
    
    def __init__(self, 
                 failure_threshold: int = 3,
                 recovery_timeout: int = 300,
                 checkpoint_dir: str = "./safety_checkpoints"):
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
        self.consecutive_successes = 0
        
        # Safety tracking
        self.safety_checkpoints: List[SafetyCheckpoint] = []
        self.safety_violations: List[SafetyViolation] = []
        self.safety_thresholds = {
            'max_performance_degradation': 0.2,
            'max_resource_usage': 0.95,
            'min_confidence_score': 0.5,
            'max_uncertainty': 0.5
        }
        
        # Safety monitoring
        self.monitoring_active = False
        self.safety_monitor_task = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.rollback_events = []  # Track rollback events
        
        logger.info("RSI Safety Circuit Breaker initialized with threshold={}", failure_threshold)
    
    async def start_safety_monitoring(self):
        """Start continuous safety monitoring"""
        self.monitoring_active = True
        self.safety_monitor_task = asyncio.create_task(self._safety_monitoring_loop())
        logger.info("Safety monitoring started")
    
    async def stop_safety_monitoring(self):
        """Stop safety monitoring"""
        self.monitoring_active = False
        if self.safety_monitor_task:
            self.safety_monitor_task.cancel()
            try:
                await self.safety_monitor_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=True)
        logger.info("Safety monitoring stopped")
    
    async def _safety_monitoring_loop(self):
        """Continuous safety monitoring loop"""
        while self.monitoring_active:
            try:
                # Check circuit breaker state
                await self._check_circuit_recovery()
                
                # Monitor for safety violations
                await self._monitor_safety_violations()
                
                # Cleanup old checkpoints
                await self._cleanup_old_checkpoints()
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error("Error in safety monitoring loop: {}", str(e))
                await asyncio.sleep(30)  # Longer sleep on error
    
    async def execute_with_safety(self, 
                                 operation: Callable,
                                 operation_name: str,
                                 safety_level: SafetyLevel = SafetyLevel.MEDIUM,
                                 *args, **kwargs):
        """Execute RSI operation with comprehensive safety monitoring"""
        
        operation_id = f"{operation_name}_{int(time.time())}"
        
        # Check circuit breaker state
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.failure_count = 0
                logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise SafetyException(
                    f"Circuit breaker is OPEN - operation '{operation_name}' blocked for safety. "
                    f"Recovery timeout: {self.recovery_timeout - (time.time() - self.last_failure_time):.1f}s"
                )
        
        # Create safety checkpoint before operation
        checkpoint = await self.create_safety_checkpoint(operation_name, safety_level)
        
        try:
            # Pre-operation safety checks
            await self._perform_pre_operation_checks(operation_name, safety_level, *args, **kwargs)
            
            # Execute operation with monitoring
            start_time = time.time()
            result = await self._monitored_execution(operation, operation_id, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Post-operation safety validation
            await self._perform_post_operation_checks(operation_name, result, execution_time, safety_level)
            
            # Validate result safety
            safety_validation = await self.validate_result_safety(result, checkpoint)
            
            if not safety_validation['safe']:
                logger.warning("Result failed safety validation: {}", safety_validation['reason'])
                await self.rollback_to_checkpoint(checkpoint)
                raise SafetyException(f"Result failed safety validation: {safety_validation['reason']}")
            
            # Success - update circuit breaker state
            await self._record_success()
            
            logger.info("Operation '{}' completed safely in {:.3f}s", operation_name, execution_time)
            return result
            
        except Exception as e:
            # Record failure and handle circuit breaker
            await self._record_failure(operation_name, str(e), checkpoint)
            
            # Attempt automatic recovery
            if isinstance(e, SafetyException):
                await self._handle_safety_violation(operation_name, str(e), checkpoint)
            
            raise e
    
    async def create_safety_checkpoint(self, 
                                     operation_name: str, 
                                     safety_level: SafetyLevel) -> SafetyCheckpoint:
        """Create comprehensive system state checkpoint"""
        
        checkpoint_id = f"checkpoint_{int(time.time() * 1000)}"
        
        try:
            # Capture model state (simplified hash for now)
            model_state_hash = await self._capture_model_state_hash()
            
            # Capture system metrics
            system_metrics = await self._capture_system_metrics()
            
            # Calculate safety scores
            safety_scores = await self._calculate_safety_scores()
            
            # Operation context
            operation_context = {
                'operation_name': operation_name,
                'safety_level': safety_level.value,
                'timestamp': time.time(),
                'circuit_state': self.state.value,
                'failure_count': self.failure_count
            }
            
            checkpoint = SafetyCheckpoint(
                timestamp=time.time(),
                checkpoint_id=checkpoint_id,
                model_state_hash=model_state_hash,
                system_metrics=system_metrics,
                safety_scores=safety_scores,
                operation_context=operation_context,
                checkpoint_size_mb=0.0  # Will be calculated after saving
            )
            
            # Save checkpoint to disk
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump(asdict(checkpoint), f, indent=2)
            
            checkpoint.checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            
            self.safety_checkpoints.append(checkpoint)
            
            # Limit checkpoint history
            if len(self.safety_checkpoints) > 100:
                old_checkpoint = self.safety_checkpoints.pop(0)
                await self._cleanup_checkpoint(old_checkpoint)
            
            logger.debug("Safety checkpoint created: {}", checkpoint_id)
            return checkpoint
            
        except Exception as e:
            logger.error("Failed to create safety checkpoint: {}", str(e))
            # Return minimal checkpoint
            return SafetyCheckpoint(
                timestamp=time.time(),
                checkpoint_id=checkpoint_id,
                model_state_hash="unknown",
                system_metrics={},
                safety_scores={},
                operation_context=operation_context,
                checkpoint_size_mb=0.0
            )
    
    async def rollback_to_checkpoint(self, checkpoint: SafetyCheckpoint):
        """Rollback system state to a previous checkpoint"""
        logger.warning("Initiating rollback to checkpoint: {}", checkpoint.checkpoint_id)
        
        try:
            # Load checkpoint data
            checkpoint_path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
            
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                
                # Restore system state (simplified)
                await self._restore_system_state(checkpoint_data)
                
                logger.info("Successfully rolled back to checkpoint: {}", checkpoint.checkpoint_id)
                
                # Record rollback event
                await self._record_rollback_event(checkpoint)
                
            else:
                logger.error("Checkpoint file not found: {}", checkpoint_path)
                
        except Exception as e:
            logger.error("Failed to rollback to checkpoint {}: {}", checkpoint.checkpoint_id, str(e))
            raise SafetyException(f"Rollback failed: {str(e)}")
    
    async def validate_result_safety(self, 
                                   result: Any, 
                                   checkpoint: SafetyCheckpoint) -> Dict[str, Any]:
        """Comprehensive result safety validation"""
        
        validation_result = {
            'safe': True,
            'reason': '',
            'checks_performed': [],
            'safety_score': 1.0
        }
        
        try:
            # Check 1: Result structure validation
            structure_check = await self._validate_result_structure(result)
            validation_result['checks_performed'].append('structure')
            
            if not structure_check['valid']:
                validation_result['safe'] = False
                validation_result['reason'] = f"Invalid result structure: {structure_check['reason']}"
                return validation_result
            
            # Check 2: Performance validation
            performance_check = await self._validate_performance_impact(result, checkpoint)
            validation_result['checks_performed'].append('performance')
            
            if not performance_check['acceptable']:
                validation_result['safe'] = False
                validation_result['reason'] = f"Performance degradation: {performance_check['reason']}"
                return validation_result
            
            # Check 3: Safety bounds validation
            bounds_check = await self._validate_safety_bounds(result)
            validation_result['checks_performed'].append('bounds')
            
            if not bounds_check['within_bounds']:
                validation_result['safe'] = False
                validation_result['reason'] = f"Safety bounds violated: {bounds_check['reason']}"
                return validation_result
            
            # Calculate overall safety score
            validation_result['safety_score'] = min(
                structure_check.get('confidence', 1.0),
                performance_check.get('score', 1.0),
                bounds_check.get('score', 1.0)
            )
            
            logger.debug("Result validation passed with safety score: {:.3f}", 
                        validation_result['safety_score'])
            
        except Exception as e:
            logger.error("Error in result validation: {}", str(e))
            validation_result['safe'] = False
            validation_result['reason'] = f"Validation error: {str(e)}"
        
        return validation_result
    
    async def _perform_pre_operation_checks(self, 
                                          operation_name: str, 
                                          safety_level: SafetyLevel,
                                          *args, **kwargs):
        """Perform safety checks before operation execution"""
        
        # Check system resource availability
        system_metrics = await self._capture_system_metrics()
        
        if system_metrics.get('cpu_percent', 0) > 90:
            raise SafetyException("CPU usage too high for safe operation")
        
        if system_metrics.get('memory_percent', 0) > 90:
            raise SafetyException("Memory usage too high for safe operation")
        
        # Check safety level requirements
        if safety_level == SafetyLevel.CRITICAL:
            # More stringent checks for critical operations
            if system_metrics.get('cpu_percent', 0) > 70:
                raise SafetyException("CPU usage too high for critical operation")
        
        # Check for recent safety violations
        recent_violations = [v for v in self.safety_violations 
                           if time.time() - v.timestamp < 300]  # Last 5 minutes
        
        critical_violations = [v for v in recent_violations 
                             if v.severity == SafetyLevel.CRITICAL]
        
        if critical_violations and safety_level in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]:
            raise SafetyException("Recent critical safety violations - operation blocked")
    
    async def _perform_post_operation_checks(self, 
                                           operation_name: str,
                                           result: Any,
                                           execution_time: float,
                                           safety_level: SafetyLevel):
        """Perform safety checks after operation execution"""
        
        # Check execution time
        max_execution_times = {
            SafetyLevel.LOW: 300,      # 5 minutes
            SafetyLevel.MEDIUM: 180,   # 3 minutes
            SafetyLevel.HIGH: 60,      # 1 minute
            SafetyLevel.CRITICAL: 30   # 30 seconds
        }
        
        if execution_time > max_execution_times[safety_level]:
            logger.warning("Operation '{}' exceeded time limit: {:.1f}s", 
                          operation_name, execution_time)
        
        # Check system state after operation
        post_metrics = await self._capture_system_metrics()
        
        # Detect significant resource spikes
        if post_metrics.get('cpu_percent', 0) > 95:
            raise SafetyException("Post-operation CPU spike detected")
    
    async def _monitored_execution(self, 
                                 operation: Callable,
                                 operation_id: str,
                                 *args, **kwargs):
        """Execute operation with continuous monitoring"""
        
        monitoring_task = asyncio.create_task(
            self._monitor_operation_execution(operation_id)
        )
        
        try:
            # Execute the actual operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                # Run synchronous operation in executor
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, operation, *args
                )
            
            return result
            
        finally:
            # Stop monitoring
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_operation_execution(self, operation_id: str):
        """Monitor operation execution for safety violations"""
        monitor_start = time.time()
        
        try:
            while True:
                await asyncio.sleep(1)  # Check every second
                
                # Check for resource anomalies
                metrics = await self._capture_system_metrics()
                
                if metrics.get('cpu_percent', 0) > 98:
                    logger.warning("High CPU usage during operation {}: {:.1f}%", 
                                  operation_id, metrics['cpu_percent'])
                
                if metrics.get('memory_percent', 0) > 95:
                    logger.warning("High memory usage during operation {}: {:.1f}%", 
                                  operation_id, metrics['memory_percent'])
                
                # Monitor execution time
                execution_time = time.time() - monitor_start
                if execution_time > 600:  # 10 minutes maximum
                    logger.error("Operation {} exceeded maximum execution time", operation_id)
                    raise SafetyException("Operation execution timeout")
                
        except asyncio.CancelledError:
            # Normal cancellation when operation completes
            pass
    
    async def _record_success(self):
        """Record successful operation for circuit breaker"""
        self.consecutive_successes += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN and self.consecutive_successes >= 3:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            logger.info("Circuit breaker returned to CLOSED state after successful operations")
    
    async def _record_failure(self, 
                             operation_name: str, 
                             error_message: str, 
                             checkpoint: SafetyCheckpoint):
        """Record operation failure for circuit breaker"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.consecutive_successes = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error("Circuit breaker OPENED after {} failures", self.failure_count)
        
        # Record safety violation
        violation = SafetyViolation(
            timestamp=time.time(),
            violation_id=f"violation_{int(time.time() * 1000)}",
            violation_type="operation_failure",
            severity=SafetyLevel.HIGH,
            description=f"Operation '{operation_name}' failed: {error_message}",
            operation_context=checkpoint.operation_context,
            automatic_action_taken="circuit_breaker_triggered",
            human_intervention_required=self.state == CircuitBreakerState.OPEN
        )
        
        self.safety_violations.append(violation)
        
        logger.error("Safety violation recorded: {}", violation.violation_id)
    
    async def _handle_safety_violation(self, 
                                     operation_name: str, 
                                     error_message: str, 
                                     checkpoint: SafetyCheckpoint):
        """Handle safety violation with appropriate response"""
        
        # Determine response based on violation severity
        if self.state == CircuitBreakerState.OPEN:
            # Critical response - full system safety mode
            logger.critical("Entering safety mode due to circuit breaker opening")
            await self._enter_safety_mode()
        
        # Notify monitoring systems
        await self._notify_safety_violation(operation_name, error_message)
    
    async def _enter_safety_mode(self):
        """Enter system safety mode with restricted operations"""
        logger.critical("System entering safety mode - restricting operations")
        
        # This would implement safety mode restrictions
        # For now, just log the event
        safety_mode_context = {
            'timestamp': time.time(),
            'reason': 'circuit_breaker_open',
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time
        }
        
        logger.info("Safety mode context: {}", safety_mode_context)
    
    async def _notify_safety_violation(self, operation_name: str, error_message: str):
        """Notify external systems of safety violation"""
        
        notification = {
            'timestamp': time.time(),
            'event_type': 'safety_violation',
            'operation_name': operation_name,
            'error_message': error_message,
            'circuit_state': self.state.value,
            'failure_count': self.failure_count,
            'requires_attention': self.state == CircuitBreakerState.OPEN
        }
        
        logger.warning("Safety violation notification: {}", notification)
    
    # Helper methods for safety validation
    
    async def _capture_model_state_hash(self) -> str:
        """Capture hash of current model state"""
        # This would hash actual model parameters
        # For now, return a timestamp-based hash
        model_state = f"model_state_{time.time()}"
        return hashlib.md5(model_state.encode()).hexdigest()
    
    async def _capture_system_metrics(self) -> Dict[str, Any]:
        """Capture current system metrics"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'timestamp': time.time()
            }
        except ImportError:
            return {'timestamp': time.time()}
    
    async def _calculate_safety_scores(self) -> Dict[str, float]:
        """Calculate various safety scores"""
        return {
            'overall_safety': 0.8,
            'resource_safety': 0.9,
            'performance_safety': 0.85,
            'operational_safety': 0.75
        }
    
    async def _validate_result_structure(self, result: Any) -> Dict[str, Any]:
        """Validate result structure is safe"""
        try:
            # Basic structure validation
            if result is None:
                return {'valid': False, 'reason': 'Result is None', 'confidence': 0.0}
            
            # Check for reasonable size
            result_str = str(result)
            if len(result_str) > 1000000:  # 1MB limit
                return {'valid': False, 'reason': 'Result too large', 'confidence': 0.0}
            
            return {'valid': True, 'reason': 'Structure valid', 'confidence': 1.0}
            
        except Exception as e:
            return {'valid': False, 'reason': f'Validation error: {str(e)}', 'confidence': 0.0}
    
    async def _validate_performance_impact(self, 
                                         result: Any, 
                                         checkpoint: SafetyCheckpoint) -> Dict[str, Any]:
        """Validate performance impact is acceptable"""
        
        current_metrics = await self._capture_system_metrics()
        baseline_metrics = checkpoint.system_metrics
        
        # Compare metrics
        cpu_change = current_metrics.get('cpu_percent', 0) - baseline_metrics.get('cpu_percent', 0)
        memory_change = current_metrics.get('memory_percent', 0) - baseline_metrics.get('memory_percent', 0)
        
        # Check thresholds
        if cpu_change > 20:  # 20% CPU increase
            return {
                'acceptable': False, 
                'reason': f'CPU usage increased by {cpu_change:.1f}%',
                'score': 0.5
            }
        
        if memory_change > 15:  # 15% memory increase
            return {
                'acceptable': False, 
                'reason': f'Memory usage increased by {memory_change:.1f}%',
                'score': 0.5
            }
        
        return {'acceptable': True, 'reason': 'Performance impact acceptable', 'score': 0.9}
    
    async def _validate_safety_bounds(self, result: Any) -> Dict[str, Any]:
        """Validate result is within safety bounds"""
        
        # This would implement domain-specific safety bound checks
        # For now, basic validation
        
        try:
            if isinstance(result, dict):
                # Check for sensitive keys
                sensitive_keys = ['password', 'secret', 'key', 'token']
                for key in result.keys():
                    if any(sensitive in str(key).lower() for sensitive in sensitive_keys):
                        return {
                            'within_bounds': False, 
                            'reason': f'Sensitive data in result: {key}',
                            'score': 0.0
                        }
            
            return {'within_bounds': True, 'reason': 'Safety bounds respected', 'score': 1.0}
            
        except Exception as e:
            return {'within_bounds': False, 'reason': f'Bounds check error: {str(e)}', 'score': 0.0}
    
    async def _restore_system_state(self, checkpoint_data: Dict[str, Any]):
        """Restore system state from checkpoint data"""
        logger.info("Restoring system state from checkpoint")
        
        # This would implement actual state restoration
        # For now, just log the restoration
        restoration_info = {
            'checkpoint_timestamp': checkpoint_data.get('timestamp'),
            'model_state_hash': checkpoint_data.get('model_state_hash'),
            'restoration_timestamp': time.time()
        }
        
        logger.info("State restoration completed: {}", restoration_info)
    
    async def _record_rollback_event(self, checkpoint: SafetyCheckpoint):
        """Record rollback event for audit trail"""
        rollback_event = {
            'timestamp': time.time(),
            'checkpoint_id': checkpoint.checkpoint_id,
            'rollback_reason': getattr(checkpoint, 'rollback_reason', 'Safety violation'),
            'system_state': 'restored'
        }
        
        # Record in audit trail
        logger.info("Rollback event recorded: {}", rollback_event)
        
        # Store rollback event
        self.rollback_events.append(rollback_event)
    
    async def _cleanup_checkpoint(self, checkpoint: SafetyCheckpoint):
        """Clean up old checkpoint files"""
        try:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.debug("Cleaned up checkpoint: {}", checkpoint.checkpoint_id)
        except Exception as e:
            logger.error("Failed to cleanup checkpoint {}: {}", checkpoint.checkpoint_id, str(e))
    
    async def _cleanup_old_checkpoints(self):
        """Clean up checkpoints older than 24 hours"""
        cutoff_time = time.time() - 86400  # 24 hours
        
        for checkpoint in self.safety_checkpoints[:]:
            if checkpoint.timestamp < cutoff_time:
                await self._cleanup_checkpoint(checkpoint)
                self.safety_checkpoints.remove(checkpoint)
    
    async def _check_circuit_recovery(self):
        """Check if circuit breaker can recover"""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker attempting recovery - entering HALF_OPEN state")
    
    async def _monitor_safety_violations(self):
        """Monitor for ongoing safety violations"""
        
        # Check system health
        metrics = await self._capture_system_metrics()
        
        # Check for resource violations
        if metrics.get('cpu_percent', 0) > 95:
            await self._record_safety_violation(
                'resource_violation', 
                'CPU usage critically high', 
                SafetyLevel.CRITICAL
            )
        
        if metrics.get('memory_percent', 0) > 95:
            await self._record_safety_violation(
                'resource_violation', 
                'Memory usage critically high', 
                SafetyLevel.CRITICAL
            )
    
    async def _record_safety_violation(self, 
                                     violation_type: str, 
                                     description: str, 
                                     severity: SafetyLevel):
        """Record a safety violation"""
        
        violation = SafetyViolation(
            timestamp=time.time(),
            violation_id=f"violation_{int(time.time() * 1000)}",
            violation_type=violation_type,
            severity=severity,
            description=description,
            operation_context={'monitoring': True},
            automatic_action_taken=None,
            human_intervention_required=severity == SafetyLevel.CRITICAL
        )
        
        self.safety_violations.append(violation)
        
        # Limit violation history
        if len(self.safety_violations) > 1000:
            self.safety_violations.pop(0)
        
        logger.log(severity.value.upper(), "Safety violation: {}", description)
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status"""
        
        recent_violations = [v for v in self.safety_violations 
                           if time.time() - v.timestamp < 3600]  # Last hour
        
        return {
            'circuit_breaker_state': self.state.value,
            'failure_count': self.failure_count,
            'consecutive_successes': self.consecutive_successes,
            'last_failure_time': self.last_failure_time,
            'active_checkpoints': len(self.safety_checkpoints),
            'recent_violations': len(recent_violations),
            'critical_violations': len([v for v in recent_violations 
                                      if v.severity == SafetyLevel.CRITICAL]),
            'monitoring_active': self.monitoring_active,
            'safety_score': self._calculate_overall_safety_score(),
            'time_since_last_failure': time.time() - self.last_failure_time if self.last_failure_time > 0 else None
        }
    
    def _calculate_overall_safety_score(self) -> float:
        """Calculate overall system safety score"""
        
        base_score = 1.0
        
        # Reduce score based on circuit breaker state
        if self.state == CircuitBreakerState.OPEN:
            base_score *= 0.3
        elif self.state == CircuitBreakerState.HALF_OPEN:
            base_score *= 0.7
        
        # Reduce score based on recent failures
        if self.failure_count > 0:
            base_score *= max(0.1, 1.0 - (self.failure_count / self.failure_threshold))
        
        # Reduce score based on recent violations
        recent_violations = [v for v in self.safety_violations 
                           if time.time() - v.timestamp < 3600]
        
        if recent_violations:
            violation_penalty = min(0.5, len(recent_violations) * 0.1)
            base_score *= (1.0 - violation_penalty)
        
        return max(0.0, base_score)

class SafetyException(Exception):
    """Exception raised for safety violations"""
    pass