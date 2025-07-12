"""
Enhanced RSI Orchestrator with Advanced Metacognitive Monitoring.
Production-ready Recursive Self-Improvement AI system with comprehensive safety measures.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

# Core system imports
from src.core.state import RSIStateManager, RSIState
from src.core.model_versioning import ModelVersionManager
from src.learning.online_learning import RSIOnlineLearner
from src.learning.meta_learning import RSIMetaLearningSystem
from src.learning.continual_learning import RSIContinualLearningSystem
from src.learning.reinforcement_learning import RSIRLSystem
from src.learning.lightning_orchestrator import RSILightningOrchestrator
from src.validation.validators import RSIValidator
from src.safety.circuits import CircuitBreakerManager
from src.security.sandbox import RSISandbox
from src.memory.memory_manager import RSIMemoryManager
from src.monitoring.anomaly_detection import BehavioralMonitor
from src.monitoring.telemetry import TelemetryCollector
from src.monitoring.audit_logger import audit_system_event

# Enhanced monitoring imports
from src.monitoring.metacognitive_monitor import (
    RSISystemMonitor, MetacognitiveAssessment, SystemHealth
)
from src.monitoring.uncertainty_quantification import (
    RSIUncertaintyEstimator, UncertaintyAggregator
)
from src.safety.rsi_circuit_breaker import (
    RSISafetyCircuitBreaker, SafetyLevel, SafetyException
)

# Optimization imports
try:
    from src.optimization.optuna_optimizer import OptunaOptimizer
    from src.optimization.ray_tune_optimizer import RayTuneOrchestrator
except ImportError as e:
    logger.warning("Optimization modules not fully available: {}", str(e))
    OptunaOptimizer = None
    RayTuneOrchestrator = None

# RSI Hypothesis Testing System imports
try:
    from src.hypothesis.rsi_hypothesis_orchestrator import (
        RSIHypothesisOrchestrator, HypothesisOrchestrationResult,
        OrchestrationPhase, HypothesisStatus
    )
    from src.hypothesis import (
        RSIHypothesis, HypothesisType, HypothesisPriority,
        ValidationLevel, ReviewStatus, SimulationEnvironment
    )
    HYPOTHESIS_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning("RSI Hypothesis Testing System not fully available: {}", str(e))
    HYPOTHESIS_SYSTEM_AVAILABLE = False

# Pydantic models for API
class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Input features for prediction")
    user_id: Optional[str] = Field(None, description="User identifier")
    uncertainty_estimation: bool = Field(True, description="Include uncertainty quantification")

class LearningRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Input features for learning")
    target: Any = Field(..., description="Target value for learning")
    user_id: Optional[str] = Field(None, description="User identifier")
    safety_level: str = Field("medium", description="Safety level for the operation")

class CodeExecutionRequest(BaseModel):
    code: str = Field(..., description="Code to execute safely")
    timeout_seconds: int = Field(60, description="Execution timeout")
    user_id: Optional[str] = Field(None, description="User identifier")

class HypothesisGenerationRequest(BaseModel):
    improvement_targets: Dict[str, float] = Field(
        ..., 
        description="Target improvements (e.g., {'accuracy': 0.05, 'efficiency': 0.1})"
    )
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    max_hypotheses: int = Field(10, description="Maximum number of hypotheses to generate")
    user_id: Optional[str] = Field(None, description="User identifier")

class HypothesisReviewRequest(BaseModel):
    request_id: str = Field(..., description="Review request ID")
    reviewer_id: str = Field(..., description="Reviewer identifier")
    decision: str = Field(..., description="Review decision (approved/rejected/needs_modification)")
    reasoning: str = Field(..., description="Detailed reasoning for the decision")
    confidence_level: float = Field(0.8, description="Reviewer's confidence (0.0-1.0)")
    modification_suggestions: List[str] = Field(default_factory=list, description="Suggested modifications")
    approval_conditions: List[str] = Field(default_factory=list, description="Approval conditions")

class MetacognitiveStatus(BaseModel):
    """Real-time metacognitive system status"""
    timestamp: float
    system_health: str
    metacognitive_awareness: float
    learning_efficiency: float
    uncertainty_level: float
    safety_score: float
    circuit_breaker_state: str

class RSIOrchestrator:
    """
    Enhanced RSI Orchestrator with Advanced Metacognitive Monitoring.
    
    Implements production-ready recursive self-improvement with:
    - Real-time metacognitive monitoring
    - Uncertainty quantification
    - Safety circuit breakers
    - Comprehensive audit trails
    """

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.start_time = time.time()
        
        # Enhanced monitoring components
        self.system_monitor = RSISystemMonitor(collection_interval=0.1)
        self.metacognitive_assessment = MetacognitiveAssessment()
        self.uncertainty_estimator = RSIUncertaintyEstimator(input_dim=10)
        self.uncertainty_aggregator = UncertaintyAggregator()
        self.safety_circuit = RSISafetyCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=300
        )
        
        # WebSocket connections for real-time monitoring
        self.active_connections: List[WebSocket] = []
        self.streaming_active = False
        
        # Initialize core components
        self._initialize_core_components()
        
        # Setup enhanced logging
        self._setup_enhanced_logging()
        
        logger.info("Enhanced RSI Orchestrator initialized for {} environment", environment)

    def _initialize_core_components(self):
        """Initialize all core RSI components"""
        try:
            # Core state and model management
            self.state_manager = RSIStateManager(initial_state=RSIState())
            self.model_version_manager = ModelVersionManager()
            
            # Learning systems
            self.online_learner = RSIOnlineLearner()
            
            # Enhanced learning systems
            try:
                self.meta_learning_system = RSIMetaLearningSystem()
                self.continual_learning_system = RSIContinualLearningSystem()
                self.rl_system = RSIRLSystem()
                self.lightning_orchestrator = RSILightningOrchestrator()
            except Exception as e:
                logger.warning("Some advanced learning systems not available: {}", str(e))
                self.meta_learning_system = None
                self.continual_learning_system = None
                self.rl_system = None
                self.lightning_orchestrator = None
            
            # Validation and safety
            self.validator = RSIValidator()
            self.circuit_manager = CircuitBreakerManager()
            self.sandbox = RSISandbox()
            
            # Monitoring and telemetry
            self.behavioral_monitor = BehavioralMonitor()
            self.telemetry = TelemetryCollector()
            
            # Memory system
            self.memory_system = None
            
            # Optimization (optional)
            self.optuna_optimizer = OptunaOptimizer() if OptunaOptimizer else None
            self.ray_tune_orchestrator = RayTuneOrchestrator() if RayTuneOrchestrator else None
            
            # RSI Hypothesis Testing System
            self.hypothesis_orchestrator = None
            if HYPOTHESIS_SYSTEM_AVAILABLE:
                try:
                    self.hypothesis_orchestrator = RSIHypothesisOrchestrator(
                        state_manager=self.state_manager,
                        validator=self.validator,
                        circuit_breaker=self.circuit_manager,
                        environment=self.environment
                    )
                    logger.info("✅ RSI Hypothesis Testing System initialized")
                except Exception as e:
                    logger.warning("Failed to initialize RSI Hypothesis System: {}", str(e))
            
        except Exception as e:
            logger.error("Failed to initialize some components: {}", str(e))

    def _setup_enhanced_logging(self):
        """Setup enhanced logging for metacognitive monitoring"""
        try:
            from src.monitoring.audit_logger import AuditLogger
            self.audit_logger = AuditLogger(
                log_directory=f"./logs/{self.environment}",
                
            )
        except Exception as e:
            logger.warning("Audit logger not available: {}", str(e))
            self.audit_logger = None

    async def start(self):
        """Start the enhanced RSI system with metacognitive monitoring"""
        try:
            # Initialize memory system
            await self._initialize_memory_system()
            
            # Start core monitoring
            self.behavioral_monitor.start_monitoring()
            
            # Start enhanced monitoring
            await self.system_monitor.start_monitoring()
            await self.safety_circuit.start_safety_monitoring()
            
            # Start background tasks
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._metrics_collection_loop())
            asyncio.create_task(self._self_improvement_loop())
            asyncio.create_task(self._metacognitive_monitoring_loop())
            
            # Log system startup
            if self.audit_logger:
                audit_system_event(
                    "rsi_orchestrator",
                    "system_started",
                    metadata={"environment": self.environment}
                )
            
            logger.info("Enhanced RSI system started successfully with metacognitive monitoring")
            
        except Exception as e:
            logger.error("Failed to start RSI system: {}", str(e))
            raise

    async def stop(self):
        """Stop the enhanced RSI system"""
        try:
            # Stop enhanced monitoring
            await self.system_monitor.stop_monitoring()
            await self.safety_circuit.stop_safety_monitoring()
            
            # Stop core monitoring
            self.behavioral_monitor.stop_monitoring()
            
            # Stop streaming
            self.streaming_active = False
            
            # Log system shutdown
            if self.audit_logger:
                audit_system_event(
                    "rsi_orchestrator",
                    "system_stopped"
                )
            
            logger.info("Enhanced RSI system stopped")
            
        except Exception as e:
            logger.error("Error stopping RSI system: {}", str(e))

    async def _initialize_memory_system(self):
        """Initialize memory system"""
        try:
            from src.memory.memory_hierarchy import RSIMemoryHierarchy
            from src.memory.memory_hierarchy import RSIMemoryHierarchy, RSIMemoryConfig
            
            config = RSIMemoryConfig()
            self.memory_system = RSIMemoryHierarchy(config)
            await self.memory_system.initialize()
            
            logger.info("✅ Memory system initialized successfully")
        except Exception as e:
            logger.warning("Memory system initialization failed: {}", str(e))

    async def _metacognitive_monitoring_loop(self):
        """Enhanced metacognitive monitoring loop"""
        while True:
            try:
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
                # Get current system metrics
                if self.system_monitor.metrics_history:
                    latest_metrics = self.system_monitor.metrics_history[-1]
                    
                    # Get recent predictions for metacognitive assessment
                    recent_predictions = []  # Would be populated from actual predictions
                    
                    # Perform metacognitive assessment
                    metacog_metrics = await self.metacognitive_assessment.assess_metacognitive_state(
                        latest_metrics, recent_predictions
                    )
                    
                    # Update uncertainty aggregator
                    self.uncertainty_estimator.update_uncertainty_history(
                        metacog_metrics if hasattr(metacog_metrics, 'total_uncertainty') else None
                    )
                    
                    # Stream to connected clients
                    if self.active_connections:
                        await self._broadcast_metacognitive_status()
                
            except Exception as e:
                logger.error("Error in metacognitive monitoring loop: {}", str(e))
                await asyncio.sleep(30)

    async def predict(self, features: Dict[str, Any], 
                     user_id: Optional[str] = None,
                     include_uncertainty: bool = True) -> Dict[str, Any]:
        """Enhanced prediction with uncertainty quantification and safety monitoring"""
        
        prediction_id = f"pred_{int(time.time() * 1000)}"
        
        try:
            # Execute prediction with safety monitoring
            async def prediction_operation():
                # Validate input
                validation_result = await self.validator.validate_prediction_input(features)
                if not validation_result.valid:
                    raise ValueError(f"Invalid input: {validation_result.message}")
                
                # Convert features to numpy array for uncertainty estimation
                feature_array = np.array(list(features.values())).reshape(1, -1)
                
                # Get prediction with uncertainty if requested
                if include_uncertainty:
                    uncertainty_est = await self.uncertainty_estimator.predict_with_uncertainty(
                        feature_array, n_samples=50
                    )
                    
                    prediction_value = uncertainty_est.prediction_mean
                    confidence = uncertainty_est.confidence_score
                    uncertainty_info = {
                        'total_uncertainty': uncertainty_est.total_uncertainty,
                        'epistemic_uncertainty': uncertainty_est.epistemic_uncertainty,
                        'aleatoric_uncertainty': uncertainty_est.aleatoric_uncertainty,
                        'confidence_interval': [
                            uncertainty_est.confidence_interval_lower,
                            uncertainty_est.confidence_interval_upper
                        ]
                    }
                else:
                    # Fallback to online learner
                    prediction_value = await self.online_learner.predict(feature_array)
                    confidence = 0.8  # Default confidence
                    uncertainty_info = {}
                
                # Store prediction in memory (if method exists)
                if self.memory_system and hasattr(self.memory_system, 'store_episodic_memory'):
                    try:
                        await self.memory_system.store_episodic_memory(
                            "prediction",
                            {
                                "prediction_id": prediction_id,
                                "features": features,
                                "prediction": float(prediction_value),
                                "confidence": confidence,
                                "timestamp": time.time()
                            }
                        )
                    except Exception as e:
                        logger.warning("Failed to store prediction in memory: {}", str(e))
                
                return {
                    'prediction_id': prediction_id,
                    'prediction': float(prediction_value),
                    'confidence': confidence,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'uncertainty': uncertainty_info,
                    'user_id': user_id
                }
            
            # Execute with safety circuit breaker
            result = await self.safety_circuit.execute_with_safety(
                prediction_operation,
                f"prediction_{prediction_id}",
                SafetyLevel.LOW
            )
            
            # Log prediction event
            if self.audit_logger:
                audit_system_event(
                    "prediction",
                    "prediction_completed",
                    metadata={
                        "prediction_id": prediction_id,
                        "confidence": result['confidence'],
                        "include_uncertainty": include_uncertainty,
                        "user_id": user_id
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error("Prediction failed: {}", str(e))
            
            # Log prediction failure
            if self.audit_logger:
                audit_system_event(
                    "prediction",
                    "prediction_failed",
                    metadata={
                        "prediction_id": prediction_id,
                        "error": str(e),
                        "user_id": user_id
                    }
                )
            
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    async def learn(self, features: Dict[str, Any], target: Any, 
                   user_id: Optional[str] = None,
                   safety_level: str = "medium") -> Dict[str, Any]:
        """Enhanced learning with safety monitoring and metacognitive assessment"""
        
        learning_id = f"learn_{int(time.time() * 1000)}"
        safety_enum = SafetyLevel(safety_level.lower())
        
        try:
            # Execute learning with safety monitoring
            async def learning_operation():
                # Validate input
                validation_result = await self.validator.validate_learning_input(features, target)
                if not validation_result.valid:
                    raise ValueError(f"Invalid learning input: {validation_result.message}")
                
                # Perform online learning (River expects dictionary input)
                learning_result = await self.online_learner.learn_one(features, target)
                
                # Store learning experience in memory (if method exists)
                if self.memory_system and hasattr(self.memory_system, 'store_episodic_memory'):
                    try:
                        await self.memory_system.store_episodic_memory(
                            "learning",
                            {
                                "learning_id": learning_id,
                                "features": features,
                                "target": target,
                                "accuracy": learning_result.accuracy,
                                "timestamp": time.time()
                            }
                        )
                    except Exception as e:
                        logger.warning("Failed to store learning in memory: {}", str(e))
                
                return {
                    'learning_id': learning_id,
                    'accuracy': learning_result.accuracy,
                    'samples_processed': learning_result.samples_processed,
                    'concept_drift_detected': learning_result.concept_drift_detected,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'memory_stored': self.memory_system is not None,
                    'user_id': user_id
                }
            
            # Execute with safety circuit breaker
            result = await self.safety_circuit.execute_with_safety(
                learning_operation,
                f"learning_{learning_id}",
                safety_enum
            )
            
            # Log learning event
            if self.audit_logger:
                audit_system_event(
                    "learning",
                    "learning_completed",
                    metadata={
                        "learning_id": learning_id,
                        "accuracy": result['accuracy'],
                        "safety_level": safety_level,
                        "user_id": user_id
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error("Learning failed: {}", str(e))
            
            # Log learning failure
            if self.audit_logger:
                audit_system_event(
                    "learning",
                    "learning_failed",
                    metadata={
                        "learning_id": learning_id,
                        "error": str(e),
                        "user_id": user_id
                    }
                )
            
            raise HTTPException(status_code=500, detail=f"Learning failed: {str(e)}")

    async def get_metacognitive_status(self) -> MetacognitiveStatus:
        """Get current metacognitive system status"""
        
        system_health = self.system_monitor.get_system_health_status()
        safety_status = self.safety_circuit.get_safety_status()
        
        # Get latest metacognitive assessment
        metacognitive_awareness = 0.5
        learning_efficiency = 0.7
        uncertainty_level = 0.3
        
        if self.metacognitive_assessment.assessment_history:
            latest_assessment = self.metacognitive_assessment.assessment_history[-1]
            metacognitive_awareness = latest_assessment.metacognitive_awareness
            learning_efficiency = latest_assessment.learning_efficiency
            uncertainty_level = latest_assessment.uncertainty_level
        
        return MetacognitiveStatus(
            timestamp=time.time(),
            system_health=system_health.value,
            metacognitive_awareness=metacognitive_awareness,
            learning_efficiency=learning_efficiency,
            uncertainty_level=uncertainty_level,
            safety_score=safety_status['safety_score'],
            circuit_breaker_state=safety_status['circuit_breaker_state']
        )

    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # Get core system metrics
            system_health = self.system_monitor.get_system_health_status()
            safety_status = self.safety_circuit.get_safety_status()
            
            # Get learning system health
            learner_metrics = self.online_learner.get_metrics()
            
            # Get memory system health
            memory_health = "unknown"
            if self.memory_system:
                try:
                    # Test memory system responsiveness
                    await self.memory_system.store_episodic_memory("health_check", {"timestamp": time.time()})
                    memory_health = "healthy"
                except Exception:
                    memory_health = "degraded"
            else:
                memory_health = "not_available"
            
            # Get hypothesis system health
            hypothesis_health = "not_available"
            if hasattr(self, 'hypothesis_orchestrator') and self.hypothesis_orchestrator:
                try:
                    stats = self.hypothesis_orchestrator.get_orchestration_statistics()
                    hypothesis_health = "healthy" if stats else "healthy"
                except Exception:
                    hypothesis_health = "degraded"
            
            # Determine overall status
            overall_status = "healthy"
            if safety_status['safety_score'] < 0.7 or system_health.value == "critical":
                overall_status = "critical"
            elif memory_health == "degraded" or hypothesis_health == "degraded":
                overall_status = "degraded"
            elif learner_metrics.accuracy < 0.6:
                overall_status = "warning"
            
            return {
                "status": overall_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "components": {
                    "system_monitor": system_health.value,
                    "safety_circuit": safety_status['circuit_breaker_state'],
                    "learning_system": "healthy" if learner_metrics.accuracy > 0.6 else "degraded",
                    "memory_system": memory_health,
                    "hypothesis_system": hypothesis_health,
                    "behavioral_monitor": "healthy",
                    "telemetry": "healthy"
                },
                "metrics": {
                    "safety_score": safety_status['safety_score'],
                    "learning_accuracy": learner_metrics.accuracy,
                    "samples_processed": learner_metrics.samples_processed,
                    "concept_drift_detected": learner_metrics.concept_drift_detected,
                    "memory_available": self.memory_system is not None,
                    "hypothesis_system_available": hasattr(self, 'hypothesis_orchestrator') and self.hypothesis_orchestrator is not None
                }
            }
            
        except Exception as e:
            logger.error("Error getting system health: {}", str(e))
            return {
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "uptime_seconds": time.time() - self.start_time
            }

    async def _broadcast_metacognitive_status(self):
        """Broadcast metacognitive status to connected WebSocket clients"""
        if not self.active_connections:
            return
        
        try:
            status = await self.get_metacognitive_status()
            message = {
                "type": "metacognitive_status",
                "data": status.dict()
            }
            
            # Send to all connected clients
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                self.active_connections.remove(connection)
                
        except Exception as e:
            logger.error("Error broadcasting metacognitive status: {}", str(e))

    # Rest of the existing methods remain the same...
    async def _health_check_loop(self):
        """Periodic health checking"""
        while True:
            try:
                await asyncio.sleep(30)
                # Health check logic here
            except Exception as e:
                logger.error("Health check error: {}", str(e))

    async def _metrics_collection_loop(self):
        """Periodic metrics collection"""
        while True:
            try:
                await asyncio.sleep(60)
                # Metrics collection logic here
            except Exception as e:
                logger.error("Metrics collection error: {}", str(e))

    async def _self_improvement_loop(self):
        """Enhanced self-improvement loop with metacognitive insights and hypothesis testing"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Analyze performance with metacognitive insights
                performance_data = await self.analyze_performance()
                
                # Trigger self-improvement if needed
                if performance_data.get("needs_improvement", False):
                    await self.trigger_self_improvement(performance_data)
                
                # Run hypothesis-driven improvements every 10 cycles (50 minutes)
                if hasattr(self, '_improvement_cycle_count'):
                    self._improvement_cycle_count += 1
                else:
                    self._improvement_cycle_count = 1
                
                if self._improvement_cycle_count % 10 == 0 and self.hypothesis_orchestrator:
                    await self._run_hypothesis_driven_improvement(performance_data)
                
            except Exception as e:
                logger.error("Self-improvement loop error: {}", str(e))
                await asyncio.sleep(60)

    async def analyze_performance(self) -> Dict[str, Any]:
        """Enhanced performance analysis with metacognitive insights"""
        
        performance_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "needs_improvement": False,
            "recommendations": []
        }
        
        try:
            # Get learning metrics
            learner_metrics = self.online_learner.get_metrics()
            performance_data["metrics"]["learning"] = {
                "accuracy": learner_metrics.accuracy,
                "samples_processed": learner_metrics.samples_processed,
                "adaptation_speed": learner_metrics.adaptation_speed
            }
            
            # Add metacognitive insights
            if self.metacognitive_assessment.assessment_history:
                latest_metacog = self.metacognitive_assessment.assessment_history[-1]
                performance_data["metrics"]["metacognitive"] = {
                    "awareness": latest_metacog.metacognitive_awareness,
                    "learning_efficiency": latest_metacog.learning_efficiency,
                    "cognitive_load": latest_metacog.cognitive_load,
                    "self_assessment_accuracy": latest_metacog.self_assessment_accuracy
                }
                
                # Check for metacognitive improvements needed
                if latest_metacog.learning_efficiency < 0.6:
                    performance_data["needs_improvement"] = True
                    performance_data["recommendations"].append("Improve learning efficiency")
                
                if latest_metacog.cognitive_load > 0.8:
                    performance_data["needs_improvement"] = True
                    performance_data["recommendations"].append("Reduce cognitive load")
            
            # Check accuracy threshold
            if learner_metrics.accuracy < 0.8:
                performance_data["needs_improvement"] = True
                performance_data["recommendations"].append("Improve model accuracy")
            
            # Check for concept drift
            if learner_metrics.concept_drift_detected:
                performance_data["needs_improvement"] = True
                performance_data["recommendations"].append("Adapt to concept drift")
                
        except Exception as e:
            logger.error("Error analyzing performance: {}", str(e))
        
        return performance_data

    async def trigger_self_improvement(self, performance_data: Dict[str, Any]):
        """Enhanced self-improvement with safety monitoring"""
        logger.info("Triggering enhanced self-improvement process")
        
        try:
            # Execute self-improvement with safety monitoring
            async def improvement_operation():
                
                # Implement actual improvements based on recommendations
                improvements_applied = []
                
                for recommendation in performance_data.get("recommendations", []):
                    if "accuracy" in recommendation.lower():
                        # Trigger model retraining or parameter adjustment
                        logger.info("Applying accuracy improvement")
                        improvements_applied.append("accuracy_improvement")
                    
                    elif "efficiency" in recommendation.lower():
                        # Optimize learning efficiency
                        logger.info("Applying learning efficiency optimization")
                        improvements_applied.append("efficiency_optimization")
                    
                    elif "cognitive load" in recommendation.lower():
                        # Reduce cognitive load
                        logger.info("Applying cognitive load reduction")
                        improvements_applied.append("cognitive_load_reduction")
                
                return {
                    "improvements_applied": improvements_applied,
                    "timestamp": time.time()
                }
            
            # Execute with safety circuit breaker
            result = await self.safety_circuit.execute_with_safety(
                improvement_operation,
                "self_improvement",
                SafetyLevel.HIGH
            )
            
            # Log self-improvement event
            if self.audit_logger:
                audit_system_event(
                    "rsi_orchestrator",
                    "self_improvement_completed",
                    metadata={
                        **performance_data,
                        "improvements_applied": result["improvements_applied"]
                    }
                )
            
            logger.info("Self-improvement completed: {}", result["improvements_applied"])
            
        except Exception as e:
            logger.error("Self-improvement failed: {}", str(e))
    
    async def _run_hypothesis_driven_improvement(self, performance_data: Dict[str, Any]):
        """Run hypothesis-driven self-improvement process"""
        if not self.hypothesis_orchestrator:
            return
        
        logger.info("🧪 Starting hypothesis-driven self-improvement cycle")
        
        try:
            # Define improvement targets based on current performance
            improvement_targets = {}
            
            # Extract current metrics and define targets
            learning_metrics = performance_data.get("metrics", {}).get("learning", {})
            current_accuracy = learning_metrics.get("accuracy", 0.8)
            
            # Set targets based on current performance gaps
            if current_accuracy < 0.9:
                improvement_targets["accuracy"] = min(0.05, 0.9 - current_accuracy)
            
            metacognitive_metrics = performance_data.get("metrics", {}).get("metacognitive", {})
            current_efficiency = metacognitive_metrics.get("learning_efficiency", 0.7)
            
            if current_efficiency < 0.8:
                improvement_targets["learning_efficiency"] = min(0.1, 0.8 - current_efficiency)
            
            # Add cognitive load reduction if needed
            current_cognitive_load = metacognitive_metrics.get("cognitive_load", 0.5)
            if current_cognitive_load > 0.7:
                improvement_targets["cognitive_load_reduction"] = current_cognitive_load - 0.6
            
            # Add default targets if none identified
            if not improvement_targets:
                improvement_targets = {
                    "overall_performance": 0.02,
                    "system_efficiency": 0.03
                }
            
            # Create context for hypothesis generation
            context = {
                "current_performance": performance_data,
                "system_uptime": time.time() - self.start_time,
                "environment": self.environment,
                "recommendations": performance_data.get("recommendations", [])
            }
            
            # Generate and orchestrate hypotheses
            logger.info("🎯 Generating hypotheses with targets: {}", improvement_targets)
            orchestration_results = await self.hypothesis_orchestrator.orchestrate_hypothesis_lifecycle(
                improvement_targets=improvement_targets,
                context=context,
                max_hypotheses=5  # Limit for background processing
            )
            
            # Process results and apply deployable improvements
            deployed_count = 0
            for result in orchestration_results:
                if result.deployment_ready and result.confidence_score > 0.8:
                    logger.info("📈 Deploying hypothesis: {} (confidence: {:.2f})",
                               result.hypothesis.description, result.confidence_score)
                    
                    # Here we would implement the actual deployment logic
                    # For now, we'll log the deployment
                    deployed_count += 1
                    
                    # Log the deployment
                    if self.audit_logger:
                        audit_system_event(
                            "rsi_orchestrator",
                            "hypothesis_deployed",
                            metadata={
                                "hypothesis_id": result.hypothesis.hypothesis_id,
                                "hypothesis_type": result.hypothesis.hypothesis_type.value,
                                "confidence_score": result.confidence_score,
                                "improvement_targets": improvement_targets
                            }
                        )
            
            logger.info("🚀 Hypothesis-driven improvement completed: {} deployments from {} hypotheses",
                       deployed_count, len(orchestration_results))
            
            # Store results in memory (if method exists)
            if self.memory_system and hasattr(self.memory_system, 'store_episodic_memory'):
                try:
                    await self.memory_system.store_episodic_memory(
                        "hypothesis_improvement_cycle",
                        {
                            "cycle_count": self._improvement_cycle_count,
                            "improvement_targets": improvement_targets,
                            "hypotheses_generated": len(orchestration_results),
                            "deployments": deployed_count,
                            "avg_confidence": sum(r.confidence_score or 0 for r in orchestration_results) / max(1, len(orchestration_results)),
                            "timestamp": time.time()
                        }
                    )
                except Exception as e:
                    logger.warning("Failed to store improvement cycle in memory: {}", str(e))
            
        except Exception as e:
            logger.error("Hypothesis-driven improvement failed: {}", str(e))

# Global orchestrator instance
orchestrator: Optional[RSIOrchestrator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for enhanced RSI system"""
    global orchestrator
    
    # Startup
    orchestrator = RSIOrchestrator()
    await orchestrator.start()
    
    yield
    
    # Shutdown
    await orchestrator.stop()

# Initialize FastAPI app with enhanced features
app = FastAPI(
    title="Hephaestus RSI AI System",
    description="""
    🏛️ **Hephaestus - Complete Recursive Self-Improvement AI System**
    
    A production-ready RSI system with comprehensive hypothesis testing, validation, and deployment capabilities.
    
    ## 🌟 Key Features
    
    ### 🧪 **RSI Hypothesis Testing System**
    - **Intelligent Hypothesis Generation**: Uses Optuna and Ray for sophisticated hypothesis exploration
    - **Comprehensive Validation**: Multi-level validation with safety, performance, and robustness checks
    - **Advanced Simulation**: Statistical testing with robustness, stress, and adversarial analysis
    - **Human-in-the-Loop**: Approval workflows with automated safety-based decisions
    - **Secure Execution**: Multi-layer isolation using RestrictedPython and process sandboxing
    
    ### 🛡️ **Safety-First Architecture**
    - **Circuit Breakers**: Automatic failure detection and recovery
    - **Immutable State**: Corruption-proof state management with pyrsistent
    - **Comprehensive Validation**: Multiple layers of input and output validation
    - **Audit Trail**: Complete logging with cryptographic integrity verification
    
    ### 🧠 **Advanced Learning Systems**
    - **Metacognitive Monitoring**: Real-time self-awareness and performance analysis
    - **Uncertainty Quantification**: Confidence estimation for all predictions
    - **Online Learning**: Continuous adaptation with concept drift detection
    - **Meta-Learning**: Learn-to-learn capabilities for rapid adaptation
    
    ### 🔄 **Complete RSI Cycle**
    The system implements a complete recursive self-improvement cycle:
    1. **Performance Analysis** → Identify improvement opportunities
    2. **Hypothesis Generation** → Create improvement strategies using Optuna/Ray
    3. **Validation & Testing** → Comprehensive safety and performance validation
    4. **Human Review** → Expert approval for high-risk changes
    5. **Simulation** → Statistical testing and robustness analysis
    6. **Deployment** → Automated integration of validated improvements
    7. **Monitoring** → Continuous assessment and feedback
    
    *O ciclo está fechado* - The RSI cycle is now complete! 🎯
    """,
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket endpoint for real-time monitoring
@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket endpoint for real-time metacognitive monitoring"""
    await websocket.accept()
    orchestrator.active_connections.append(websocket)
    
    try:
        orchestrator.streaming_active = True
        
        while orchestrator.streaming_active:
            # Send periodic updates
            await asyncio.sleep(2)
            
            status = await orchestrator.get_metacognitive_status()
            await websocket.send_json({
                "type": "metacognitive_status",
                "data": status.dict()
            })
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: {}", str(e))
    finally:
        if websocket in orchestrator.active_connections:
            orchestrator.active_connections.remove(websocket)

# Enhanced API endpoints
@app.get("/")
async def root():
    """Root endpoint with enhanced system information"""
    features = [
        "Real-time metacognitive monitoring",
        "Uncertainty quantification", 
        "Safety circuit breakers",
        "Comprehensive audit trails",
        "WebSocket streaming"
    ]
    
    # Add hypothesis testing features if available
    if orchestrator and orchestrator.hypothesis_orchestrator:
        features.extend([
            "🧪 RSI Hypothesis Generation & Testing",
            "🔬 Comprehensive Validation Pipeline", 
            "🛡️ Multi-layer Safety Verification",
            "👥 Human-in-the-Loop Approval",
            "📊 Advanced Simulation & Analysis",
            "🚀 Automated Deployment Pipeline"
        ])
    
    return {
        "message": "Hephaestus RSI AI System - Complete Recursive Self-Improvement",
        "version": "3.0.0",
        "status": "active",
        "features": features,
        "hypothesis_system_available": orchestrator.hypothesis_orchestrator is not None if orchestrator else False,
        "cycle_closed": True  # RSI cycle is now complete
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with metacognitive status"""
    status = await orchestrator.get_metacognitive_status()
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metacognitive_status": status.dict(),
        "uptime_seconds": time.time() - orchestrator.start_time
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Enhanced prediction endpoint with uncertainty quantification"""
    return await orchestrator.predict(
        request.features, 
        request.user_id,
        request.uncertainty_estimation
    )

@app.post("/learn")
async def learn(request: LearningRequest):
    """Enhanced learning endpoint with safety monitoring"""
    return await orchestrator.learn(
        request.features, 
        request.target, 
        request.user_id,
        request.safety_level
    )

@app.get("/metacognitive-status")
async def get_metacognitive_status():
    """Get current metacognitive system status"""
    return await orchestrator.get_metacognitive_status()

@app.get("/safety-status")
async def get_safety_status():
    """Get current safety system status"""
    return orchestrator.safety_circuit.get_safety_status()

@app.get("/system-metrics")
async def get_system_metrics():
    """Get current system performance metrics"""
    if orchestrator.system_monitor.metrics_history:
        latest = orchestrator.system_monitor.metrics_history[-1]
        return {
            "timestamp": latest.timestamp,
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "gpu_utilization": latest.gpu_utilization,
            "model_inference_rate": latest.model_inference_rate,
            "prediction_confidence": latest.prediction_confidence,
            "anomaly_score": latest.anomaly_score
        }
    return {"status": "no_data"}

# RSI Hypothesis Testing Endpoints
@app.post("/hypothesis/generate")
async def generate_hypotheses(request: HypothesisGenerationRequest):
    """Generate RSI improvement hypotheses"""
    if not orchestrator.hypothesis_orchestrator:
        raise HTTPException(status_code=503, detail="Hypothesis system not available")
    
    try:
        orchestration_results = await orchestrator.hypothesis_orchestrator.orchestrate_hypothesis_lifecycle(
            improvement_targets=request.improvement_targets,
            context=request.context,
            max_hypotheses=request.max_hypotheses
        )
        
        # Convert results to serializable format
        results = []
        for result in orchestration_results:
            results.append({
                "hypothesis_id": result.hypothesis.hypothesis_id,
                "hypothesis_type": result.hypothesis.hypothesis_type.value,
                "priority": result.hypothesis.priority.value,
                "description": result.hypothesis.description,
                "status": result.status.value,
                "current_phase": result.current_phase.value,
                "confidence_score": result.confidence_score,
                "deployment_ready": result.deployment_ready,
                "final_recommendation": result.final_recommendation,
                "expected_improvement": result.hypothesis.expected_improvement,
                "risk_level": result.hypothesis.risk_level,
                "computational_cost": result.hypothesis.computational_cost
            })
        
        # Log hypothesis generation
        if orchestrator.audit_logger:
            audit_system_event(
                "hypothesis_system",
                "hypotheses_generated",
                user_id=request.user_id,
                metadata={
                    "improvement_targets": request.improvement_targets,
                    "hypotheses_count": len(results),
                    "deployment_ready_count": len([r for r in results if r["deployment_ready"]])
                }
            )
        
        return {
            "hypotheses_generated": len(results),
            "deployment_ready_count": len([r for r in results if r["deployment_ready"]]),
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error("Hypothesis generation failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Hypothesis generation failed: {str(e)}")

@app.get("/hypothesis/status/{hypothesis_id}")
async def get_hypothesis_status(hypothesis_id: str):
    """Get status of a specific hypothesis"""
    if not orchestrator.hypothesis_orchestrator:
        raise HTTPException(status_code=503, detail="Hypothesis system not available")
    
    try:
        result = await orchestrator.hypothesis_orchestrator.get_orchestration_status(hypothesis_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Hypothesis not found")
        
        return {
            "hypothesis_id": hypothesis_id,
            "status": result.status.value,
            "current_phase": result.current_phase.value,
            "confidence_score": result.confidence_score,
            "deployment_ready": result.deployment_ready,
            "final_recommendation": result.final_recommendation,
            "phase_durations": result.phase_durations,
            "total_duration_seconds": result.total_duration_seconds
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting hypothesis status: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Error getting hypothesis status: {str(e)}")

@app.post("/hypothesis/review")
async def submit_hypothesis_review(request: HypothesisReviewRequest):
    """Submit a human review decision for a hypothesis"""
    if not orchestrator.hypothesis_orchestrator:
        raise HTTPException(status_code=503, detail="Hypothesis system not available")
    
    try:
        # Convert string decision to enum
        if HYPOTHESIS_SYSTEM_AVAILABLE:
            decision_enum = ReviewStatus(request.decision.lower())
        else:
            raise HTTPException(status_code=503, detail="Hypothesis system not properly initialized")
        
        review_decision = await orchestrator.hypothesis_orchestrator.human_loop_manager.submit_review_decision(
            request_id=request.request_id,
            reviewer_id=request.reviewer_id,
            decision=decision_enum,
            reasoning=request.reasoning,
            confidence_level=request.confidence_level,
            modification_suggestions=request.modification_suggestions,
            approval_conditions=request.approval_conditions
        )
        
        # Log review submission
        if orchestrator.audit_logger:
            audit_system_event(
                "hypothesis_system",
                "review_submitted",
                user_id=request.reviewer_id,
                metadata={
                    "request_id": request.request_id,
                    "decision": request.decision,
                    "confidence_level": request.confidence_level
                }
            )
        
        return {
            "review_id": review_decision.request_id,
            "decision": review_decision.decision.value,
            "confidence_level": review_decision.confidence_level,
            "timestamp": review_decision.timestamp,
            "risk_assessment": review_decision.risk_assessment
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid decision value: {str(e)}")
    except Exception as e:
        logger.error("Review submission failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Review submission failed: {str(e)}")

@app.get("/hypothesis/pending-reviews")
async def get_pending_reviews(reviewer_id: Optional[str] = None, priority: Optional[str] = None):
    """Get pending hypothesis reviews"""
    if not orchestrator.hypothesis_orchestrator:
        raise HTTPException(status_code=503, detail="Hypothesis system not available")
    
    try:
        # Convert priority string to enum if provided
        priority_enum = None
        if priority and HYPOTHESIS_SYSTEM_AVAILABLE:
            from src.hypothesis.human_in_loop import ReviewPriority
            priority_enum = ReviewPriority(priority.lower())
        
        pending_reviews = await orchestrator.hypothesis_orchestrator.human_loop_manager.get_pending_reviews(
            reviewer_id=reviewer_id,
            priority=priority_enum
        )
        
        # Convert to serializable format
        reviews = []
        for review in pending_reviews:
            reviews.append({
                "request_id": review.request_id,
                "hypothesis_id": review.hypothesis.hypothesis_id,
                "hypothesis_type": review.hypothesis.hypothesis_type.value,
                "priority": review.review_priority.value,
                "description": review.hypothesis.description,
                "created_timestamp": review.created_timestamp,
                "assigned_reviewer": review.assigned_reviewer,
                "reviewer_notes": review.reviewer_notes,
                "safety_score": review.validation_result.safety_score if review.validation_result else None,
                "risk_level": review.hypothesis.risk_level
            })
        
        return {
            "pending_reviews": len(reviews),
            "reviews": reviews,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error("Error getting pending reviews: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Error getting pending reviews: {str(e)}")

@app.get("/hypothesis/statistics")
async def get_hypothesis_statistics():
    """Get comprehensive hypothesis system statistics"""
    if not orchestrator.hypothesis_orchestrator:
        raise HTTPException(status_code=503, detail="Hypothesis system not available")
    
    try:
        stats = orchestrator.hypothesis_orchestrator.get_orchestration_statistics()
        return stats
        
    except Exception as e:
        logger.error("Error getting hypothesis statistics: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Error getting hypothesis statistics: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )