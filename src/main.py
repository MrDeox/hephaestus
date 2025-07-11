"""
Main RSI System Orchestrator and API.
Integrates all components into a cohesive Recursive Self-Improvement system.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import all RSI components
from .core.state import RSIState, StateManager, update_configuration, update_model_weights
# Memory system imports
try:
    from .memory import RSIMemoryHierarchy, RSIMemoryConfig
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError:
    MEMORY_SYSTEM_AVAILABLE = False
    print("⚠️  Memory system not available")
try:
    from .core.model_versioning import ModelVersionManager, ModelMetadata, ModelType, ModelStatus
    MODEL_VERSIONING_AVAILABLE = True
except ImportError:
    try:
        from .core.simple_model_versioning import SimpleModelVersionManager, ModelMetadata, ModelType, ModelStatus
        MODEL_VERSIONING_AVAILABLE = True
        print("✅ Using simple model versioning (MLflow alternative)")
    except ImportError:
        MODEL_VERSIONING_AVAILABLE = False
        print("⚠️  Model versioning not available")
from .learning.online_learning import OnlineLearner, create_ensemble_learner, ConceptDriftType
# Advanced learning imports with fallback handling
try:
    from .learning.meta_learning import create_meta_learning_system, MetaLearningAlgorithm
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False
    print("⚠️  Meta-learning not available - missing PyTorch dependencies")

try:
    from .learning.lightning_orchestrator import create_lightning_orchestrator, TaskType
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("⚠️  Lightning orchestrator not available - missing PyTorch Lightning dependencies")

try:
    from .learning.reinforcement_learning import create_rl_system, RLAlgorithm, RSITaskType
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("⚠️  Reinforcement learning not available - missing Stable-Baselines3 dependencies")

try:
    from .learning.continual_learning import create_continual_learning_system, ContinualLearningStrategy
    CONTINUAL_LEARNING_AVAILABLE = True
except ImportError:
    CONTINUAL_LEARNING_AVAILABLE = False
    print("⚠️  Continual learning not available - missing PyTorch dependencies")

try:
    from .optimization.optuna_optimizer import create_optuna_optimizer, OptimizationObjective
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna optimizer not available - missing Optuna dependencies")

try:
    from .optimization.ray_tune_optimizer import create_ray_tune_orchestrator, SearchAlgorithm
    RAY_TUNE_AVAILABLE = True
    print("✅ Ray Tune available")
except ImportError as e:
    RAY_TUNE_AVAILABLE = False
    print(f"⚠️  Ray Tune not available - {e}")
from .validation.validators import RSIValidator, create_strict_validator
from .safety.circuits import RSICircuitManager, CircuitConfig
from .security.sandbox import RSISandbox, create_production_sandbox
from .monitoring.telemetry import initialize_telemetry, get_telemetry_provider, trace_operation
from .monitoring.anomaly_detection import BehavioralMonitor, create_behavioral_monitor
from .monitoring.audit_logger import initialize_audit_logger, get_audit_logger, audit_user_action, audit_system_event

from loguru import logger
import logging

# Set up logging for import warnings
logging.basicConfig(level=logging.WARNING)


# API Models
class LearningRequest(BaseModel):
    """Request model for learning operations."""
    features: Dict[str, Any] = Field(..., description="Feature dictionary")
    target: Any = Field(..., description="Target value")
    user_id: Optional[str] = Field(None, description="User ID for audit trail")


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: Dict[str, Any] = Field(..., description="Feature dictionary")
    user_id: Optional[str] = Field(None, description="User ID for audit trail")


class CodeExecutionRequest(BaseModel):
    """Request model for code execution."""
    code: str = Field(..., description="Python code to execute")
    timeout_seconds: Optional[int] = Field(60, description="Execution timeout")
    user_id: Optional[str] = Field(None, description="User ID for audit trail")


class ModelCreateRequest(BaseModel):
    """Request model for model creation."""
    name: str = Field(..., description="Model name")
    model_type: ModelType = Field(..., description="Model type")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters")
    user_id: Optional[str] = Field(None, description="User ID for audit trail")


class RSIOrchestrator:
    """
    Main orchestrator for the RSI system.
    Coordinates all components and manages system lifecycle.
    """
    
    def __init__(
        self,
        environment: str = "production",
        enable_monitoring: bool = True,
        enable_audit: bool = True,
        mlflow_uri: str = "sqlite:///mlflow.db",
        db_url: str = "sqlite:///rsi_system.db"
    ):
        self.environment = environment
        self.enable_monitoring = enable_monitoring
        self.enable_audit = enable_audit
        
        # Initialize core components
        self.state_manager = self._initialize_state_manager()
        self.validator = self._initialize_validator()
        self.circuit_manager = self._initialize_circuit_manager()
        self.sandbox = self._initialize_sandbox()
        self.model_manager = self._initialize_model_manager(mlflow_uri, db_url)
        self.online_learner = self._initialize_online_learner()
        
        # Initialize monitoring components
        if enable_monitoring:
            self.telemetry = self._initialize_telemetry()
            self.anomaly_monitor = self._initialize_anomaly_monitor()
        else:
            self.telemetry = None
            self.anomaly_monitor = None
        
        # Initialize advanced learning components
        self.meta_learning_system = self._initialize_meta_learning()
        self.lightning_orchestrator = self._initialize_lightning_orchestrator()
        self.rl_system = self._initialize_rl_system()
        self.continual_learning_system = self._initialize_continual_learning()
        self.optuna_optimizer = self._initialize_optuna_optimizer()
        self.ray_tune_orchestrator = self._initialize_ray_tune_orchestrator()
        
        # Initialize memory system
        self.memory_system = self._initialize_memory_system()
        
        # Initialize audit logging
        if enable_audit:
            self.audit_logger = self._initialize_audit_logger()
        else:
            self.audit_logger = None
        
        # System state
        self.is_running = False
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info(f"RSI Orchestrator initialized for {environment} environment")
    
    def _initialize_state_manager(self) -> StateManager:
        """Initialize state manager with default RSI state."""
        initial_state = RSIState(
            configuration={"environment": self.environment},
            safety_status={"initialized": True, "safety_checks_enabled": True}
        )
        return StateManager(initial_state)
    
    def _initialize_validator(self) -> RSIValidator:
        """Initialize validator based on environment."""
        if self.environment == "production":
            return create_strict_validator()
        else:
            from .validation.validators import create_development_validator
            return create_development_validator()
    
    def _initialize_circuit_manager(self) -> RSICircuitManager:
        """Initialize circuit breaker manager."""
        circuit_manager = RSICircuitManager()
        
        # Create standard circuits
        circuit_manager.create_circuit(
            "model_operations",
            CircuitConfig(fail_max=5, reset_timeout=300)
        )
        circuit_manager.create_circuit(
            "learning_operations",
            CircuitConfig(fail_max=10, reset_timeout=120)
        )
        circuit_manager.create_circuit(
            "code_execution",
            CircuitConfig(fail_max=3, reset_timeout=600)
        )
        
        return circuit_manager
    
    def _initialize_sandbox(self) -> RSISandbox:
        """Initialize secure sandbox."""
        if self.environment == "production":
            return create_production_sandbox()
        else:
            from .security.sandbox import create_development_sandbox
            return create_development_sandbox()
    
    def _initialize_model_manager(self, mlflow_uri: str, db_url: str):
        """Initialize model version manager."""
        if not MODEL_VERSIONING_AVAILABLE:
            return None
        try:
            # Try MLflow first, then fallback to simple versioning
            try:
                return ModelVersionManager(
                    mlflow_tracking_uri=mlflow_uri,
                    database_url=db_url,
                    validator=self.validator
                )
            except:
                # Use simple model versioning as fallback
                return SimpleModelVersionManager(
                    db_path=db_url,
                    models_dir="models/"
                )
        except Exception as e:
            logger.warning(f"Model versioning initialization failed: {e}")
            return None
    
    def _initialize_online_learner(self) -> OnlineLearner:
        """Initialize online learning component."""
        return create_ensemble_learner(
            state_manager=self.state_manager,
            validator=self.validator
        )
    
    def _initialize_telemetry(self):
        """Initialize telemetry system."""
        # Disable telemetry for now to avoid import issues
        return None
    
    def _initialize_anomaly_monitor(self) -> BehavioralMonitor:
        """Initialize anomaly detection."""
        monitor = create_behavioral_monitor(
            state_manager=self.state_manager,
            algorithm="ecod" if self.environment == "production" else "iforest"
        )
        
        # Add alert callback
        def alert_callback(alert):
            logger.warning(f"Anomaly detected: {alert.description}")
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    f"Anomaly detected: {alert.anomaly_type.value}",
                    severity=alert.severity.value,
                    details=alert.to_dict()
                )
        
        monitor.add_alert_callback(alert_callback)
        return monitor
    
    def _initialize_audit_logger(self):
        """Initialize audit logging."""
        return initialize_audit_logger(
            log_directory=f"./logs/{self.environment}",
            encryption_key="rsi_audit_key" if self.environment == "production" else None
        )
    
    def _initialize_meta_learning(self):
        """Initialize meta-learning system."""
        if not META_LEARNING_AVAILABLE:
            return None
        try:
            from .learning.meta_learning import MetaLearningAlgorithm
            return create_meta_learning_system(
                algorithm=MetaLearningAlgorithm.MAML,
                state_manager=self.state_manager,
                validator=self.validator,
                num_ways=5,
                num_shots=5,
                adaptation_steps=3
            )
        except Exception as e:
            logger.warning(f"Meta-learning initialization failed: {e}")
            return None
    
    def _initialize_lightning_orchestrator(self):
        """Initialize PyTorch Lightning orchestrator."""
        if not LIGHTNING_AVAILABLE:
            return None
        try:
            from .learning.lightning_orchestrator import TaskType
            task_configs = {
                'classification_task': {
                    'type': TaskType.CLASSIFICATION,
                    'input_dim': 20,
                    'output_dim': 10,
                    'weight': 1.0
                }
            }
            return create_lightning_orchestrator(
                task_configs=task_configs,
                state_manager=self.state_manager,
                validator=self.validator
            )
        except Exception as e:
            logger.warning(f"Lightning orchestrator initialization failed: {e}")
            return None
    
    def _initialize_rl_system(self):
        """Initialize reinforcement learning system."""
        if not RL_AVAILABLE:
            return None
        try:
            from .learning.reinforcement_learning import RLAlgorithm, RSITaskType
            return create_rl_system(
                task_type=RSITaskType.HYPERPARAMETER_OPTIMIZATION,
                algorithm=RLAlgorithm.PPO,
                state_manager=self.state_manager,
                validator=self.validator,
                total_timesteps=10000
            )
        except Exception as e:
            logger.warning(f"RL system initialization failed: {e}")
            return None
    
    def _initialize_continual_learning(self):
        """Initialize continual learning system."""
        if not CONTINUAL_LEARNING_AVAILABLE:
            return None
        try:
            from .learning.continual_learning import ContinualLearningStrategy
            return create_continual_learning_system(
                strategy=ContinualLearningStrategy.EWC,
                input_size=20,
                output_size=10,
                state_manager=self.state_manager,
                validator=self.validator
            )
        except Exception as e:
            logger.warning(f"Continual learning initialization failed: {e}")
            return None
    
    def _initialize_optuna_optimizer(self):
        """Initialize Optuna optimizer."""
        if not OPTUNA_AVAILABLE:
            return None
        try:
            from .optimization.optuna_optimizer import OptimizationObjective
            return create_optuna_optimizer(
                study_name="rsi_optimization",
                objective_type=OptimizationObjective.MAXIMIZE,
                n_trials=50,
                state_manager=self.state_manager,
                validator=self.validator
            )
        except Exception as e:
            logger.warning(f"Optuna optimizer initialization failed: {e}")
            return None
    
    def _initialize_ray_tune_orchestrator(self):
        """Initialize Ray Tune orchestrator."""
        if not RAY_TUNE_AVAILABLE:
            return None
        try:
            from .optimization.ray_tune_optimizer import SearchAlgorithm
            return create_ray_tune_orchestrator(
                experiment_name="rsi_distributed_optimization",
                search_algorithm=SearchAlgorithm.BAYESOPT,
                num_samples=20,
                state_manager=self.state_manager,
                validator=self.validator
            )
        except Exception as e:
            logger.warning(f"Ray Tune orchestrator initialization failed: {e}")
            return None
    
    def _initialize_memory_system(self):
        """Initialize hierarchical memory system."""
        if not MEMORY_SYSTEM_AVAILABLE:
            logger.warning("Memory system not available")
            return None
        
        try:
            # Configure memory system for production
            memory_config = RSIMemoryConfig(
                working_memory_capacity=20000,
                semantic_memory_backend="networkx",
                episodic_memory_backend="eventsourcing",
                vector_db_type="chroma",
                graph_db_type="networkx",
                embedding_dimension=768,
                replay_buffer_size=5000,
                ewc_lambda=0.4,
                meta_learning_enabled=True,
                ann_algorithm="hnsw",
                index_ef_construction=200,
                index_m=16,
                ray_object_store_memory="8GB",
                redis_cluster_nodes=1,
                compression_algorithm="blosc",
                max_memory_usage_gb=32,
                cache_ttl_seconds=3600,
                monitoring_enabled=self.enable_monitoring,
                max_working_memory_size=100000,
                memory_consolidation_threshold=0.8,
                automatic_cleanup_enabled=True
            )
            
            memory_system = RSIMemoryHierarchy(memory_config)
            logger.info("✅ Memory system initialized successfully")
            return memory_system
            
        except Exception as e:
            logger.error(f"Memory system initialization failed: {e}")
            return None
    
    async def start(self):
        """Start the RSI system."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring
        if self.anomaly_monitor:
            self.anomaly_monitor.start_monitoring()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._self_improvement_loop())
        ]
        
        # Log system startup
        if self.audit_logger:
            self.audit_logger.log_system_event(
                "rsi_orchestrator",
                "system_started",
                metadata={"environment": self.environment}
            )
        
        logger.info("RSI system started successfully")
    
    async def stop(self):
        """Stop the RSI system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop monitoring
        if self.anomaly_monitor:
            self.anomaly_monitor.stop_monitoring()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Log system shutdown
        if self.audit_logger:
            self.audit_logger.log_system_event(
                "rsi_orchestrator",
                "system_stopped"
            )
        
        logger.info("RSI system stopped")
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self.is_running:
            try:
                # Check system health
                health_data = await self.get_system_health()
                
                # Update telemetry
                if self.telemetry:
                    self.telemetry.update_resource_metrics()
                
                # Check for issues
                if health_data["overall_status"] != "healthy":
                    logger.warning(f"System health issue: {health_data['issues']}")
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while self.is_running:
            try:
                # Collect system metrics
                if self.anomaly_monitor:
                    # Get current metrics
                    import psutil
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    
                    # Feed to anomaly detector
                    self.anomaly_monitor.collect_resource_data({
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_used_gb": memory.used / (1024**3)
                    })
                
                await asyncio.sleep(60)  # Metrics collection every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _self_improvement_loop(self):
        """Background self-improvement loop."""
        while self.is_running:
            try:
                # Analyze system performance
                performance_data = await self.analyze_performance()
                
                # Check if improvements are needed
                if performance_data.get("needs_improvement", False):
                    await self.trigger_self_improvement(performance_data)
                
                await asyncio.sleep(300)  # Self-improvement check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Self-improvement loop error: {e}")
                await asyncio.sleep(300)
    
    @trace_operation("predict")
    async def predict(self, features: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Make a prediction using the online learner."""
        try:
            # Basic input validation
            if not isinstance(features, dict):
                raise ValueError("Features must be a dictionary")
            if not features:
                raise ValueError("Features cannot be empty")
            
            # Make prediction
            prediction, confidence = await self.online_learner.ensemble_predict(features)
            
            # Log prediction
            if self.audit_logger:
                audit_user_action(
                    user_id or "system",
                    "predict",
                    "online_model",
                    success=True,
                    details={"confidence": confidence}
                )
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            if self.audit_logger:
                audit_user_action(
                    user_id or "system",
                    "predict",
                    "online_model",
                    success=False,
                    details={"error": str(e)}
                )
            raise HTTPException(status_code=500, detail=str(e))
    
    @trace_operation("learn")
    async def learn(self, features: Dict[str, Any], target: Any, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Learn from a new example with memory integration."""
        try:
            # Learn from example
            metrics_list = await self.online_learner.ensemble_learn(features, target)
            
            # Get average metrics from ensemble
            if metrics_list:
                avg_accuracy = sum(m.accuracy for m in metrics_list) / len(metrics_list)
                avg_samples = sum(m.samples_processed for m in metrics_list) / len(metrics_list)
                any_drift = any(m.concept_drift_detected for m in metrics_list)
            else:
                avg_accuracy = 0.0
                avg_samples = 0
                any_drift = False
            
            # Store learning experience in memory system
            if self.memory_system:
                learning_experience = {
                    "event": "learning_session",
                    "description": f"Learned from features with accuracy {avg_accuracy:.3f}",
                    "context": {
                        "features": features,
                        "target": target,
                        "accuracy": avg_accuracy,
                        "samples_processed": int(avg_samples),
                        "concept_drift": any_drift,
                        "user_id": user_id or "system"
                    },
                    "importance": min(1.0, avg_accuracy),
                    "emotions": {"satisfaction": avg_accuracy},
                    "tags": ["learning", "online_model", "rsi"],
                    "outcome": {
                        "success": True,
                        "performance_improvement": avg_accuracy > 0.7
                    }
                }
                
                try:
                    await self.memory_system.store_information(learning_experience, memory_type="episodic")
                    
                    # If concept drift detected, store as knowledge
                    if any_drift:
                        drift_knowledge = {
                            "concept": "concept_drift_detection",
                            "description": f"Concept drift detected during learning session",
                            "type": "learning_insight",
                            "confidence": 0.9,
                            "context": features,
                            "source": "online_learning"
                        }
                        await self.memory_system.store_information(drift_knowledge, memory_type="semantic")
                        
                except Exception as e:
                    logger.warning(f"Failed to store learning experience in memory: {e}")
            
            # Log learning
            if self.audit_logger:
                audit_user_action(
                    user_id or "system",
                    "learn",
                    "online_model",
                    success=True,
                    details={"accuracy": avg_accuracy}
                )
            
            return {
                "accuracy": avg_accuracy,
                "samples_processed": int(avg_samples),
                "concept_drift_detected": any_drift,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_stored": self.memory_system is not None
            }
            
        except Exception as e:
            logger.error(f"Learning error: {e}")
            if self.audit_logger:
                audit_user_action(
                    user_id or "system",
                    "learn",
                    "online_model",
                    success=False,
                    details={"error": str(e)}
                )
            raise HTTPException(status_code=500, detail=str(e))
    
    @trace_operation("execute_code")
    async def execute_code(self, code: str, timeout_seconds: int = 60, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute code safely in sandbox."""
        try:
            # Execute code
            result = self.sandbox.execute(code, timeout_seconds)
            
            # Log execution
            if self.audit_logger:
                audit_user_action(
                    user_id or "system",
                    "execute_code",
                    "sandbox",
                    success=result.status.value == "success",
                    details={
                        "status": result.status.value,
                        "execution_time_ms": result.execution_time_ms
                    }
                )
            
            return {
                "status": result.status.value,
                "output": result.output,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
                "security_violations": result.security_violations,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            if self.audit_logger:
                audit_user_action(
                    user_id or "system",
                    "execute_code",
                    "sandbox",
                    success=False,
                    details={"error": str(e)}
                )
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_data = {
            "overall_status": "healthy",
            "issues": [],
            "components": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Check circuit breakers
        circuit_health = self.circuit_manager.health_check()
        health_data["components"]["circuits"] = circuit_health
        
        if circuit_health["open_circuits"] > 0:
            health_data["overall_status"] = "degraded"
            health_data["issues"].append("Circuit breakers open")
        
        # Check anomaly alerts
        if self.anomaly_monitor:
            active_alerts = self.anomaly_monitor.get_active_alerts()
            health_data["components"]["anomaly_detection"] = {
                "active_alerts": len(active_alerts),
                "monitoring_active": self.anomaly_monitor.monitoring_active
            }
            
            if active_alerts:
                health_data["overall_status"] = "degraded"
                health_data["issues"].append(f"{len(active_alerts)} active anomaly alerts")
        
        # Check telemetry
        if self.telemetry:
            telemetry_health = self.telemetry.health_check()
            health_data["components"]["telemetry"] = telemetry_health
            
            if telemetry_health["status"] != "healthy":
                health_data["overall_status"] = "degraded"
                health_data["issues"].extend(telemetry_health["issues"])
        
        return health_data
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance for self-improvement."""
        performance_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "needs_improvement": False,
            "recommendations": []
        }
        
        # Get online learner metrics
        if hasattr(self.online_learner, 'current_metrics'):
            learner_metrics = self.online_learner.current_metrics
        else:
            # For ensemble orchestrator, get metrics from first learner
            if hasattr(self.online_learner, 'learners') and self.online_learner.learners:
                learner_metrics = self.online_learner.learners[0].current_metrics
            else:
                # Create default metrics
                from src.learning.online_learning import LearningMetrics
                learner_metrics = LearningMetrics(
                    accuracy=0.0,
                    loss=0.0,
                    samples_processed=0,
                    learning_rate=0.01,
                    concept_drift_detected=False,
                    drift_type=ConceptDriftType.NONE,
                    model_complexity=0,
                    prediction_confidence=0.0,
                    adaptation_speed=0.0,
                    timestamp=datetime.now(timezone.utc)
                )
        performance_data["metrics"]["learning"] = {
            "accuracy": learner_metrics.accuracy,
            "samples_processed": learner_metrics.samples_processed,
            "adaptation_speed": learner_metrics.adaptation_speed
        }
        
        # Check if accuracy is declining
        if learner_metrics.accuracy < 0.8:
            performance_data["needs_improvement"] = True
            performance_data["recommendations"].append("Improve model accuracy")
        
        # Check for concept drift
        if learner_metrics.concept_drift_detected:
            performance_data["needs_improvement"] = True
            performance_data["recommendations"].append("Adapt to concept drift")
        
        return performance_data
    
    async def trigger_self_improvement(self, performance_data: Dict[str, Any]):
        """Trigger self-improvement based on performance analysis."""
        logger.info("Triggering self-improvement process")
        
        # Log self-improvement trigger
        if self.audit_logger:
            audit_system_event(
                "rsi_orchestrator",
                "self_improvement_triggered",
                metadata=performance_data
            )
        
        # Implement self-improvement logic here
        # This could include:
        # - Retraining models
        # - Adjusting hyperparameters
        # - Updating system configuration
        # - Optimizing resource usage
        
        # For now, just log the event
        logger.info(f"Self-improvement recommendations: {performance_data['recommendations']}")


# Global orchestrator instance
orchestrator: Optional[RSIOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    global orchestrator
    
    # Startup
    orchestrator = RSIOrchestrator()
    await orchestrator.start()
    
    yield
    
    # Shutdown
    if orchestrator:
        await orchestrator.stop()


# Create FastAPI app
app = FastAPI(
    title="Hephaestus RSI System",
    description="Recursive Self-Improvement AI System",
    version="1.0.0",
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


# API Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hephaestus RSI System", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if orchestrator:
        return await orchestrator.get_system_health()
    return {"status": "initializing"}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make a prediction."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not ready")
    
    return await orchestrator.predict(request.features, request.user_id)


@app.post("/learn")
async def learn(request: LearningRequest):
    """Learn from new data."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not ready")
    
    return await orchestrator.learn(request.features, request.target, request.user_id)


@app.post("/execute")
async def execute_code(request: CodeExecutionRequest):
    """Execute code safely."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not ready")
    
    return await orchestrator.execute_code(
        request.code, 
        request.timeout_seconds or 60,
        request.user_id
    )


@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not ready")
    
    metrics = {}
    
    # Get circuit metrics
    metrics["circuits"] = orchestrator.circuit_manager.get_all_metrics()
    
    # Get anomaly detection metrics
    if orchestrator.anomaly_monitor:
        metrics["anomaly_detection"] = orchestrator.anomaly_monitor.get_monitoring_stats()
    
    # Get telemetry metrics
    if orchestrator.telemetry:
        metrics["telemetry"] = orchestrator.telemetry.get_metrics_snapshot()
    
    return metrics


@app.get("/alerts")
async def get_alerts():
    """Get active alerts."""
    if not orchestrator or not orchestrator.anomaly_monitor:
        raise HTTPException(status_code=503, detail="System not ready")
    
    active_alerts = orchestrator.anomaly_monitor.get_active_alerts()
    return {
        "active_alerts": [alert.to_dict() for alert in active_alerts],
        "count": len(active_alerts)
    }


@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert."""
    if not orchestrator or not orchestrator.anomaly_monitor:
        raise HTTPException(status_code=503, detail="System not ready")
    
    resolved = orchestrator.anomaly_monitor.resolve_alert(alert_id)
    
    if resolved:
        return {"message": "Alert resolved successfully"}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")


@app.get("/performance")
async def get_performance():
    """Get performance analysis."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not ready")
    
    return await orchestrator.analyze_performance()


@app.post("/self-improve")
async def trigger_self_improvement():
    """Manually trigger self-improvement."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not ready")
    
    performance_data = await orchestrator.analyze_performance()
    await orchestrator.trigger_self_improvement(performance_data)
    
    return {"message": "Self-improvement process triggered"}


# Memory System Endpoints

@app.get("/memory/status")
async def get_memory_status():
    """Get comprehensive memory system status."""
    if not orchestrator or not orchestrator.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not available")
    
    return await orchestrator.memory_system.get_memory_status()


@app.post("/memory/store")
async def store_information(
    information: Dict[str, Any],
    memory_type: str = "auto"
):
    """Store information in memory system."""
    if not orchestrator or not orchestrator.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not available")
    
    try:
        success = await orchestrator.memory_system.store_information(information, memory_type)
        return {
            "success": success,
            "message": f"Information stored in {memory_type} memory",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/retrieve")
async def retrieve_information(
    query: Dict[str, Any],
    memory_types: Optional[List[str]] = None
):
    """Retrieve information from memory system."""
    if not orchestrator or not orchestrator.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not available")
    
    try:
        results = await orchestrator.memory_system.retrieve_information(query, memory_types)
        return {
            "results": results,
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/consolidate")
async def consolidate_memory():
    """Trigger memory consolidation process."""
    if not orchestrator or not orchestrator.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not available")
    
    try:
        result = await orchestrator.memory_system.consolidate_memory()
        return {
            "consolidation_result": result,
            "message": "Memory consolidation completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/optimize")
async def optimize_memory():
    """Optimize memory system performance."""
    if not orchestrator or not orchestrator.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not available")
    
    try:
        result = await orchestrator.memory_system.optimize_memory()
        return {
            "optimization_result": result,
            "message": "Memory optimization completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/statistics")
async def get_memory_statistics():
    """Get detailed memory system statistics."""
    if not orchestrator or not orchestrator.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not available")
    
    try:
        stats = {
            "working_memory": orchestrator.memory_system.working_memory.get_stats(),
            "semantic_memory": orchestrator.memory_system.semantic_memory.get_stats(),
            "episodic_memory": orchestrator.memory_system.episodic_memory.get_stats(),
        }
        
        # Add procedural memory stats if available
        if orchestrator.memory_system.procedural_memory:
            stats["procedural_memory"] = orchestrator.memory_system.procedural_memory.get_stats()
        
        # Add retrieval engine stats if available
        if orchestrator.memory_system.retrieval_engine:
            stats["retrieval_engine"] = orchestrator.memory_system.retrieval_engine.get_stats()
        
        # Add memory manager stats if available
        if orchestrator.memory_system.memory_manager:
            stats["memory_manager"] = orchestrator.memory_system.memory_manager.get_stats()
        
        return {
            "statistics": stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )