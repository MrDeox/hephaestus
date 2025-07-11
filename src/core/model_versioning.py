"""
Model versioning and persistence layer for RSI system.
Provides comprehensive model lifecycle management with MLflow integration.
"""

import asyncio
import pickle
import json
import hashlib
import shutil
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import os

import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ..core.state import RSIState, StateManager
from ..validation.validators import RSIValidator
from ..monitoring.telemetry import trace_operation, record_safety_event
from ..safety.circuits import RSICircuitBreaker, create_database_circuit
from loguru import logger


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelType(str, Enum):
    """Types of models in the RSI system."""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    ENSEMBLE = "ensemble"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TRANSFORMER = "transformer"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_score: Optional[float] = None
    
    # Training information
    training_dataset_size: Optional[int] = None
    training_duration_seconds: Optional[float] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Validation information
    validation_accuracy: Optional[float] = None
    validation_dataset_size: Optional[int] = None
    cross_validation_scores: List[float] = field(default_factory=list)
    
    # Deployment information
    deployment_target: Optional[str] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage information
    parent_model_id: Optional[str] = None
    training_run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    
    # Safety and compliance
    safety_validation_passed: bool = False
    compliance_checks_passed: bool = False
    ethical_review_passed: bool = False
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    checksum: str = ""


Base = declarative_base()


class ModelRecord(Base):
    """Database model for persisting model metadata."""
    
    __tablename__ = "model_records"
    
    id = Column(Integer, primary_key=True)
    model_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    version = Column(String(100), nullable=False)
    model_type = Column(String(100), nullable=False)
    status = Column(String(100), nullable=False)
    
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    
    # Training information
    training_dataset_size = Column(Integer)
    training_duration_seconds = Column(Float)
    hyperparameters = Column(Text)  # JSON string
    
    # Validation information
    validation_accuracy = Column(Float)
    validation_dataset_size = Column(Integer)
    cross_validation_scores = Column(Text)  # JSON string
    
    # Deployment information
    deployment_target = Column(String(255))
    resource_requirements = Column(Text)  # JSON string
    
    # Lineage information
    parent_model_id = Column(String(255))
    training_run_id = Column(String(255))
    experiment_id = Column(String(255))
    
    # Safety and compliance
    safety_validation_passed = Column(Boolean, default=False)
    compliance_checks_passed = Column(Boolean, default=False)
    ethical_review_passed = Column(Boolean, default=False)
    
    # Additional metadata
    tags = Column(Text)  # JSON string
    notes = Column(Text)
    checksum = Column(String(255))


class ModelVersionManager:
    """
    Manages model versions and lifecycle with MLflow integration.
    """
    
    def __init__(
        self,
        mlflow_tracking_uri: str = "sqlite:///mlflow.db",
        database_url: str = "sqlite:///model_registry.db",
        model_storage_path: str = "./model_storage",
        validator: Optional[RSIValidator] = None,
        circuit_breaker: Optional[RSICircuitBreaker] = None
    ):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.database_url = database_url
        self.model_storage_path = Path(model_storage_path)
        self.validator = validator
        self.circuit_breaker = circuit_breaker or create_database_circuit()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_client = MlflowClient()
        
        # Initialize database
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create model storage directory
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model Version Manager initialized with MLflow URI: {mlflow_tracking_uri}")
    
    @trace_operation("create_model_version")
    def create_model_version(
        self,
        model: Any,
        metadata: ModelMetadata,
        artifacts: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new model version.
        
        Args:
            model: The trained model object
            metadata: Model metadata
            artifacts: Additional artifacts to store
            
        Returns:
            Model version ID
        """
        try:
            # Validate metadata
            if self.validator:
                # Convert metadata to dict for validation
                metadata_dict = {
                    "accuracy": metadata.accuracy or 0.0,
                    "training_duration": metadata.training_duration_seconds or 0.0,
                    "model_type": metadata.model_type.value
                }
                
                validation_result = self.validator.validate_performance_metrics(metadata_dict)
                if not validation_result.valid:
                    raise ValueError(f"Model metadata validation failed: {validation_result.message}")
            
            # Generate model ID if not provided
            if not metadata.model_id:
                metadata.model_id = str(uuid.uuid4())
            
            # Calculate model checksum
            model_bytes = pickle.dumps(model)
            metadata.checksum = hashlib.sha256(model_bytes).hexdigest()
            
            # Create MLflow experiment if it doesn't exist
            experiment_name = f"rsi_model_{metadata.model_type.value}"
            try:
                experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = self.mlflow_client.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            except MlflowException:
                experiment_id = self.mlflow_client.create_experiment(experiment_name)
            
            metadata.experiment_id = experiment_id
            
            # Start MLflow run
            with mlflow.start_run(experiment_id=experiment_id) as run:
                metadata.training_run_id = run.info.run_id
                
                # Log model with MLflow
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=metadata.name
                )
                
                # Log parameters
                if metadata.hyperparameters:
                    mlflow.log_params(metadata.hyperparameters)
                
                # Log metrics
                if metadata.accuracy is not None:
                    mlflow.log_metric("accuracy", metadata.accuracy)
                if metadata.precision is not None:
                    mlflow.log_metric("precision", metadata.precision)
                if metadata.recall is not None:
                    mlflow.log_metric("recall", metadata.recall)
                if metadata.f1_score is not None:
                    mlflow.log_metric("f1_score", metadata.f1_score)
                if metadata.validation_accuracy is not None:
                    mlflow.log_metric("validation_accuracy", metadata.validation_accuracy)
                
                # Log artifacts
                if artifacts:
                    for artifact_name, artifact_data in artifacts.items():
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                            json.dump(artifact_data, f)
                            mlflow.log_artifact(f.name, artifact_name)
                            os.unlink(f.name)
                
                # Log tags
                if metadata.tags:
                    mlflow.set_tags({f"tag_{i}": tag for i, tag in enumerate(metadata.tags)})
                
                # Set additional tags
                mlflow.set_tags({
                    "model_type": metadata.model_type.value,
                    "status": metadata.status.value,
                    "safety_validated": str(metadata.safety_validation_passed),
                    "compliance_checked": str(metadata.compliance_checks_passed)
                })
            
            # Store model locally
            model_file_path = self.model_storage_path / f"{metadata.model_id}.pkl"
            with open(model_file_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Store metadata in database
            self._store_model_metadata(metadata)
            
            logger.info(f"Created model version: {metadata.model_id}")
            return metadata.model_id
            
        except Exception as e:
            logger.error(f"Failed to create model version: {e}")
            record_safety_event(
                "model_version_creation_failed",
                "error",
                {"error": str(e), "model_id": metadata.model_id}
            )
            raise
    
    def _store_model_metadata(self, metadata: ModelMetadata):
        """Store model metadata in database."""
        with self.SessionLocal() as session:
            model_record = ModelRecord(
                model_id=metadata.model_id,
                name=metadata.name,
                version=metadata.version,
                model_type=metadata.model_type.value,
                status=metadata.status.value,
                created_at=metadata.created_at,
                updated_at=metadata.updated_at,
                accuracy=metadata.accuracy,
                precision=metadata.precision,
                recall=metadata.recall,
                f1_score=metadata.f1_score,
                auc_score=metadata.auc_score,
                training_dataset_size=metadata.training_dataset_size,
                training_duration_seconds=metadata.training_duration_seconds,
                hyperparameters=json.dumps(metadata.hyperparameters),
                validation_accuracy=metadata.validation_accuracy,
                validation_dataset_size=metadata.validation_dataset_size,
                cross_validation_scores=json.dumps(metadata.cross_validation_scores),
                deployment_target=metadata.deployment_target,
                resource_requirements=json.dumps(metadata.resource_requirements),
                parent_model_id=metadata.parent_model_id,
                training_run_id=metadata.training_run_id,
                experiment_id=metadata.experiment_id,
                safety_validation_passed=metadata.safety_validation_passed,
                compliance_checks_passed=metadata.compliance_checks_passed,
                ethical_review_passed=metadata.ethical_review_passed,
                tags=json.dumps(metadata.tags),
                notes=metadata.notes,
                checksum=metadata.checksum
            )
            
            session.add(model_record)
            session.commit()
    
    @trace_operation("load_model_version")
    def load_model_version(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """
        Load a specific model version.
        
        Args:
            model_id: Model version ID
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            # Load metadata from database
            metadata = self._load_model_metadata(model_id)
            if not metadata:
                raise ValueError(f"Model version not found: {model_id}")
            
            # Load model from local storage
            model_file_path = self.model_storage_path / f"{model_id}.pkl"
            if not model_file_path.exists():
                raise ValueError(f"Model file not found: {model_file_path}")
            
            with open(model_file_path, 'rb') as f:
                model = pickle.load(f)
            
            # Verify model integrity
            model_bytes = pickle.dumps(model)
            computed_checksum = hashlib.sha256(model_bytes).hexdigest()
            
            if computed_checksum != metadata.checksum:
                record_safety_event(
                    "model_checksum_mismatch",
                    "critical",
                    {"model_id": model_id, "expected": metadata.checksum, "actual": computed_checksum}
                )
                raise ValueError(f"Model checksum mismatch for {model_id}")
            
            logger.info(f"Loaded model version: {model_id}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model version {model_id}: {e}")
            raise
    
    def _load_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Load model metadata from database."""
        with self.SessionLocal() as session:
            record = session.query(ModelRecord).filter(
                ModelRecord.model_id == model_id
            ).first()
            
            if not record:
                return None
            
            return ModelMetadata(
                model_id=record.model_id,
                name=record.name,
                version=record.version,
                model_type=ModelType(record.model_type),
                status=ModelStatus(record.status),
                created_at=record.created_at,
                updated_at=record.updated_at,
                accuracy=record.accuracy,
                precision=record.precision,
                recall=record.recall,
                f1_score=record.f1_score,
                auc_score=record.auc_score,
                training_dataset_size=record.training_dataset_size,
                training_duration_seconds=record.training_duration_seconds,
                hyperparameters=json.loads(record.hyperparameters or "{}"),
                validation_accuracy=record.validation_accuracy,
                validation_dataset_size=record.validation_dataset_size,
                cross_validation_scores=json.loads(record.cross_validation_scores or "[]"),
                deployment_target=record.deployment_target,
                resource_requirements=json.loads(record.resource_requirements or "{}"),
                parent_model_id=record.parent_model_id,
                training_run_id=record.training_run_id,
                experiment_id=record.experiment_id,
                safety_validation_passed=record.safety_validation_passed,
                compliance_checks_passed=record.compliance_checks_passed,
                ethical_review_passed=record.ethical_review_passed,
                tags=json.loads(record.tags or "[]"),
                notes=record.notes or "",
                checksum=record.checksum
            )
    
    def list_model_versions(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        limit: int = 100
    ) -> List[ModelMetadata]:
        """
        List model versions with optional filtering.
        
        Args:
            model_name: Filter by model name
            model_type: Filter by model type
            status: Filter by status
            limit: Maximum number of results
            
        Returns:
            List of model metadata
        """
        with self.SessionLocal() as session:
            query = session.query(ModelRecord)
            
            if model_name:
                query = query.filter(ModelRecord.name == model_name)
            
            if model_type:
                query = query.filter(ModelRecord.model_type == model_type.value)
            
            if status:
                query = query.filter(ModelRecord.status == status.value)
            
            query = query.order_by(ModelRecord.created_at.desc()).limit(limit)
            
            records = query.all()
            
            return [self._record_to_metadata(record) for record in records]
    
    def _record_to_metadata(self, record: ModelRecord) -> ModelMetadata:
        """Convert database record to metadata object."""
        return ModelMetadata(
            model_id=record.model_id,
            name=record.name,
            version=record.version,
            model_type=ModelType(record.model_type),
            status=ModelStatus(record.status),
            created_at=record.created_at,
            updated_at=record.updated_at,
            accuracy=record.accuracy,
            precision=record.precision,
            recall=record.recall,
            f1_score=record.f1_score,
            auc_score=record.auc_score,
            training_dataset_size=record.training_dataset_size,
            training_duration_seconds=record.training_duration_seconds,
            hyperparameters=json.loads(record.hyperparameters or "{}"),
            validation_accuracy=record.validation_accuracy,
            validation_dataset_size=record.validation_dataset_size,
            cross_validation_scores=json.loads(record.cross_validation_scores or "[]"),
            deployment_target=record.deployment_target,
            resource_requirements=json.loads(record.resource_requirements or "{}"),
            parent_model_id=record.parent_model_id,
            training_run_id=record.training_run_id,
            experiment_id=record.experiment_id,
            safety_validation_passed=record.safety_validation_passed,
            compliance_checks_passed=record.compliance_checks_passed,
            ethical_review_passed=record.ethical_review_passed,
            tags=json.loads(record.tags or "[]"),
            notes=record.notes or "",
            checksum=record.checksum
        )
    
    def promote_model(self, model_id: str, target_status: ModelStatus) -> bool:
        """
        Promote a model to a new status.
        
        Args:
            model_id: Model ID to promote
            target_status: Target status
            
        Returns:
            True if successful
        """
        try:
            with self.SessionLocal() as session:
                record = session.query(ModelRecord).filter(
                    ModelRecord.model_id == model_id
                ).first()
                
                if not record:
                    raise ValueError(f"Model not found: {model_id}")
                
                # Validate promotion path
                current_status = ModelStatus(record.status)
                if not self._is_valid_promotion(current_status, target_status):
                    raise ValueError(f"Invalid promotion from {current_status} to {target_status}")
                
                # Update status
                record.status = target_status.value
                record.updated_at = datetime.now(timezone.utc)
                
                session.commit()
                
                logger.info(f"Promoted model {model_id} from {current_status} to {target_status}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to promote model {model_id}: {e}")
            return False
    
    def _is_valid_promotion(self, current: ModelStatus, target: ModelStatus) -> bool:
        """Check if promotion path is valid."""
        valid_transitions = {
            ModelStatus.TRAINING: [ModelStatus.VALIDATION, ModelStatus.ARCHIVED],
            ModelStatus.VALIDATION: [ModelStatus.STAGING, ModelStatus.ARCHIVED],
            ModelStatus.STAGING: [ModelStatus.PRODUCTION, ModelStatus.ARCHIVED],
            ModelStatus.PRODUCTION: [ModelStatus.DEPRECATED, ModelStatus.ARCHIVED],
            ModelStatus.DEPRECATED: [ModelStatus.ARCHIVED],
            ModelStatus.ARCHIVED: []  # Terminal state
        }
        
        return target in valid_transitions.get(current, [])
    
    def archive_model(self, model_id: str) -> bool:
        """Archive a model version."""
        return self.promote_model(model_id, ModelStatus.ARCHIVED)
    
    def get_production_models(self) -> List[ModelMetadata]:
        """Get all models in production status."""
        return self.list_model_versions(status=ModelStatus.PRODUCTION)
    
    def get_model_lineage(self, model_id: str) -> List[ModelMetadata]:
        """Get the lineage of a model (parent and children)."""
        lineage = []
        
        # Get the model itself
        metadata = self._load_model_metadata(model_id)
        if metadata:
            lineage.append(metadata)
            
            # Get parent models
            current_parent = metadata.parent_model_id
            while current_parent:
                parent_metadata = self._load_model_metadata(current_parent)
                if parent_metadata:
                    lineage.insert(0, parent_metadata)
                    current_parent = parent_metadata.parent_model_id
                else:
                    break
            
            # Get child models
            with self.SessionLocal() as session:
                children = session.query(ModelRecord).filter(
                    ModelRecord.parent_model_id == model_id
                ).all()
                
                for child in children:
                    lineage.append(self._record_to_metadata(child))
        
        return lineage
    
    def cleanup_old_models(self, retention_days: int = 30) -> int:
        """
        Clean up old archived models.
        
        Args:
            retention_days: Number of days to retain archived models
            
        Returns:
            Number of models cleaned up
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cleaned_count = 0
        
        with self.SessionLocal() as session:
            old_models = session.query(ModelRecord).filter(
                ModelRecord.status == ModelStatus.ARCHIVED.value,
                ModelRecord.updated_at < cutoff_date
            ).all()
            
            for model in old_models:
                try:
                    # Remove model file
                    model_file_path = self.model_storage_path / f"{model.model_id}.pkl"
                    if model_file_path.exists():
                        model_file_path.unlink()
                    
                    # Remove database record
                    session.delete(model)
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to cleanup model {model.model_id}: {e}")
            
            session.commit()
        
        logger.info(f"Cleaned up {cleaned_count} old models")
        return cleaned_count


# Factory functions for different configurations
def create_development_model_manager(
    base_path: str = "./dev_models"
) -> ModelVersionManager:
    """Create model manager for development environment."""
    return ModelVersionManager(
        mlflow_tracking_uri=f"sqlite:///{base_path}/mlflow.db",
        database_url=f"sqlite:///{base_path}/model_registry.db",
        model_storage_path=f"{base_path}/models"
    )


def create_production_model_manager(
    mlflow_tracking_uri: str,
    database_url: str,
    model_storage_path: str,
    validator: RSIValidator
) -> ModelVersionManager:
    """Create model manager for production environment."""
    return ModelVersionManager(
        mlflow_tracking_uri=mlflow_tracking_uri,
        database_url=database_url,
        model_storage_path=model_storage_path,
        validator=validator
    )