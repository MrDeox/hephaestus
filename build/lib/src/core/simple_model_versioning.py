"""
Simple Model Versioning - Alternative to MLflow with dependency conflicts.
Provides basic model versioning and tracking without external dependencies.
"""

import json
import sqlite3
import pickle
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models supported."""
    ONLINE_LEARNER = "online_learner"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    META_LEARNER = "meta_learner"
    CONTINUAL_LEARNER = "continual_learner"
    RL_AGENT = "rl_agent"


class ModelStatus(Enum):
    """Status of model versions."""
    PENDING = "pending"
    TRAINING = "training"
    READY = "ready"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata for model versions."""
    model_id: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    tags: Dict[str, str]
    description: Optional[str] = None
    file_path: Optional[str] = None
    checksum: Optional[str] = None


class SimpleModelVersionManager:
    """Simple model version manager without external dependencies."""
    
    def __init__(self, db_path: str = "model_versions.db", models_dir: str = "models/"):
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        logger.info(f"Simple Model Version Manager initialized: {db_path}")
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                model_type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metrics TEXT NOT NULL,
                parameters TEXT NOT NULL,
                tags TEXT NOT NULL,
                description TEXT,
                file_path TEXT,
                checksum TEXT,
                UNIQUE(model_id, version)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_model(
        self,
        model: Any,
        model_id: str,
        version: str,
        model_type: ModelType,
        metrics: Dict[str, Any] = None,
        parameters: Dict[str, Any] = None,
        tags: Dict[str, str] = None,
        description: Optional[str] = None
    ) -> ModelMetadata:
        """Save a model version."""
        
        # Create file path
        file_path = self.models_dir / f"{model_id}_{version}.pkl"
        
        # Save model to file
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Calculate checksum
        checksum = self._calculate_checksum(file_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            model_type=model_type,
            status=ModelStatus.READY,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metrics=metrics or {},
            parameters=parameters or {},
            tags=tags or {},
            description=description,
            file_path=str(file_path),
            checksum=checksum
        )
        
        # Save to database
        self._save_metadata(metadata)
        
        logger.info(f"Model saved: {model_id} v{version}")
        return metadata
    
    def load_model(self, model_id: str, version: str = None) -> Any:
        """Load a model version."""
        
        if version is None:
            version = self.get_latest_version(model_id)
        
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Model not found: {model_id} v{version}")
        
        # Load model from file
        with open(metadata.file_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded: {model_id} v{version}")
        return model
    
    def get_model_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Get metadata for a model version."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM model_versions 
            WHERE model_id = ? AND version = ?
        """, (model_id, version))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return self._row_to_metadata(row)
    
    def list_models(self) -> List[str]:
        """List all model IDs."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT model_id FROM model_versions")
        rows = cursor.fetchall()
        conn.close()
        
        return [row[0] for row in rows]
    
    def list_versions(self, model_id: str) -> List[str]:
        """List all versions for a model."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT version FROM model_versions 
            WHERE model_id = ? 
            ORDER BY created_at DESC
        """, (model_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [row[0] for row in rows]
    
    def get_latest_version(self, model_id: str) -> str:
        """Get the latest version of a model."""
        
        versions = self.list_versions(model_id)
        if not versions:
            raise ValueError(f"No versions found for model: {model_id}")
        
        return versions[0]
    
    def update_model_status(self, model_id: str, version: str, status: ModelStatus):
        """Update model status."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE model_versions 
            SET status = ?, updated_at = ?
            WHERE model_id = ? AND version = ?
        """, (status.value, datetime.now(timezone.utc).isoformat(), model_id, version))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Model status updated: {model_id} v{version} -> {status.value}")
    
    def delete_model(self, model_id: str, version: str):
        """Delete a model version."""
        
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Model not found: {model_id} v{version}")
        
        # Delete file
        Path(metadata.file_path).unlink(missing_ok=True)
        
        # Delete from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM model_versions 
            WHERE model_id = ? AND version = ?
        """, (model_id, version))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Model deleted: {model_id} v{version}")
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO model_versions (
                model_id, version, model_type, status, created_at, updated_at,
                metrics, parameters, tags, description, file_path, checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.model_id,
            metadata.version,
            metadata.model_type.value,
            metadata.status.value,
            metadata.created_at.isoformat(),
            metadata.updated_at.isoformat(),
            json.dumps(metadata.metrics),
            json.dumps(metadata.parameters),
            json.dumps(metadata.tags),
            metadata.description,
            metadata.file_path,
            metadata.checksum
        ))
        
        conn.commit()
        conn.close()
    
    def _row_to_metadata(self, row) -> ModelMetadata:
        """Convert database row to ModelMetadata."""
        
        return ModelMetadata(
            model_id=row[1],
            version=row[2],
            model_type=ModelType(row[3]),
            status=ModelStatus(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            metrics=json.loads(row[7]),
            parameters=json.loads(row[8]),
            tags=json.loads(row[9]),
            description=row[10],
            file_path=row[11],
            checksum=row[12]
        )
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def get_model_performance(self, model_id: str, version: str = None) -> Dict[str, Any]:
        """Get performance metrics for a model."""
        
        if version is None:
            version = self.get_latest_version(model_id)
        
        metadata = self.get_model_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Model not found: {model_id} v{version}")
        
        return metadata.metrics
    
    def compare_models(self, model_specs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Compare multiple model versions."""
        
        comparison = {}
        
        for model_id, version in model_specs:
            metadata = self.get_model_metadata(model_id, version)
            if metadata:
                comparison[f"{model_id}_v{version}"] = {
                    'metrics': metadata.metrics,
                    'parameters': metadata.parameters,
                    'status': metadata.status.value,
                    'created_at': metadata.created_at.isoformat()
                }
        
        return comparison