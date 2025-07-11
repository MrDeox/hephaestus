"""
PyTorch Lightning orchestrator for multi-task learning in RSI systems.
Provides enterprise-grade training orchestration with built-in monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import ModelSummary

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import logging
import os
import json

from ..core.state import StateManager, RSIState
from ..validation.validators import RSIValidator
from ..monitoring.audit_logger import get_audit_logger


logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of learning tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    UNSUPERVISED = "unsupervised"
    MULTI_MODAL = "multi_modal"


@dataclass
class LightningConfig:
    """Configuration for PyTorch Lightning orchestrator."""
    # Training configuration
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Model configuration
    model_name: str = "multi_task_rsi"
    hidden_dims: List[int] = None
    dropout_rate: float = 0.1
    
    # Multi-task configuration
    task_weights: Dict[str, float] = None
    shared_layers: int = 2
    task_specific_layers: int = 1
    
    # Optimization configuration
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    
    # Distributed training
    num_gpus: int = 0  # Set to 0 for CPU-only environment
    num_nodes: int = 1
    strategy: str = "ddp"
    precision: int = 32  # Use 32-bit precision for CPU
    
    # Monitoring
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0
    
    # Callbacks
    early_stopping_patience: int = 10
    checkpoint_every_n_epochs: int = 5
    
    # Safety constraints
    gradient_clip_val: float = 1.0
    max_memory_gb: float = 16.0
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        if self.task_weights is None:
            self.task_weights = {}


class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning."""
    
    def __init__(self, data: List[Dict[str, Any]], task_types: Dict[str, TaskType]):
        self.data = data
        self.task_types = task_types
        self.task_names = list(task_types.keys())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Prepare features
        features = torch.tensor(sample['features'], dtype=torch.float32)
        
        # Prepare targets for each task
        targets = {}
        for task_name, task_type in self.task_types.items():
            if task_name in sample:
                target = sample[task_name]
                if task_type == TaskType.CLASSIFICATION:
                    targets[task_name] = torch.tensor(target, dtype=torch.long)
                else:  # REGRESSION
                    targets[task_name] = torch.tensor(target, dtype=torch.float32)
        
        return {'features': features, 'targets': targets}


class MultiTaskModel(nn.Module):
    """Multi-task neural network with shared and task-specific layers."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        task_configs: Dict[str, Dict[str, Any]],
        shared_layers: int = 2,
        task_specific_layers: int = 1,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.task_configs = task_configs
        self.shared_layers = shared_layers
        self.task_specific_layers = task_specific_layers
        
        # Shared layers
        shared_layer_list = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:shared_layers]):
            shared_layer_list.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*shared_layer_list)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            task_layers = []
            current_dim = prev_dim
            
            # Task-specific layers
            for i in range(task_specific_layers):
                if i < len(hidden_dims) - shared_layers:
                    next_dim = hidden_dims[shared_layers + i]
                else:
                    next_dim = current_dim // 2
                
                task_layers.extend([
                    nn.Linear(current_dim, next_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.BatchNorm1d(next_dim)
                ])
                current_dim = next_dim
            
            # Output layer
            output_dim = config['output_dim']
            task_layers.append(nn.Linear(current_dim, output_dim))
            
            self.task_heads[task_name] = nn.Sequential(*task_layers)
    
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Task-specific predictions
        outputs = {}
        for task_name, task_head in self.task_heads.items():
            outputs[task_name] = task_head(shared_features)
        
        return outputs


class LightningRSISystem(pl.LightningModule):
    """PyTorch Lightning module for RSI multi-task learning."""
    
    def __init__(
        self,
        config: LightningConfig,
        task_configs: Dict[str, Dict[str, Any]],
        state_manager: StateManager,
        validator: RSIValidator
    ):
        super().__init__()
        
        self.config = config
        self.task_configs = task_configs
        self.state_manager = state_manager
        self.validator = validator
        self.audit_logger = get_audit_logger()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['state_manager', 'validator'])
        
        # Initialize model
        input_dim = task_configs.get('input_dim', 10)
        self.model = MultiTaskModel(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            task_configs=task_configs,
            shared_layers=config.shared_layers,
            task_specific_layers=config.task_specific_layers,
            dropout_rate=config.dropout_rate
        )
        
        # Task weights for balancing
        self.task_weights = config.task_weights
        
        # Metrics tracking
        self.training_metrics = {}
        self.validation_metrics = {}
        
        # Initialize task-specific metrics
        for task_name, task_config in task_configs.items():
            if task_config['type'] == TaskType.CLASSIFICATION:
                self.training_metrics[f"{task_name}_accuracy"] = []
                self.validation_metrics[f"{task_name}_accuracy"] = []
            
            self.training_metrics[f"{task_name}_loss"] = []
            self.validation_metrics[f"{task_name}_loss"] = []
        
        logger.info(f"Lightning RSI system initialized with {len(task_configs)} tasks")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step for multi-task learning."""
        features = batch['features']
        targets = batch['targets']
        
        # Forward pass
        outputs = self.forward(features)
        
        # Compute losses for each task
        task_losses = {}
        total_loss = 0.0
        
        for task_name, task_config in self.task_configs.items():
            if task_name in targets:
                task_output = outputs[task_name]
                task_target = targets[task_name]
                
                if task_config['type'] == TaskType.CLASSIFICATION:
                    task_loss = F.cross_entropy(task_output, task_target)
                    
                    # Compute accuracy
                    predictions = torch.argmax(task_output, dim=1)
                    accuracy = (predictions == task_target).float().mean()
                    self.log(f"train_{task_name}_accuracy", accuracy, prog_bar=True)
                    
                elif task_config['type'] == TaskType.REGRESSION:
                    task_loss = F.mse_loss(task_output, task_target)
                
                # Weight the loss
                task_weight = self.task_weights.get(task_name, 1.0)
                weighted_loss = task_weight * task_loss
                
                task_losses[task_name] = task_loss
                total_loss += weighted_loss
                
                # Log task-specific loss
                self.log(f"train_{task_name}_loss", task_loss, prog_bar=True)
        
        # Log total loss
        self.log("train_loss", total_loss, prog_bar=True)
        
        # Update metrics
        self._update_training_metrics(task_losses)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for multi-task learning."""
        features = batch['features']
        targets = batch['targets']
        
        # Forward pass
        outputs = self.forward(features)
        
        # Compute losses for each task
        task_losses = {}
        total_loss = 0.0
        
        for task_name, task_config in self.task_configs.items():
            if task_name in targets:
                task_output = outputs[task_name]
                task_target = targets[task_name]
                
                if task_config['type'] == TaskType.CLASSIFICATION:
                    task_loss = F.cross_entropy(task_output, task_target)
                    
                    # Compute accuracy
                    predictions = torch.argmax(task_output, dim=1)
                    accuracy = (predictions == task_target).float().mean()
                    self.log(f"val_{task_name}_accuracy", accuracy, prog_bar=True)
                    
                elif task_config['type'] == TaskType.REGRESSION:
                    task_loss = F.mse_loss(task_output, task_target)
                
                # Weight the loss
                task_weight = self.task_weights.get(task_name, 1.0)
                weighted_loss = task_weight * task_loss
                
                task_losses[task_name] = task_loss
                total_loss += weighted_loss
                
                # Log task-specific loss
                self.log(f"val_{task_name}_loss", task_loss, prog_bar=True)
        
        # Log total loss
        self.log("val_loss", total_loss, prog_bar=True)
        
        # Update metrics
        self._update_validation_metrics(task_losses)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Optimizer
        if self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Scheduler
        if self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.max_epochs // 3,
                gamma=0.1
            )
        else:
            scheduler = None
        
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        else:
            return optimizer
    
    def _update_training_metrics(self, task_losses: Dict[str, torch.Tensor]):
        """Update training metrics."""
        for task_name, loss in task_losses.items():
            loss_key = f"{task_name}_loss"
            if loss_key not in self.training_metrics:
                self.training_metrics[loss_key] = []
            self.training_metrics[loss_key].append(loss.item())
    
    def _update_validation_metrics(self, task_losses: Dict[str, torch.Tensor]):
        """Update validation metrics."""
        for task_name, loss in task_losses.items():
            loss_key = f"{task_name}_loss"
            if loss_key not in self.validation_metrics:
                self.validation_metrics[loss_key] = []
            self.validation_metrics[loss_key].append(loss.item())
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        summary = {
            'training': {},
            'validation': {}
        }
        
        for metric_name, values in self.training_metrics.items():
            if values:
                summary['training'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        for metric_name, values in self.validation_metrics.items():
            if values:
                summary['validation'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        return summary


class LightningOrchestrator:
    """Orchestrator for PyTorch Lightning training."""
    
    def __init__(
        self,
        config: LightningConfig,
        task_configs: Dict[str, Dict[str, Any]],
        state_manager: StateManager,
        validator: RSIValidator
    ):
        self.config = config
        self.task_configs = task_configs
        self.state_manager = state_manager
        self.validator = validator
        self.audit_logger = get_audit_logger()
        
        # Initialize Lightning module
        self.lightning_module = LightningRSISystem(
            config, task_configs, state_manager, validator
        )
        
        # Initialize trainer
        self.trainer = self._create_trainer()
        
        logger.info("Lightning orchestrator initialized")
    
    def _create_trainer(self) -> pl.Trainer:
        """Create PyTorch Lightning trainer."""
        # Callbacks
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./checkpoints/{self.config.model_name}",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            every_n_epochs=self.config.checkpoint_every_n_epochs
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            mode="min",
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        # Logger
        logger = MLFlowLogger(
            experiment_name=f"rsi_{self.config.model_name}",
            tracking_uri="./mlruns"
        )
        
        # Strategy
        if self.config.num_gpus > 1:
            strategy = DDPStrategy(find_unused_parameters=False)
        else:
            strategy = "auto"
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu" if self.config.num_gpus > 0 else "cpu",
            devices=self.config.num_gpus if self.config.num_gpus > 0 else 1,
            num_nodes=self.config.num_nodes,
            strategy=strategy,
            precision=self.config.precision,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=self.config.log_every_n_steps,
            val_check_interval=self.config.val_check_interval,
            gradient_clip_val=self.config.gradient_clip_val,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True
        )
        
        return trainer
    
    async def train(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Train the multi-task model."""
        try:
            # Validate input data
            validation_result = self.validator.validate_performance_metrics({
                'train_size': len(train_data),
                'val_size': len(val_data)
            })
            
            if not validation_result.valid:
                raise ValueError(f"Invalid training data: {validation_result.message}")
            
            # Create datasets
            task_types = {
                name: TaskType(config['type']) 
                for name, config in self.task_configs.items()
            }
            
            train_dataset = MultiTaskDataset(train_data, task_types)
            val_dataset = MultiTaskDataset(val_data, task_types)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Train the model
            self.trainer.fit(
                self.lightning_module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
            )
            
            # Get training results
            training_results = {
                'best_model_path': self.trainer.checkpoint_callback.best_model_path,
                'best_model_score': self.trainer.checkpoint_callback.best_model_score.item(),
                'current_epoch': self.trainer.current_epoch,
                'global_step': self.trainer.global_step,
                'metrics_summary': self.lightning_module.get_metrics_summary(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Log training completion
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    "lightning_orchestrator",
                    "training_completed",
                    metadata=training_results
                )
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    async def predict(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make predictions using the trained model."""
        try:
            # Prepare data
            features = torch.tensor([item['features'] for item in data], dtype=torch.float32)
            
            # Make predictions
            self.lightning_module.eval()
            with torch.no_grad():
                outputs = self.lightning_module(features)
            
            # Process predictions
            predictions = {}
            for task_name, task_output in outputs.items():
                task_config = self.task_configs[task_name]
                
                if task_config['type'] == TaskType.CLASSIFICATION:
                    # Get class probabilities and predictions
                    probabilities = F.softmax(task_output, dim=1)
                    predicted_classes = torch.argmax(task_output, dim=1)
                    
                    predictions[task_name] = {
                        'classes': predicted_classes.cpu().numpy().tolist(),
                        'probabilities': probabilities.cpu().numpy().tolist()
                    }
                elif task_config['type'] == TaskType.REGRESSION:
                    predictions[task_name] = {
                        'values': task_output.cpu().numpy().tolist()
                    }
            
            result = {
                'predictions': predictions,
                'num_samples': len(data),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


def create_lightning_orchestrator(
    task_configs: Dict[str, Dict[str, Any]],
    state_manager: StateManager,
    validator: RSIValidator,
    config: Optional[LightningConfig] = None
) -> LightningOrchestrator:
    """Factory function to create Lightning orchestrator."""
    if config is None:
        config = LightningConfig()
    
    return LightningOrchestrator(config, task_configs, state_manager, validator)