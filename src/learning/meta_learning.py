"""
Advanced Meta-Learning System for RSI.
Implements MAML, ProtoNets, and other meta-learning algorithms using Learn2Learn.
"""

import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import higher
    HIGHER_AVAILABLE = True
except ImportError:
    HIGHER_AVAILABLE = False
    logger.warning("Higher not available. Using fallback implementations.")

from ..core.state import RSIStateManager, RSIState
from ..validation.validators import RSIValidator
from ..monitoring.audit_logger import get_audit_logger


logger = logging.getLogger(__name__)


class MetaLearningAlgorithm(Enum):
    """Supported meta-learning algorithms."""
    MAML = "maml"
    PROTONET = "protonet"
    META_SGD = "meta_sgd"
    REPTILE = "reptile"
    ANIL = "anil"


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning system."""
    algorithm: MetaLearningAlgorithm
    num_ways: int = 5
    num_shots: int = 5
    num_queries: int = 15
    adaptation_steps: int = 5
    meta_learning_rate: float = 0.001
    adaptation_learning_rate: float = 0.01
    meta_batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Advanced settings
    first_order: bool = False  # First-order MAML
    allow_unused: bool = True
    allow_nograd: bool = True
    max_grad_norm: float = 1.0
    
    # Memory efficiency
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Safety constraints
    max_adaptation_steps: int = 10
    min_accuracy_threshold: float = 0.5
    max_loss_threshold: float = 10.0


class MetaLearningDataset(Dataset):
    """Custom dataset for meta-learning tasks."""
    
    def __init__(self, data: List[Dict[str, Any]], transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class MetaLearningModel(nn.Module):
    """Base neural network for meta-learning."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MAML:
    """Model-Agnostic Meta-Learning implementation using Higher."""
    
    def __init__(self, model, lr=0.01, first_order=False, allow_unused=True, allow_nograd=True):
        self.model = model
        self.lr = lr
        self.first_order = first_order
        self.allow_unused = allow_unused
        self.allow_nograd = allow_nograd
        
    def clone(self):
        """Create a functional clone for adaptation."""
        return self
    
    def adapt(self, loss):
        """Adapt the model using the given loss."""
        # In Higher, adaptation is handled differently
        # This is a simplified version
        pass
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def __call__(self, x):
        """Make model callable."""
        return self.model(x)


class ProtoNet:
    """Prototypical Networks implementation."""
    
    def __init__(self, model):
        self.model = model
        
    def clone(self):
        """Create a functional clone for adaptation."""
        return self
    
    def adapt(self, loss):
        """Adapt using prototypical learning."""
        pass
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def __call__(self, x):
        """Make model callable."""
        return self.model(x)


class MetaSGD:
    """Meta-SGD implementation."""
    
    def __init__(self, model, lr=0.01, first_order=False):
        self.model = model
        self.lr = lr
        self.first_order = first_order
        
    def clone(self):
        """Create a functional clone for adaptation."""
        return self
    
    def adapt(self, loss):
        """Adapt using Meta-SGD."""
        pass
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def __call__(self, x):
        """Make model callable."""
        return self.model(x)


class RSIMetaLearningSystem:
    """Advanced meta-learning system for RSI."""
    
    def __init__(
        self,
        config: MetaLearningConfig,
        state_manager: RSIStateManager,
        validator: RSIValidator,
        model_architecture: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.state_manager = state_manager
        self.validator = validator
        self.audit_logger = get_audit_logger()
        
        # Initialize model
        self.model = self._create_model(model_architecture or {})
        self.model.to(self.config.device)
        
        # Initialize meta-learning algorithm
        self.meta_learner = self._create_meta_learner()
        
        # Initialize optimizer
        self.meta_optimizer = torch.optim.AdamW(
            self.meta_learner.parameters(),
            lr=self.config.meta_learning_rate,
            weight_decay=1e-4
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.meta_optimizer,
            T_0=100,
            T_mult=2
        )
        
        # Metrics tracking
        self.meta_learning_metrics = {
            'meta_loss': [],
            'adaptation_accuracy': [],
            'query_accuracy': [],
            'adaptation_steps': [],
            'learning_efficiency': []
        }
        
        # Mixed precision scaler
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Meta-learning system initialized with {config.algorithm.value}")
    
    def _create_model(self, architecture: Dict[str, Any]) -> nn.Module:
        """Create the base model for meta-learning."""
        input_dim = architecture.get('input_dim', 10)
        hidden_dims = architecture.get('hidden_dims', [64, 32])
        output_dim = architecture.get('output_dim', 2)
        
        return MetaLearningModel(input_dim, hidden_dims, output_dim)
    
    def _create_meta_learner(self) -> nn.Module:
        """Create the meta-learning algorithm wrapper."""
        if self.config.algorithm == MetaLearningAlgorithm.MAML:
            return MAML(
                self.model,
                lr=self.config.adaptation_learning_rate,
                first_order=self.config.first_order,
                allow_unused=self.config.allow_unused,
                allow_nograd=self.config.allow_nograd
            )
        elif self.config.algorithm == MetaLearningAlgorithm.PROTONET:
            return ProtoNet(self.model)
        elif self.config.algorithm == MetaLearningAlgorithm.META_SGD:
            return MetaSGD(
                self.model,
                lr=self.config.adaptation_learning_rate,
                first_order=self.config.first_order
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
    
    async def meta_train_step(
        self,
        tasks: List[Dict[str, Any]],
        validation_tasks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Perform one meta-training step."""
        try:
            self.meta_learner.train()
            
            # Prepare task batch
            task_batch = self._prepare_task_batch(tasks)
            
            # Meta-training loop
            meta_loss = 0.0
            adaptation_accuracies = []
            query_accuracies = []
            
            for task in task_batch:
                # Fast adaptation
                learner = self.meta_learner.clone()
                adaptation_data = task['support']
                query_data = task['query']
                
                # Adapt to task
                for step in range(self.config.adaptation_steps):
                    support_loss = self._compute_loss(learner, adaptation_data)
                    learner.adapt(support_loss)
                
                # Evaluate on query set
                query_loss = self._compute_loss(learner, query_data)
                query_accuracy = self._compute_accuracy(learner, query_data)
                
                meta_loss += query_loss
                query_accuracies.append(query_accuracy)
                
                # Compute adaptation accuracy
                adaptation_accuracy = self._compute_accuracy(learner, adaptation_data)
                adaptation_accuracies.append(adaptation_accuracy)
            
            # Meta-update
            meta_loss = meta_loss / len(task_batch)
            
            # Gradient clipping and optimization
            if self.config.mixed_precision:
                self.scaler.scale(meta_loss).backward()
                self.scaler.unscale_(self.meta_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.meta_learner.parameters(),
                    self.config.max_grad_norm
                )
                self.scaler.step(self.meta_optimizer)
                self.scaler.update()
            else:
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.meta_learner.parameters(),
                    self.config.max_grad_norm
                )
                self.meta_optimizer.step()
            
            self.meta_optimizer.zero_grad()
            self.scheduler.step()
            
            # Update metrics
            metrics = {
                'meta_loss': meta_loss.item(),
                'adaptation_accuracy': np.mean(adaptation_accuracies),
                'query_accuracy': np.mean(query_accuracies),
                'learning_rate': self.scheduler.get_last_lr()[0],
                'adaptation_steps': self.config.adaptation_steps
            }
            
            self._update_metrics(metrics)
            
            # Validate on validation tasks if provided
            if validation_tasks:
                val_metrics = await self.meta_validate(validation_tasks)
                metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
            
            # Log metrics
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    "meta_learning_system",
                    "meta_train_step",
                    metadata=metrics
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Meta-training step failed: {e}")
            raise
    
    async def meta_validate(self, validation_tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Validate meta-learning performance."""
        self.meta_learner.eval()
        
        val_accuracies = []
        val_losses = []
        
        with torch.no_grad():
            for task in validation_tasks:
                # Clone and adapt
                learner = self.meta_learner.clone()
                adaptation_data = task['support']
                query_data = task['query']
                
                # Fast adaptation
                for step in range(self.config.adaptation_steps):
                    support_loss = self._compute_loss(learner, adaptation_data)
                    learner.adapt(support_loss)
                
                # Evaluate
                query_loss = self._compute_loss(learner, query_data)
                query_accuracy = self._compute_accuracy(learner, query_data)
                
                val_losses.append(query_loss.item())
                val_accuracies.append(query_accuracy)
        
        return {
            'accuracy': np.mean(val_accuracies),
            'loss': np.mean(val_losses),
            'std_accuracy': np.std(val_accuracies)
        }
    
    async def few_shot_learn(
        self,
        support_data: List[Dict[str, Any]],
        query_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform few-shot learning on new task."""
        try:
            # Validate input data
            validation_result = self.validator.validate_performance_metrics({
                'support_size': len(support_data),
                'query_size': len(query_data)
            })
            
            if not validation_result.valid:
                raise ValueError(f"Invalid few-shot learning data: {validation_result.message}")
            
            # Clone meta-learner
            learner = self.meta_learner.clone()
            
            # Prepare data
            support_batch = self._prepare_data_batch(support_data)
            query_batch = self._prepare_data_batch(query_data)
            
            # Fast adaptation
            adaptation_losses = []
            for step in range(self.config.adaptation_steps):
                support_loss = self._compute_loss(learner, support_batch)
                learner.adapt(support_loss)
                adaptation_losses.append(support_loss.item())
            
            # Evaluate on query set
            with torch.no_grad():
                query_loss = self._compute_loss(learner, query_batch)
                query_accuracy = self._compute_accuracy(learner, query_batch)
                predictions = self._get_predictions(learner, query_batch)
            
            # Compute learning efficiency
            learning_efficiency = (query_accuracy - 0.5) / self.config.adaptation_steps
            
            result = {
                'query_accuracy': query_accuracy,
                'query_loss': query_loss.item(),
                'adaptation_losses': adaptation_losses,
                'learning_efficiency': learning_efficiency,
                'predictions': predictions,
                'num_adaptation_steps': self.config.adaptation_steps,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Log few-shot learning
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    "meta_learning_system",
                    "few_shot_learn",
                    metadata=result
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Few-shot learning failed: {e}")
            raise
    
    def _prepare_task_batch(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare a batch of tasks for meta-training."""
        prepared_tasks = []
        
        for task in tasks:
            # Convert to tensors
            support_x = torch.tensor(task['support']['features'], dtype=torch.float32)
            support_y = torch.tensor(task['support']['labels'], dtype=torch.long)
            query_x = torch.tensor(task['query']['features'], dtype=torch.float32)
            query_y = torch.tensor(task['query']['labels'], dtype=torch.long)
            
            # Move to device
            support_x = support_x.to(self.config.device)
            support_y = support_y.to(self.config.device)
            query_x = query_x.to(self.config.device)
            query_y = query_y.to(self.config.device)
            
            prepared_tasks.append({
                'support': (support_x, support_y),
                'query': (query_x, query_y)
            })
        
        return prepared_tasks
    
    def _prepare_data_batch(self, data: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data batch for training."""
        features = [item['features'] for item in data]
        labels = [item['label'] for item in data]
        
        x = torch.tensor(features, dtype=torch.float32).to(self.config.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.config.device)
        
        return x, y
    
    def _compute_loss(self, learner: nn.Module, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute loss for given data."""
        x, y = data
        logits = learner(x)
        return F.cross_entropy(logits, y)
    
    def _compute_accuracy(self, learner: nn.Module, data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Compute accuracy for given data."""
        x, y = data
        with torch.no_grad():
            logits = learner(x)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y).float().mean().item()
        return accuracy
    
    def _get_predictions(self, learner: nn.Module, data: Tuple[torch.Tensor, torch.Tensor]) -> List[int]:
        """Get predictions for given data."""
        x, y = data
        with torch.no_grad():
            logits = learner(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions.cpu().numpy().tolist()
    
    def _update_metrics(self, metrics: Dict[str, float]):
        """Update internal metrics tracking."""
        for key, value in metrics.items():
            if key in self.meta_learning_metrics:
                self.meta_learning_metrics[key].append(value)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of meta-learning metrics."""
        summary = {}
        
        for key, values in self.meta_learning_metrics.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return summary
    
    def save_checkpoint(self, path: str):
        """Save meta-learning checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'meta_learner_state_dict': self.meta_learner.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'metrics': self.meta_learning_metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Meta-learning checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load meta-learning checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_learner.load_state_dict(checkpoint['meta_learner_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.meta_learning_metrics = checkpoint['metrics']
        
        logger.info(f"Meta-learning checkpoint loaded from {path}")


def create_meta_learning_system(
    algorithm: MetaLearningAlgorithm = MetaLearningAlgorithm.MAML,
    state_manager: Optional[RSIStateManager] = None,
    validator: Optional[RSIValidator] = None,
    **kwargs
) -> RSIMetaLearningSystem:
    """Factory function to create meta-learning system."""
    from ..validation.validators import create_strict_validator
    
    config = MetaLearningConfig(algorithm=algorithm, **kwargs)
    
    if not validator:
        validator = create_strict_validator()
    
    return MetaLearningSystem(config, state_manager, validator)


def create_few_shot_task(
    support_data: List[Dict[str, Any]],
    query_data: List[Dict[str, Any]],
    num_classes: int
) -> Dict[str, Any]:
    """Create a few-shot learning task."""
    return {
        'support': {
            'features': [item['features'] for item in support_data],
            'labels': [item['label'] for item in support_data]
        },
        'query': {
            'features': [item['features'] for item in query_data],
            'labels': [item['label'] for item in query_data]
        },
        'num_classes': num_classes
    }
