"""
Ray Tune integration for distributed optimization and scaling.
Provides enterprise-scale distributed optimization with linear scaling.
"""

import ray
from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, HyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.ax import AxSearch
try:
    from ray.tune.integration.mlflow import MLflowLoggerCallback
    MLFLOW_INTEGRATION_AVAILABLE = True
except ImportError:
    MLFLOW_INTEGRATION_AVAILABLE = False

try:
    from ray.tune.integration.wandb import WandbLoggerCallback
    WANDB_INTEGRATION_AVAILABLE = True
except ImportError:
    WANDB_INTEGRATION_AVAILABLE = False
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.experiment import Experiment
from ray.tune.resources import PlacementGroupFactory

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import os
import time
from pathlib import Path
import tempfile
import shutil

from ..core.state import StateManager, RSIState
from ..validation.validators import RSIValidator
from ..monitoring.audit_logger import get_audit_logger


logger = logging.getLogger(__name__)


class SearchAlgorithm(Enum):
    """Types of search algorithms."""
    RANDOM = "random"
    GRID = "grid"
    BAYESOPT = "bayesopt"
    HYPEROPT = "hyperopt"
    OPTUNA = "optuna"
    AX = "ax"


class SchedulerType(Enum):
    """Types of schedulers."""
    FIFO = "fifo"
    ASHA = "asha"
    HYPERBAND = "hyperband"
    POPULATION_BASED = "population_based"
    MEDIAN_STOPPING = "median_stopping"


@dataclass
class RayTuneConfig:
    """Configuration for Ray Tune optimization."""
    # Experiment configuration
    experiment_name: str = "rsi_ray_tune"
    local_dir: str = "./ray_results"
    
    # Resource configuration
    num_cpus: int = 4
    num_gpus: int = 0
    num_workers: int = 1
    resources_per_trial: Dict[str, float] = field(default_factory=lambda: {"cpu": 1, "gpu": 0})
    
    # Search configuration
    search_algorithm: SearchAlgorithm = SearchAlgorithm.BAYESOPT
    search_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduler configuration
    scheduler_type: SchedulerType = SchedulerType.ASHA
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Trial configuration
    num_samples: int = 100
    max_concurrent_trials: int = 4
    timeout: Optional[float] = None
    max_failures: int = 3
    
    # Stopping criteria
    stop_criteria: Dict[str, Any] = field(default_factory=dict)
    plateau_threshold: float = 0.01
    plateau_num_trials: int = 10
    
    # Checkpointing
    checkpoint_freq: int = 10
    checkpoint_at_end: bool = True
    keep_checkpoints_num: int = 3
    
    # Logging
    verbose: int = 2
    log_to_file: bool = True
    progress_reporter: str = "cli"  # "cli", "jupyter", or "none"
    
    # Integration
    use_mlflow: bool = False
    use_wandb: bool = False
    mlflow_tracking_uri: Optional[str] = None
    wandb_project: Optional[str] = None
    
    # Safety constraints
    trial_timeout: float = 3600.0  # 1 hour per trial
    max_trial_memory: float = 16.0  # GB
    
    # Ray cluster configuration
    ray_address: Optional[str] = None
    ray_runtime_env: Optional[Dict[str, Any]] = None


class RayTuneObjective:
    """Base class for Ray Tune objectives."""
    
    def __init__(self, state_manager: StateManager, validator: RSIValidator):
        self.state_manager = state_manager
        self.validator = validator
        self.audit_logger = get_audit_logger()
    
    def __call__(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Objective function to be optimized."""
        raise NotImplementedError
    
    def setup(self, config: Dict[str, Any]):
        """Setup called once per trial."""
        pass
    
    def cleanup(self):
        """Cleanup called at the end of each trial."""
        pass


class MLModelRayObjective(RayTuneObjective):
    """ML model optimization objective for Ray Tune."""
    
    def __init__(
        self,
        state_manager: StateManager,
        validator: RSIValidator,
        train_data: Any,
        val_data: Any,
        model_class: type,
        evaluation_metric: str = "accuracy"
    ):
        super().__init__(state_manager, validator)
        self.train_data = train_data
        self.val_data = val_data
        self.model_class = model_class
        self.evaluation_metric = evaluation_metric
        self.model = None
        self.training_step = 0
    
    def __call__(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate model with given config."""
        try:
            # Validate configuration
            validation_result = self.validator.validate_performance_metrics(config)
            if not validation_result.valid:
                raise ValueError(f"Invalid configuration: {validation_result.message}")
            
            # Setup model
            self.setup(config)
            
            # Train model
            results = self.train_model(config)
            
            # Cleanup
            self.cleanup()
            
            return results
            
        except Exception as e:
            logger.error(f"Ray Tune trial failed: {e}")
            raise
    
    def setup(self, config: Dict[str, Any]):
        """Setup model for training."""
        # Initialize model with config
        self.model = self._create_model(config)
        self.training_step = 0
    
    def train_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train model and return metrics."""
        # Simulate training process
        num_epochs = config.get('num_epochs', 10)
        learning_rate = config.get('learning_rate', 0.001)
        batch_size = config.get('batch_size', 32)
        
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            # Simulate training epoch
            train_loss = self._simulate_training_epoch(config, epoch)
            
            # Simulate validation
            val_accuracy = self._simulate_validation(config, epoch)
            
            # Update best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                
                # Save checkpoint
                if epoch % 5 == 0:
                    tune.report(
                        accuracy=val_accuracy,
                        loss=train_loss,
                        epoch=epoch,
                        best_accuracy=best_accuracy
                    )
            
            # Early stopping check
            if epoch > 5 and val_accuracy < 0.5:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            self.training_step += 1
        
        # Final results
        final_results = {
            'accuracy': best_accuracy,
            'loss': train_loss,
            'epochs_trained': epoch + 1,
            'training_steps': self.training_step,
            'model_size': config.get('hidden_size', 64) * config.get('num_layers', 2)
        }
        
        return final_results
    
    def _create_model(self, config: Dict[str, Any]):
        """Create model with given configuration."""
        # This is a placeholder - in practice, you'd create the actual model
        return {
            'learning_rate': config.get('learning_rate', 0.001),
            'hidden_size': config.get('hidden_size', 64),
            'num_layers': config.get('num_layers', 2),
            'dropout': config.get('dropout', 0.1)
        }
    
    def _simulate_training_epoch(self, config: Dict[str, Any], epoch: int) -> float:
        """Simulate training epoch."""
        # Simulate decreasing loss over epochs
        base_loss = 1.0
        learning_rate = config.get('learning_rate', 0.001)
        
        # Better learning rate leads to faster convergence
        lr_factor = min(learning_rate * 1000, 1.0)
        epoch_factor = 1.0 / (1 + epoch * 0.1)
        
        loss = base_loss * epoch_factor * (2 - lr_factor)
        loss += np.random.normal(0, 0.1)  # Add noise
        
        return max(loss, 0.01)
    
    def _simulate_validation(self, config: Dict[str, Any], epoch: int) -> float:
        """Simulate validation accuracy."""
        # Simulate improving accuracy over epochs
        base_accuracy = 0.5
        
        # Model complexity affects final accuracy
        hidden_size = config.get('hidden_size', 64)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.1)
        
        # Optimal configuration
        size_factor = min(hidden_size / 128, 1.0)
        layer_factor = min(num_layers / 4, 1.0)
        dropout_factor = 1.0 - abs(dropout - 0.2) / 0.3
        
        # Accuracy improves with training
        epoch_factor = 1.0 - np.exp(-epoch * 0.1)
        
        accuracy = base_accuracy + 0.4 * (size_factor + layer_factor + dropout_factor) / 3 * epoch_factor
        accuracy += np.random.normal(0, 0.02)  # Add noise
        
        return np.clip(accuracy, 0.0, 1.0)
    
    def cleanup(self):
        """Cleanup model resources."""
        self.model = None


class RayTuneOrchestrator:
    """Ray Tune orchestrator for distributed optimization."""
    
    def __init__(
        self,
        config: RayTuneConfig,
        state_manager: StateManager,
        validator: RSIValidator
    ):
        self.config = config
        self.state_manager = state_manager
        self.validator = validator
        self.audit_logger = get_audit_logger()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                address=config.ray_address,
                runtime_env=config.ray_runtime_env,
                num_cpus=config.num_cpus,
                num_gpus=config.num_gpus,
                log_to_driver=config.log_to_file
            )
        
        # Initialize search algorithm
        self.search_algorithm = self._create_search_algorithm()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize reporter
        self.reporter = self._create_reporter()
        
        # Initialize callbacks
        self.callbacks = self._create_callbacks()
        
        # Initialize stopper
        self.stopper = self._create_stopper()
        
        logger.info(f"Ray Tune orchestrator initialized with {config.search_algorithm.value} search")
    
    def _create_search_algorithm(self):
        """Create search algorithm."""
        if self.config.search_algorithm == SearchAlgorithm.RANDOM:
            return None  # Ray Tune uses random search by default
        
        elif self.config.search_algorithm == SearchAlgorithm.GRID:
            return None  # Grid search is handled by parameter specification
        
        elif self.config.search_algorithm == SearchAlgorithm.BAYESOPT:
            search_alg = BayesOptSearch(
                metric="accuracy",
                mode="max",
                **self.config.search_kwargs
            )
            return ConcurrencyLimiter(search_alg, max_concurrent=self.config.max_concurrent_trials)
        
        elif self.config.search_algorithm == SearchAlgorithm.HYPEROPT:
            search_alg = HyperOptSearch(
                metric="accuracy",
                mode="max",
                **self.config.search_kwargs
            )
            return ConcurrencyLimiter(search_alg, max_concurrent=self.config.max_concurrent_trials)
        
        elif self.config.search_algorithm == SearchAlgorithm.OPTUNA:
            search_alg = OptunaSearch(
                metric="accuracy",
                mode="max",
                **self.config.search_kwargs
            )
            return ConcurrencyLimiter(search_alg, max_concurrent=self.config.max_concurrent_trials)
        
        elif self.config.search_algorithm == SearchAlgorithm.AX:
            search_alg = AxSearch(
                metric="accuracy",
                mode="max",
                **self.config.search_kwargs
            )
            return ConcurrencyLimiter(search_alg, max_concurrent=self.config.max_concurrent_trials)
        
        else:
            return None
    
    def _create_scheduler(self):
        """Create scheduler."""
        if self.config.scheduler_type == SchedulerType.FIFO:
            return None  # Default FIFO scheduler
        
        elif self.config.scheduler_type == SchedulerType.ASHA:
            return ASHAScheduler(
                metric="accuracy",
                mode="max",
                max_t=100,
                grace_period=10,
                reduction_factor=2,
                **self.config.scheduler_kwargs
            )
        
        elif self.config.scheduler_type == SchedulerType.HYPERBAND:
            return HyperBandScheduler(
                metric="accuracy",
                mode="max",
                max_t=100,
                **self.config.scheduler_kwargs
            )
        
        elif self.config.scheduler_type == SchedulerType.POPULATION_BASED:
            return PopulationBasedTraining(
                metric="accuracy",
                mode="max",
                perturbation_interval=10,
                hyperparam_mutations={
                    "learning_rate": tune.loguniform(1e-5, 1e-1),
                    "dropout": tune.uniform(0.1, 0.5)
                },
                **self.config.scheduler_kwargs
            )
        
        else:
            return None
    
    def _create_reporter(self):
        """Create progress reporter."""
        if self.config.progress_reporter == "cli":
            return CLIReporter(
                metric_columns=["accuracy", "loss", "epoch", "training_iteration"],
                max_report_frequency=30
            )
        elif self.config.progress_reporter == "jupyter":
            return JupyterNotebookReporter(
                metric_columns=["accuracy", "loss", "epoch", "training_iteration"],
                max_report_frequency=30
            )
        else:
            return None
    
    def _create_callbacks(self) -> List:
        """Create callbacks for logging and monitoring."""
        callbacks = []
        
        # MLflow callback
        if self.config.use_mlflow and MLFLOW_INTEGRATION_AVAILABLE:
            mlflow_callback = MLflowLoggerCallback(
                tracking_uri=self.config.mlflow_tracking_uri,
                experiment_name=self.config.experiment_name,
                save_artifact=True
            )
            callbacks.append(mlflow_callback)
        elif self.config.use_mlflow and not MLFLOW_INTEGRATION_AVAILABLE:
            logger.warning("MLflow integration requested but not available")
        
        # Weights & Biases callback
        if self.config.use_wandb and WANDB_INTEGRATION_AVAILABLE:
            wandb_callback = WandbLoggerCallback(
                project=self.config.wandb_project,
                group=self.config.experiment_name,
                save_checkpoints=True
            )
            callbacks.append(wandb_callback)
        elif self.config.use_wandb and not WANDB_INTEGRATION_AVAILABLE:
            logger.warning("Weights & Biases integration requested but not available")
        
        return callbacks
    
    def _create_stopper(self):
        """Create early stopping criteria."""
        if self.config.stop_criteria:
            return self.config.stop_criteria
        
        # Use plateau stopper by default
        return TrialPlateauStopper(
            metric="accuracy",
            std=self.config.plateau_threshold,
            num_results=self.config.plateau_num_trials,
            grace_period=5,
            metric_threshold=0.9,
            mode="max"
        )
    
    async def optimize(
        self,
        objective_function: RayTuneObjective,
        search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run distributed optimization."""
        try:
            # Log optimization start
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    "ray_tune_orchestrator",
                    "optimization_started",
                    metadata={
                        'experiment_name': self.config.experiment_name,
                        'num_samples': self.config.num_samples,
                        'search_algorithm': self.config.search_algorithm.value,
                        'scheduler': self.config.scheduler_type.value,
                        'search_space': search_space
                    }
                )
            
            # Create trainable function
            trainable = tune.with_parameters(
                objective_function,
                state_manager=self.state_manager,
                validator=self.validator
            )
            
            # Configure resources
            if self.config.num_workers > 1:
                # Multi-worker setup
                resources = PlacementGroupFactory([
                    {"CPU": self.config.resources_per_trial.get("cpu", 1), 
                     "GPU": self.config.resources_per_trial.get("gpu", 0)}
                    for _ in range(self.config.num_workers)
                ])
            else:
                resources = self.config.resources_per_trial
            
            # Run optimization
            analysis = tune.run(
                trainable,
                config=search_space,
                search_alg=self.search_algorithm,
                scheduler=self.scheduler,
                progress_reporter=self.reporter,
                callbacks=self.callbacks,
                stop=self.stopper,
                num_samples=self.config.num_samples,
                resources_per_trial=resources,
                local_dir=self.config.local_dir,
                name=self.config.experiment_name,
                max_failures=self.config.max_failures,
                time_budget_s=self.config.timeout,
                checkpoint_freq=self.config.checkpoint_freq,
                checkpoint_at_end=self.config.checkpoint_at_end,
                keep_checkpoints_num=self.config.keep_checkpoints_num,
                verbose=self.config.verbose,
                raise_on_failed_trial=False
            )
            
            # Process results
            results = self._process_results(analysis)
            
            # Log optimization completion
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    "ray_tune_orchestrator",
                    "optimization_completed",
                    metadata=results
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Ray Tune optimization failed: {e}")
            raise
    
    def _process_results(self, analysis) -> Dict[str, Any]:
        """Process optimization results."""
        # Get best trial
        best_trial = analysis.get_best_trial(metric="accuracy", mode="max")
        
        # Get all trials dataframe
        df = analysis.results_df
        
        # Summary statistics
        results = {
            'experiment_name': self.config.experiment_name,
            'best_config': best_trial.config,
            'best_accuracy': best_trial.last_result.get('accuracy', 0),
            'best_trial_id': best_trial.trial_id,
            'best_checkpoint': best_trial.checkpoint.value if best_trial.checkpoint else None,
            'total_trials': len(df),
            'successful_trials': len(df[df['trial_status'] == 'TERMINATED']),
            'failed_trials': len(df[df['trial_status'] == 'ERROR']),
            'stopped_trials': len(df[df['trial_status'] == 'STOPPED']),
            'total_time': analysis.total_time,
            'best_result': best_trial.last_result,
            'experiment_path': analysis.experiment_path,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add trial history
        results['trial_history'] = []
        for trial in analysis.trials:
            trial_data = {
                'trial_id': trial.trial_id,
                'config': trial.config,
                'status': trial.status,
                'last_result': trial.last_result,
                'logdir': trial.logdir
            }
            results['trial_history'].append(trial_data)
        
        # Performance statistics
        if 'accuracy' in df.columns:
            results['performance_stats'] = {
                'mean_accuracy': df['accuracy'].mean(),
                'std_accuracy': df['accuracy'].std(),
                'min_accuracy': df['accuracy'].min(),
                'max_accuracy': df['accuracy'].max(),
                'accuracy_percentiles': df['accuracy'].quantile([0.25, 0.5, 0.75]).to_dict()
            }
        
        return results
    
    def get_best_config(self, analysis) -> Dict[str, Any]:
        """Get best configuration from analysis."""
        best_trial = analysis.get_best_trial(metric="accuracy", mode="max")
        return best_trial.config
    
    def get_best_checkpoint(self, analysis) -> Optional[str]:
        """Get best checkpoint path."""
        best_trial = analysis.get_best_trial(metric="accuracy", mode="max")
        return best_trial.checkpoint.value if best_trial.checkpoint else None
    
    def save_results(self, analysis, save_path: str):
        """Save optimization results."""
        # Save analysis object
        analysis.save(save_path)
        
        # Save results summary
        results = self._process_results(analysis)
        results_path = Path(save_path) / "results_summary.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Ray Tune results saved to {save_path}")
    
    def load_results(self, load_path: str):
        """Load optimization results."""
        return tune.Analysis(load_path)


def create_ray_tune_orchestrator(
    experiment_name: str,
    search_algorithm: SearchAlgorithm = SearchAlgorithm.BAYESOPT,
    scheduler_type: SchedulerType = SchedulerType.ASHA,
    num_samples: int = 100,
    state_manager: Optional[StateManager] = None,
    validator: Optional[RSIValidator] = None,
    **kwargs
) -> RayTuneOrchestrator:
    """Factory function to create Ray Tune orchestrator."""
    from ..validation.validators import create_strict_validator
    
    config = RayTuneConfig(
        experiment_name=experiment_name,
        search_algorithm=search_algorithm,
        scheduler_type=scheduler_type,
        num_samples=num_samples,
        **kwargs
    )
    
    if not validator:
        validator = create_strict_validator()
    
    return RayTuneOrchestrator(config, state_manager, validator)


def create_ml_ray_objective(
    train_data: Any,
    val_data: Any,
    model_class: type,
    evaluation_metric: str = "accuracy",
    state_manager: Optional[StateManager] = None,
    validator: Optional[RSIValidator] = None
) -> MLModelRayObjective:
    """Factory function to create ML Ray Tune objective."""
    from ..validation.validators import create_strict_validator
    
    if not validator:
        validator = create_strict_validator()
    
    return MLModelRayObjective(
        state_manager,
        validator,
        train_data,
        val_data,
        model_class,
        evaluation_metric
    )


# Common search space definitions
HYPERPARAMETER_SEARCH_SPACES = {
    "neural_network": {
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128, 256]),
        "hidden_size": tune.choice([32, 64, 128, 256, 512]),
        "num_layers": tune.choice([2, 3, 4, 5, 6]),
        "dropout": tune.uniform(0.1, 0.5),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "optimizer": tune.choice(["adam", "sgd", "adamw"]),
        "scheduler": tune.choice(["cosine", "step", "exponential"])
    },
    
    "reinforcement_learning": {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "gamma": tune.uniform(0.9, 0.999),
        "epsilon": tune.uniform(0.01, 0.3),
        "buffer_size": tune.choice([10000, 50000, 100000, 500000]),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "target_update_freq": tune.choice([1000, 5000, 10000]),
        "exploration_fraction": tune.uniform(0.1, 0.3)
    },
    
    "meta_learning": {
        "meta_learning_rate": tune.loguniform(1e-5, 1e-2),
        "adaptation_learning_rate": tune.loguniform(1e-4, 1e-1),
        "adaptation_steps": tune.choice([1, 3, 5, 10]),
        "meta_batch_size": tune.choice([16, 32, 64]),
        "num_ways": tune.choice([3, 5, 10]),
        "num_shots": tune.choice([1, 3, 5, 10]),
        "first_order": tune.choice([True, False])
    }
}