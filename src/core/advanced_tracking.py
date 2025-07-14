"""
Advanced Experiment Tracking System.
Integrates Weights & Biases, Neptune.ai, and DVC for comprehensive ML lifecycle management.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
import tempfile
import pickle

# Import multiple tracking systems
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False

try:
    import dvc.api
    DVC_AVAILABLE = True
except ImportError:
    DVC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    project_name: str
    experiment_name: str
    tags: List[str] = None
    description: str = None
    
    # W&B settings
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # online, offline, disabled
    
    # Neptune settings
    neptune_project: Optional[str] = None
    neptune_api_token: Optional[str] = None
    neptune_mode: str = "async"  # async, sync, debug, offline
    
    # DVC settings
    dvc_remote: Optional[str] = None
    dvc_repo: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class AdvancedExperimentTracker:
    """Advanced experiment tracking using multiple platforms."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.active_run = None
        self.active_trackers = {}
        
        # Initialize available trackers
        self._init_wandb()
        self._init_neptune()
        self._init_dvc()
        
        logger.info(f"Advanced tracker initialized with {len(self.active_trackers)} systems")
    
    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if not WANDB_AVAILABLE:
            return
        
        try:
            # Check if we're in offline mode or if API key is available
            if self.config.wandb_mode == "offline" or os.getenv("WANDB_API_KEY"):
                wandb.init(
                    project=self.config.wandb_project or self.config.project_name,
                    entity=self.config.wandb_entity,
                    name=self.config.experiment_name,
                    tags=self.config.tags,
                    notes=self.config.description,
                    mode=self.config.wandb_mode,
                    reinit=True
                )
                self.active_trackers['wandb'] = wandb
                logger.info("✅ W&B tracker initialized")
            else:
                logger.info("⚠️  W&B skipped (no API key, use offline mode)")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")
    
    def _init_neptune(self):
        """Initialize Neptune.ai."""
        if not NEPTUNE_AVAILABLE:
            return
        
        try:
            # Check if API token is available
            api_token = self.config.neptune_api_token or os.getenv("NEPTUNE_API_TOKEN")
            if api_token or self.config.neptune_mode in ["debug", "offline"]:
                run = neptune.init_run(
                    project=self.config.neptune_project or self.config.project_name,
                    api_token=api_token,
                    name=self.config.experiment_name,
                    tags=self.config.tags,
                    description=self.config.description,
                    mode=self.config.neptune_mode
                )
                self.active_trackers['neptune'] = run
                logger.info("✅ Neptune tracker initialized")
            else:
                logger.info("⚠️  Neptune skipped (no API token)")
        except Exception as e:
            logger.warning(f"Neptune initialization failed: {e}")
    
    def _init_dvc(self):
        """Initialize DVC."""
        if not DVC_AVAILABLE:
            return
        
        try:
            # Check if we're in a DVC repo
            if Path('.dvc').exists() or self.config.dvc_repo:
                self.active_trackers['dvc'] = dvc.api
                logger.info("✅ DVC tracker initialized")
            else:
                logger.info("⚠️  DVC skipped (no DVC repo)")
        except Exception as e:
            logger.warning(f"DVC initialization failed: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters across all active trackers."""
        for tracker_name, tracker in self.active_trackers.items():
            try:
                if tracker_name == 'wandb':
                    tracker.config.update(params)
                elif tracker_name == 'neptune':
                    for key, value in params.items():
                        tracker[f"parameters/{key}"] = value
                elif tracker_name == 'dvc':
                    # DVC handles parameters through params.yaml
                    self._save_dvc_params(params)
                    
            except Exception as e:
                logger.warning(f"Failed to log params to {tracker_name}: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics across all active trackers."""
        for tracker_name, tracker in self.active_trackers.items():
            try:
                if tracker_name == 'wandb':
                    tracker.log(metrics, step=step)
                elif tracker_name == 'neptune':
                    for key, value in metrics.items():
                        if step is not None:
                            tracker[f"metrics/{key}"].append(value, step=step)
                        else:
                            tracker[f"metrics/{key}"].append(value)
                elif tracker_name == 'dvc':
                    # DVC handles metrics through metrics.json
                    self._save_dvc_metrics(metrics, step)
                    
            except Exception as e:
                logger.warning(f"Failed to log metrics to {tracker_name}: {e}")
    
    def log_artifact(self, file_path: str, artifact_type: str = "model"):
        """Log artifacts across all active trackers."""
        for tracker_name, tracker in self.active_trackers.items():
            try:
                if tracker_name == 'wandb':
                    artifact = tracker.Artifact(
                        name=f"{artifact_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        type=artifact_type
                    )
                    artifact.add_file(file_path)
                    tracker.log_artifact(artifact)
                    
                elif tracker_name == 'neptune':
                    tracker[f"artifacts/{artifact_type}"].upload(file_path)
                    
                elif tracker_name == 'dvc':
                    # DVC handles artifacts through dvc add
                    self._save_dvc_artifact(file_path, artifact_type)
                    
            except Exception as e:
                logger.warning(f"Failed to log artifact to {tracker_name}: {e}")
    
    def log_model(self, model: Any, model_name: str, metadata: Dict[str, Any] = None):
        """Log model across all active trackers."""
        # Save model to temporary file
        temp_dir = Path(tempfile.mkdtemp())
        model_path = temp_dir / f"{model_name}.pkl"
        
        try:
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Log to all trackers
            for tracker_name, tracker in self.active_trackers.items():
                try:
                    if tracker_name == 'wandb':
                        # Create model artifact
                        model_artifact = tracker.Artifact(
                            name=model_name,
                            type="model",
                            description=metadata.get('description', ''),
                            metadata=metadata or {}
                        )
                        model_artifact.add_file(str(model_path))
                        tracker.log_artifact(model_artifact)
                        
                    elif tracker_name == 'neptune':
                        tracker[f"models/{model_name}"].upload(str(model_path))
                        if metadata:
                            for key, value in metadata.items():
                                tracker[f"models/{model_name}/metadata/{key}"] = value
                                
                    elif tracker_name == 'dvc':
                        # DVC model tracking
                        self._save_dvc_model(model_path, model_name, metadata)
                        
                except Exception as e:
                    logger.warning(f"Failed to log model to {tracker_name}: {e}")
                    
        finally:
            # Cleanup
            if model_path.exists():
                model_path.unlink()
            temp_dir.rmdir()
    
    def log_code(self, code_dir: str = "."):
        """Log code across all active trackers."""
        for tracker_name, tracker in self.active_trackers.items():
            try:
                if tracker_name == 'wandb':
                    tracker.run.log_code(code_dir)
                elif tracker_name == 'neptune':
                    tracker["source_code"].upload_files(code_dir)
                elif tracker_name == 'dvc':
                    # DVC handles code through git integration
                    pass
                    
            except Exception as e:
                logger.warning(f"Failed to log code to {tracker_name}: {e}")
    
    def log_system_info(self):
        """Log system information."""
        import psutil
        import platform
        
        system_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_gb": round(psutil.disk_usage('/').total / (1024**3), 2)
        }
        
        self.log_params({"system": system_info})
    
    def _save_dvc_params(self, params: Dict[str, Any]):
        """Save parameters for DVC tracking."""
        params_file = Path("params.yaml")
        
        # Load existing params or create new
        existing_params = {}
        if params_file.exists():
            try:
                import yaml
                with open(params_file, 'r') as f:
                    existing_params = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load existing params: {e}")
        
        # Update with new params
        existing_params.update(params)
        
        # Save updated params
        try:
            import yaml
            with open(params_file, 'w') as f:
                yaml.safe_dump(existing_params, f)
        except Exception as e:
            logger.warning(f"Failed to save DVC params: {e}")
    
    def _save_dvc_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int]):
        """Save metrics for DVC tracking."""
        metrics_file = Path("metrics.json")
        
        # Load existing metrics or create new
        existing_metrics = {}
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing metrics: {e}")
        
        # Update with new metrics
        if step is not None:
            if "steps" not in existing_metrics:
                existing_metrics["steps"] = {}
            existing_metrics["steps"][str(step)] = metrics
        else:
            existing_metrics.update(metrics)
        
        # Save updated metrics
        try:
            with open(metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save DVC metrics: {e}")
    
    def _save_dvc_artifact(self, file_path: str, artifact_type: str):
        """Save artifact for DVC tracking."""
        try:
            # Create artifacts directory
            artifacts_dir = Path("artifacts") / artifact_type
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file to artifacts
            import shutil
            dest_path = artifacts_dir / Path(file_path).name
            shutil.copy2(file_path, dest_path)
            
            # Add to DVC if available
            if 'dvc' in self.active_trackers:
                try:
                    import subprocess
                    subprocess.run(['dvc', 'add', str(dest_path)], check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"DVC add failed: {e}")
                except FileNotFoundError:
                    logger.warning("DVC not found in PATH")
                
        except Exception as e:
            logger.warning(f"Failed to save DVC artifact: {e}")
    
    def _save_dvc_model(self, model_path: Path, model_name: str, metadata: Dict[str, Any]):
        """Save model for DVC tracking."""
        try:
            # Create models directory
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Copy model to models directory
            import shutil
            dest_path = models_dir / f"{model_name}.pkl"
            shutil.copy2(model_path, dest_path)
            
            # Save metadata
            if metadata:
                metadata_path = models_dir / f"{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Add to DVC if available
            if 'dvc' in self.active_trackers:
                try:
                    import subprocess
                    subprocess.run(['dvc', 'add', str(dest_path)], check=True, capture_output=True)
                    if metadata:
                        subprocess.run(['dvc', 'add', str(metadata_path)], check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"DVC add failed: {e}")
                except FileNotFoundError:
                    logger.warning("DVC not found in PATH")
                    
        except Exception as e:
            logger.warning(f"Failed to save DVC model: {e}")
    
    def finish(self):
        """Finish all active tracking runs."""
        for tracker_name, tracker in self.active_trackers.items():
            try:
                if tracker_name == 'wandb':
                    tracker.finish()
                elif tracker_name == 'neptune':
                    tracker.stop()
                elif tracker_name == 'dvc':
                    # DVC doesn't need explicit finish
                    pass
                    
            except Exception as e:
                logger.warning(f"Failed to finish {tracker_name}: {e}")
        
        logger.info("All tracking runs finished")
    
    def get_run_url(self) -> Dict[str, str]:
        """Get URLs for all active runs."""
        urls = {}
        
        for tracker_name, tracker in self.active_trackers.items():
            try:
                if tracker_name == 'wandb':
                    urls['wandb'] = tracker.run.get_url()
                elif tracker_name == 'neptune':
                    urls['neptune'] = tracker.get_url()
                elif tracker_name == 'dvc':
                    urls['dvc'] = "Local DVC repository"
                    
            except Exception as e:
                logger.warning(f"Failed to get URL for {tracker_name}: {e}")
        
        return urls
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        comparison_data = {}
        
        for tracker_name, tracker in self.active_trackers.items():
            try:
                if tracker_name == 'wandb':
                    # W&B comparison logic
                    api = wandb.Api()
                    runs = [api.run(f"{self.config.wandb_project}/{run_id}") for run_id in run_ids]
                    comparison_data['wandb'] = {
                        'runs': [run.summary for run in runs],
                        'configs': [run.config for run in runs]
                    }
                    
                elif tracker_name == 'neptune':
                    # Neptune comparison logic
                    project = neptune.init_project(
                        project=self.config.neptune_project,
                        api_token=self.config.neptune_api_token
                    )
                    runs_df = project.fetch_runs_table().to_pandas()
                    comparison_data['neptune'] = runs_df[runs_df['sys/id'].isin(run_ids)]
                    
            except Exception as e:
                logger.warning(f"Failed to compare runs for {tracker_name}: {e}")
        
        return comparison_data


def create_advanced_tracker(
    project_name: str,
    experiment_name: str,
    **kwargs
) -> AdvancedExperimentTracker:
    """Factory function to create advanced experiment tracker."""
    config = ExperimentConfig(
        project_name=project_name,
        experiment_name=experiment_name,
        **kwargs
    )
    
    return AdvancedExperimentTracker(config)