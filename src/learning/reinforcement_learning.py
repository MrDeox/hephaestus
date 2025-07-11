"""
Reinforcement Learning integration for RSI systems using Stable-Baselines3.
Provides production-ready RL algorithms with continuous policy improvement.
"""

import asyncio
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import logging
import pickle
import os
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, A2C, TD3, DQN, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

from ..core.state import StateManager, RSIState
from ..validation.validators import RSIValidator
from ..monitoring.audit_logger import get_audit_logger


logger = logging.getLogger(__name__)


class RLAlgorithm(Enum):
    """Supported reinforcement learning algorithms."""
    PPO = "ppo"
    SAC = "sac"
    A2C = "a2c"
    TD3 = "td3"
    DQN = "dqn"
    DDPG = "ddpg"


class RSITaskType(Enum):
    """Types of RSI tasks for RL."""
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    ARCHITECTURE_SEARCH = "architecture_search"
    CURRICULUM_LEARNING = "curriculum_learning"
    RESOURCE_ALLOCATION = "resource_allocation"
    STRATEGY_SELECTION = "strategy_selection"


@dataclass
class RLConfig:
    """Configuration for reinforcement learning system."""
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    task_type: RSITaskType = RSITaskType.HYPERPARAMETER_OPTIMIZATION
    
    # Training configuration
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    
    # Environment configuration
    env_name: str = "RSI-v0"
    n_envs: int = 4
    normalize_env: bool = True
    
    # Algorithm-specific parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Evaluation configuration
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    eval_deterministic: bool = True
    
    # Safety constraints
    max_episode_steps: int = 1000
    reward_threshold: float = 500.0
    min_reward_threshold: float = -1000.0
    
    # Logging and monitoring
    log_interval: int = 1000
    save_freq: int = 50000
    verbose: int = 1
    
    # Model configuration
    policy_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.policy_kwargs is None:
            self.policy_kwargs = {
                'net_arch': [64, 64],
                'activation_fn': torch.nn.ReLU
            }


class RSIEnvironment(gym.Env):
    """Custom environment for RSI tasks."""
    
    def __init__(
        self,
        task_type: RSITaskType,
        state_manager: StateManager,
        validator: RSIValidator,
        config: Dict[str, Any] = None
    ):
        super().__init__()
        
        self.task_type = task_type
        self.state_manager = state_manager
        self.validator = validator
        self.config = config or {}
        
        # Initialize action and observation spaces based on task type
        self._initialize_spaces()
        
        # Initialize task-specific components
        self._initialize_task_components()
        
        # State tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.best_reward = float('-inf')
        
        # Safety constraints
        self.max_episode_steps = self.config.get('max_episode_steps', 1000)
        self.reward_threshold = self.config.get('reward_threshold', 500.0)
        
        logger.info(f"RSI Environment initialized for {task_type.value}")
    
    def _initialize_spaces(self):
        """Initialize action and observation spaces."""
        if self.task_type == RSITaskType.HYPERPARAMETER_OPTIMIZATION:
            # Continuous action space for hyperparameter values
            self.action_space = spaces.Box(
                low=np.array([0.0001, 0.0001, 0.1, 0.5]),  # [lr, weight_decay, dropout, momentum]
                high=np.array([0.1, 0.01, 0.9, 0.99]),
                dtype=np.float32
            )
            
            # Observation space: current metrics + hyperparameters
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),  # [accuracy, loss, lr, wd, dropout, momentum, epoch, samples, time, memory]
                dtype=np.float32
            )
            
        elif self.task_type == RSITaskType.ARCHITECTURE_SEARCH:
            # Discrete action space for architecture choices
            self.action_space = spaces.MultiDiscrete([
                5,  # Number of layers (1-5)
                4,  # Layer width multiplier (1, 2, 4, 8)
                3,  # Activation function (ReLU, Tanh, GELU)
                2,  # Dropout (True/False)
                3   # Optimizer (Adam, SGD, RMSprop)
            ])
            
            # Observation space: current architecture + metrics
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(15,),
                dtype=np.float32
            )
            
        elif self.task_type == RSITaskType.RESOURCE_ALLOCATION:
            # Continuous action space for resource allocation
            self.action_space = spaces.Box(
                low=np.array([0.1, 0.1, 0.1, 0.1]),  # [cpu, memory, gpu, disk]
                high=np.array([1.0, 1.0, 1.0, 1.0]),
                dtype=np.float32
            )
            
            # Observation space: current usage + performance metrics
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(12,),
                dtype=np.float32
            )
            
        else:
            # Default spaces
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    
    def _initialize_task_components(self):
        """Initialize task-specific components."""
        self.task_state = {
            'current_config': None,
            'best_config': None,
            'performance_history': [],
            'iteration_count': 0
        }
        
        # Task-specific initialization
        if self.task_type == RSITaskType.HYPERPARAMETER_OPTIMIZATION:
            self.task_state['current_config'] = {
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'dropout': 0.5,
                'momentum': 0.9
            }
            
        elif self.task_type == RSITaskType.ARCHITECTURE_SEARCH:
            self.task_state['current_config'] = {
                'num_layers': 3,
                'width_multiplier': 2,
                'activation': 'relu',
                'use_dropout': True,
                'optimizer': 'adam'
            }
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        
        # Reset task state
        self.task_state['iteration_count'] = 0
        self.task_state['performance_history'] = []
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        """Execute one step in the environment."""
        self.current_step += 1
        self.episode_length += 1
        
        # Apply action to update configuration
        reward = self._apply_action(action)
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Additional info
        info = {
            'current_step': self.current_step,
            'episode_reward': self.episode_reward,
            'best_reward': self.best_reward,
            'task_config': self.task_state['current_config'].copy()
        }
        
        # Update episode reward
        self.episode_reward += reward
        
        # Update best reward
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self.task_state['best_config'] = self.task_state['current_config'].copy()
        
        # Truncated (time limit)
        truncated = self.episode_length >= self.max_episode_steps
        
        return observation, reward, done, truncated, info
    
    def _apply_action(self, action) -> float:
        """Apply action and return reward."""
        if self.task_type == RSITaskType.HYPERPARAMETER_OPTIMIZATION:
            return self._apply_hyperparameter_action(action)
        elif self.task_type == RSITaskType.ARCHITECTURE_SEARCH:
            return self._apply_architecture_action(action)
        elif self.task_type == RSITaskType.RESOURCE_ALLOCATION:
            return self._apply_resource_action(action)
        else:
            return 0.0
    
    def _apply_hyperparameter_action(self, action) -> float:
        """Apply hyperparameter optimization action."""
        # Update hyperparameters
        self.task_state['current_config'] = {
            'learning_rate': float(action[0]),
            'weight_decay': float(action[1]),
            'dropout': float(action[2]),
            'momentum': float(action[3])
        }
        
        # Simulate training with new hyperparameters
        # In practice, this would trigger actual training
        performance = self._simulate_training_performance()
        
        # Calculate reward based on performance improvement
        reward = self._calculate_performance_reward(performance)
        
        return reward
    
    def _apply_architecture_action(self, action) -> float:
        """Apply architecture search action."""
        # Update architecture configuration
        self.task_state['current_config'] = {
            'num_layers': int(action[0]) + 1,
            'width_multiplier': 2 ** int(action[1]),
            'activation': ['relu', 'tanh', 'gelu'][int(action[2])],
            'use_dropout': bool(action[3]),
            'optimizer': ['adam', 'sgd', 'rmsprop'][int(action[4])]
        }
        
        # Simulate architecture performance
        performance = self._simulate_architecture_performance()
        
        # Calculate reward
        reward = self._calculate_performance_reward(performance)
        
        return reward
    
    def _apply_resource_action(self, action) -> float:
        """Apply resource allocation action."""
        # Update resource allocation
        self.task_state['current_config'] = {
            'cpu_allocation': float(action[0]),
            'memory_allocation': float(action[1]),
            'gpu_allocation': float(action[2]),
            'disk_allocation': float(action[3])
        }
        
        # Simulate resource utilization efficiency
        performance = self._simulate_resource_efficiency()
        
        # Calculate reward
        reward = self._calculate_resource_reward(performance)
        
        return reward
    
    def _simulate_training_performance(self) -> Dict[str, float]:
        """Simulate training performance with current hyperparameters."""
        config = self.task_state['current_config']
        
        # Simulate accuracy based on hyperparameters
        # This is a simplified simulation - in practice, you'd run actual training
        lr = config['learning_rate']
        wd = config['weight_decay']
        dropout = config['dropout']
        
        # Optimal ranges for hyperparameters
        lr_factor = 1.0 - abs(lr - 0.001) / 0.001
        wd_factor = 1.0 - abs(wd - 0.0001) / 0.0001
        dropout_factor = 1.0 - abs(dropout - 0.5) / 0.5
        
        # Simulate accuracy with some noise
        base_accuracy = 0.7
        accuracy = base_accuracy + 0.2 * (lr_factor + wd_factor + dropout_factor) / 3
        accuracy += np.random.normal(0, 0.02)  # Add noise
        accuracy = np.clip(accuracy, 0.0, 1.0)
        
        # Simulate loss
        loss = 1.0 - accuracy + np.random.normal(0, 0.1)
        loss = np.clip(loss, 0.0, 2.0)
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'training_time': np.random.uniform(10, 100),
            'memory_usage': np.random.uniform(0.5, 1.0)
        }
    
    def _simulate_architecture_performance(self) -> Dict[str, float]:
        """Simulate architecture performance."""
        config = self.task_state['current_config']
        
        # Simulate performance based on architecture choices
        num_layers = config['num_layers']
        width_mult = config['width_multiplier']
        
        # Optimal architecture has 3-4 layers with 2-4x width
        layer_factor = 1.0 - abs(num_layers - 3.5) / 3.5
        width_factor = 1.0 - abs(width_mult - 3) / 3
        
        base_accuracy = 0.75
        accuracy = base_accuracy + 0.15 * (layer_factor + width_factor) / 2
        accuracy += np.random.normal(0, 0.03)
        accuracy = np.clip(accuracy, 0.0, 1.0)
        
        # Model complexity affects training time
        complexity = num_layers * width_mult
        training_time = complexity * 5 + np.random.uniform(0, 10)
        
        return {
            'accuracy': accuracy,
            'loss': 1.0 - accuracy,
            'training_time': training_time,
            'model_size': complexity * 1000,
            'inference_time': complexity * 0.1
        }
    
    def _simulate_resource_efficiency(self) -> Dict[str, float]:
        """Simulate resource utilization efficiency."""
        config = self.task_state['current_config']
        
        # Calculate resource utilization efficiency
        cpu_util = config['cpu_allocation']
        memory_util = config['memory_allocation']
        gpu_util = config['gpu_allocation']
        
        # Efficiency is best when resources are balanced
        resource_balance = 1.0 - np.std([cpu_util, memory_util, gpu_util])
        
        # Simulate throughput based on resource allocation
        throughput = (cpu_util + memory_util + gpu_util) / 3 * resource_balance
        throughput += np.random.normal(0, 0.1)
        throughput = np.clip(throughput, 0.0, 1.0)
        
        return {
            'throughput': throughput,
            'resource_efficiency': resource_balance,
            'cost': sum(config.values()) * 100,
            'utilization': (cpu_util + memory_util + gpu_util) / 3
        }
    
    def _calculate_performance_reward(self, performance: Dict[str, float]) -> float:
        """Calculate reward based on performance metrics."""
        accuracy = performance['accuracy']
        loss = performance['loss']
        
        # Base reward from accuracy
        reward = accuracy * 10
        
        # Penalty for high loss
        reward -= loss * 5
        
        # Bonus for improvement over previous best
        if hasattr(self, 'best_accuracy'):
            if accuracy > self.best_accuracy:
                reward += (accuracy - self.best_accuracy) * 20
        
        self.best_accuracy = max(getattr(self, 'best_accuracy', 0), accuracy)
        
        # Add performance to history
        self.task_state['performance_history'].append(performance)
        
        return float(reward)
    
    def _calculate_resource_reward(self, performance: Dict[str, float]) -> float:
        """Calculate reward based on resource efficiency."""
        throughput = performance['throughput']
        efficiency = performance['resource_efficiency']
        cost = performance['cost']
        
        # Reward for high throughput and efficiency
        reward = throughput * 10 + efficiency * 5
        
        # Penalty for high cost
        reward -= cost / 100
        
        return float(reward)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.task_type == RSITaskType.HYPERPARAMETER_OPTIMIZATION:
            return self._get_hyperparameter_observation()
        elif self.task_type == RSITaskType.ARCHITECTURE_SEARCH:
            return self._get_architecture_observation()
        elif self.task_type == RSITaskType.RESOURCE_ALLOCATION:
            return self._get_resource_observation()
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _get_hyperparameter_observation(self) -> np.ndarray:
        """Get observation for hyperparameter optimization."""
        config = self.task_state['current_config']
        
        # Get recent performance
        if self.task_state['performance_history']:
            latest_perf = self.task_state['performance_history'][-1]
            accuracy = latest_perf['accuracy']
            loss = latest_perf['loss']
            training_time = latest_perf['training_time']
            memory_usage = latest_perf['memory_usage']
        else:
            accuracy = 0.5
            loss = 0.5
            training_time = 50.0
            memory_usage = 0.5
        
        observation = np.array([
            accuracy,
            loss,
            config['learning_rate'],
            config['weight_decay'],
            config['dropout'],
            config['momentum'],
            float(self.current_step) / self.max_episode_steps,
            float(len(self.task_state['performance_history'])) / 100,
            training_time / 100,
            memory_usage
        ], dtype=np.float32)
        
        return observation
    
    def _get_architecture_observation(self) -> np.ndarray:
        """Get observation for architecture search."""
        config = self.task_state['current_config']
        
        # Get recent performance
        if self.task_state['performance_history']:
            latest_perf = self.task_state['performance_history'][-1]
            accuracy = latest_perf['accuracy']
            loss = latest_perf['loss']
            training_time = latest_perf['training_time']
            model_size = latest_perf['model_size']
            inference_time = latest_perf['inference_time']
        else:
            accuracy = 0.5
            loss = 0.5
            training_time = 50.0
            model_size = 1000.0
            inference_time = 1.0
        
        observation = np.array([
            accuracy,
            loss,
            float(config['num_layers']) / 5,
            float(config['width_multiplier']) / 8,
            1.0 if config['activation'] == 'relu' else 0.0,
            1.0 if config['activation'] == 'tanh' else 0.0,
            1.0 if config['activation'] == 'gelu' else 0.0,
            1.0 if config['use_dropout'] else 0.0,
            1.0 if config['optimizer'] == 'adam' else 0.0,
            1.0 if config['optimizer'] == 'sgd' else 0.0,
            1.0 if config['optimizer'] == 'rmsprop' else 0.0,
            training_time / 100,
            model_size / 10000,
            inference_time / 10,
            float(self.current_step) / self.max_episode_steps
        ], dtype=np.float32)
        
        return observation
    
    def _get_resource_observation(self) -> np.ndarray:
        """Get observation for resource allocation."""
        config = self.task_state['current_config']
        
        # Get recent performance
        if self.task_state['performance_history']:
            latest_perf = self.task_state['performance_history'][-1]
            throughput = latest_perf['throughput']
            efficiency = latest_perf['resource_efficiency']
            cost = latest_perf['cost']
            utilization = latest_perf['utilization']
        else:
            throughput = 0.5
            efficiency = 0.5
            cost = 100.0
            utilization = 0.5
        
        observation = np.array([
            throughput,
            efficiency,
            cost / 1000,
            utilization,
            config['cpu_allocation'],
            config['memory_allocation'],
            config['gpu_allocation'],
            config['disk_allocation'],
            float(self.current_step) / self.max_episode_steps,
            float(len(self.task_state['performance_history'])) / 100,
            np.random.uniform(0.5, 1.0),  # Simulated current load
            np.random.uniform(0.3, 0.8)   # Simulated system efficiency
        ], dtype=np.float32)
        
        return observation
    
    def _is_episode_done(self) -> bool:
        """Check if episode is done."""
        # Done if we've reached reward threshold
        if self.episode_reward >= self.reward_threshold:
            return True
        
        # Done if reward is too low (safety constraint)
        if self.episode_reward <= self.config.get('min_reward_threshold', -1000):
            return True
        
        return False


class RSICallback(BaseCallback):
    """Custom callback for RSI reinforcement learning."""
    
    def __init__(self, state_manager: StateManager, audit_logger, verbose=0):
        super().__init__(verbose)
        self.state_manager = state_manager
        self.audit_logger = audit_logger
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = float('-inf')
        
    def _on_step(self) -> bool:
        """Called after each step."""
        # Log training metrics
        if self.locals.get('done', False):
            episode_reward = self.locals.get('episode_reward', 0)
            episode_length = self.locals.get('episode_length', 0)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Update best reward
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                
                # Log improvement
                if self.audit_logger:
                    self.audit_logger.log_system_event(
                        "rl_system",
                        "new_best_reward",
                        metadata={
                            'reward': episode_reward,
                            'episode_length': episode_length,
                            'total_steps': self.num_timesteps
                        }
                    )
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.audit_logger:
            self.audit_logger.log_system_event(
                "rl_system",
                "training_completed",
                metadata={
                    'total_episodes': len(self.episode_rewards),
                    'best_reward': self.best_reward,
                    'mean_reward': np.mean(self.episode_rewards[-100:]),
                    'mean_length': np.mean(self.episode_lengths[-100:])
                }
            )


class ReinforcementLearningSystem:
    """Reinforcement Learning system for RSI."""
    
    def __init__(
        self,
        config: RLConfig,
        state_manager: StateManager,
        validator: RSIValidator
    ):
        self.config = config
        self.state_manager = state_manager
        self.validator = validator
        self.audit_logger = get_audit_logger()
        
        # Initialize environment
        self.env = RSIEnvironment(
            config.task_type,
            state_manager,
            validator,
            config.__dict__
        )
        
        # Wrap environment for monitoring
        self.env = Monitor(self.env, filename=f"./logs/rl_{config.task_type.value}.log")
        
        # Create vectorized environment with proper env creation function
        def make_env():
            env = RSIEnvironment(
                config.task_type,
                state_manager,
                validator,
                config.__dict__
            )
            return Monitor(env, filename=f"./logs/rl_{config.task_type.value}.log")
        
        self.vec_env = make_vec_env(
            make_env,
            n_envs=config.n_envs,
            seed=42
        )
        
        # Normalize environment if requested
        if config.normalize_env:
            self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True)
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize callbacks
        self.callbacks = self._create_callbacks()
        
        logger.info(f"RL system initialized with {config.algorithm.value} for {config.task_type.value}")
    
    def _create_model(self):
        """Create the RL model."""
        algorithm_class = {
            RLAlgorithm.PPO: PPO,
            RLAlgorithm.SAC: SAC,
            RLAlgorithm.A2C: A2C,
            RLAlgorithm.TD3: TD3,
            RLAlgorithm.DQN: DQN,
            RLAlgorithm.DDPG: DDPG
        }[self.config.algorithm]
        
        # Algorithm-specific parameters
        model_kwargs = {
            'policy': 'MlpPolicy',  # Use Multi-Layer Perceptron policy
            'env': self.vec_env,
            'learning_rate': self.config.learning_rate,
            'verbose': self.config.verbose,
            'policy_kwargs': self.config.policy_kwargs,
            'tensorboard_log': f"./logs/rl_tensorboard_{self.config.task_type.value}/"
        }
        
        # Add algorithm-specific parameters
        if self.config.algorithm == RLAlgorithm.PPO:
            model_kwargs.update({
                'n_steps': self.config.n_steps,
                'batch_size': self.config.batch_size,
                'n_epochs': self.config.n_epochs,
                'gamma': self.config.gamma,
                'gae_lambda': self.config.gae_lambda,
                'clip_range': self.config.clip_range,
                'ent_coef': self.config.ent_coef,
                'vf_coef': self.config.vf_coef,
                'max_grad_norm': self.config.max_grad_norm
            })
        
        elif self.config.algorithm == RLAlgorithm.SAC:
            model_kwargs.update({
                'buffer_size': 1000000,
                'gamma': self.config.gamma,
                'tau': 0.005,
                'batch_size': self.config.batch_size
            })
        
        return algorithm_class(**model_kwargs)
    
    def _create_callbacks(self) -> List[BaseCallback]:
        """Create training callbacks."""
        callbacks = []
        
        # Custom RSI callback
        rsi_callback = RSICallback(self.state_manager, self.audit_logger)
        callbacks.append(rsi_callback)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.vec_env,
            best_model_save_path=f"./models/rl_best_{self.config.task_type.value}",
            log_path=f"./logs/rl_eval_{self.config.task_type.value}",
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=self.config.eval_deterministic,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Stop training on reward threshold
        if self.config.reward_threshold > 0:
            stop_callback = StopTrainingOnRewardThreshold(
                reward_threshold=self.config.reward_threshold,
                verbose=self.config.verbose
            )
            callbacks.append(stop_callback)
        
        return callbacks
    
    async def train(self) -> Dict[str, Any]:
        """Train the RL model."""
        try:
            # Validate configuration
            validation_result = self.validator.validate_performance_metrics({
                'total_timesteps': self.config.total_timesteps,
                'learning_rate': self.config.learning_rate
            })
            
            if not validation_result.valid:
                raise ValueError(f"Invalid RL configuration: {validation_result.message}")
            
            # Train the model
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=self.callbacks,
                log_interval=self.config.log_interval,
                progress_bar=True
            )
            
            # Save the model
            model_path = f"./models/rl_final_{self.config.task_type.value}"
            self.model.save(model_path)
            
            # Evaluate final model
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.vec_env,
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=self.config.eval_deterministic
            )
            
            training_results = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'total_timesteps': self.config.total_timesteps,
                'model_path': model_path,
                'task_type': self.config.task_type.value,
                'algorithm': self.config.algorithm.value,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Log training completion
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    "rl_system",
                    "training_completed",
                    metadata=training_results
                )
            
            return training_results
            
        except Exception as e:
            logger.error(f"RL training failed: {e}")
            raise
    
    async def optimize_task(self, task_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Use trained model to optimize a specific task."""
        try:
            # Reset environment with task parameters
            observation, _ = self.env.reset()
            
            # Run optimization episode
            episode_rewards = []
            episode_actions = []
            best_config = None
            best_reward = float('-inf')
            
            for step in range(self.config.max_episode_steps):
                # Get action from model
                action, _ = self.model.predict(observation, deterministic=True)
                
                # Take step
                observation, reward, done, truncated, info = self.env.step(action)
                
                episode_rewards.append(reward)
                episode_actions.append(action.tolist())
                
                # Track best configuration
                if reward > best_reward:
                    best_reward = reward
                    best_config = info.get('task_config', {})
                
                if done or truncated:
                    break
            
            optimization_results = {
                'best_config': best_config,
                'best_reward': best_reward,
                'total_reward': sum(episode_rewards),
                'episode_length': len(episode_rewards),
                'actions': episode_actions,
                'task_type': self.config.task_type.value,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Log optimization
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    "rl_system",
                    "task_optimization",
                    metadata=optimization_results
                )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Task optimization failed: {e}")
            raise
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.model = self.model.load(path, env=self.vec_env)
        logger.info(f"Model loaded from {path}")
    
    def save_model(self, path: str):
        """Save the current model."""
        self.model.save(path)
        logger.info(f"Model saved to {path}")


def create_rl_system(
    task_type: RSITaskType,
    algorithm: RLAlgorithm = RLAlgorithm.PPO,
    state_manager: Optional[StateManager] = None,
    validator: Optional[RSIValidator] = None,
    **kwargs
) -> ReinforcementLearningSystem:
    """Factory function to create RL system."""
    from ..validation.validators import create_strict_validator
    
    config = RLConfig(algorithm=algorithm, task_type=task_type, **kwargs)
    
    if not validator:
        validator = create_strict_validator()
    
    return ReinforcementLearningSystem(config, state_manager, validator)