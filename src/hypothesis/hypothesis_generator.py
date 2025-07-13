"""
Advanced RSI Hypothesis Generation using Optuna and Ray.
Implements sophisticated hypothesis exploration with safety constraints.
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
import numpy as np
from loguru import logger

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available - using fallback hypothesis generation")

try:
    import ray
    from ray import tune
    from ray.air import ScalingConfig, RunConfig
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available - hypothesis generation will be single-threaded")

from ..core.state import RSIStateManager
from ..validation.validators import RSIValidator
from ..safety.circuits import RSICircuitBreaker


class HypothesisType(str, Enum):
    ARCHITECTURE_CHANGE = "architecture_change"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization" 
    ALGORITHM_MODIFICATION = "algorithm_modification"
    FEATURE_ENGINEERING = "feature_engineering"
    ENSEMBLE_STRATEGY = "ensemble_strategy"
    SAFETY_ENHANCEMENT = "safety_enhancement"


class HypothesisPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RSIHypothesis:
    """Comprehensive RSI hypothesis definition"""
    hypothesis_id: str
    hypothesis_type: HypothesisType
    priority: HypothesisPriority
    description: str
    parameters: Dict[str, Any]
    expected_improvement: Dict[str, float]
    safety_constraints: Dict[str, Any]
    computational_cost: float
    risk_level: float
    parent_hypothesis_id: Optional[str] = None
    generation_timestamp: float = None
    validation_results: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.generation_timestamp is None:
            self.generation_timestamp = time.time()


@dataclass 
class HypothesisGenerationConfig:
    """Configuration for hypothesis generation"""
    max_hypotheses_per_iteration: int = 100
    optimization_direction: str = "maximize"
    sampler_type: str = "tpe"  # tpe, cmaes, random
    pruner_enabled: bool = True
    safety_threshold: float = 0.8
    computational_budget: float = 1000.0  # seconds
    parallel_trials: int = 10
    exploration_weight: float = 0.3
    

class RSIHypothesisGenerator:
    """
    Advanced hypothesis generation for RSI systems using Optuna and Ray.
    Provides comprehensive hypothesis exploration with safety guarantees.
    """
    
    def __init__(self, 
                 config: HypothesisGenerationConfig,
                 state_manager: Optional[RSIStateManager] = None,
                 validator: Optional[RSIValidator] = None,
                 circuit_breaker: Optional[RSICircuitBreaker] = None):
        
        self.config = config
        self.state_manager = state_manager
        self.validator = validator
        self.circuit_breaker = circuit_breaker
        
        # Hypothesis tracking
        self.generated_hypotheses: List[RSIHypothesis] = []
        self.hypothesis_history: Dict[str, List[RSIHypothesis]] = {}
        self.performance_tracker = {}
        
        # Optuna study initialization
        self.study = None
        if OPTUNA_AVAILABLE:
            self._initialize_optuna_study()
        
        # Ray initialization
        self.ray_initialized = False
        if RAY_AVAILABLE:
            self._initialize_ray()
            
        logger.info("RSI Hypothesis Generator initialized with {} sampler", 
                   config.sampler_type)
    
    def _initialize_optuna_study(self):
        """Initialize Optuna study with appropriate sampler and pruner"""
        
        # Choose sampler based on configuration
        if self.config.sampler_type == "tpe":
            sampler = TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                multivariate=True,
                group=True
            )
        elif self.config.sampler_type == "cmaes":
            sampler = CmaEsSampler(
                n_startup_trials=10,
                restart_strategy="ipop"
            )
        else:
            sampler = optuna.samplers.RandomSampler()
        
        # Configure pruner if enabled
        pruner = None
        if self.config.pruner_enabled:
            pruner = SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=4,
                min_early_stopping_rate=0
            )
        
        self.study = optuna.create_study(
            direction=self.config.optimization_direction,
            sampler=sampler,
            pruner=pruner,
            study_name=f"rsi_hypothesis_study_{int(time.time())}"
        )
        
        logger.info("Optuna study initialized: {}", self.study.study_name)
    
    def _initialize_ray(self):
        """Initialize Ray for distributed hypothesis generation"""
        try:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            self.ray_initialized = True
            logger.info("Ray initialized for distributed hypothesis generation")
        except Exception as e:
            logger.warning("Failed to initialize Ray: {}", str(e))
            self.ray_initialized = False
    
    async def generate_hypotheses_batch(self, 
                                      improvement_targets: Dict[str, float],
                                      context: Dict[str, Any]) -> List[RSIHypothesis]:
        """
        Generate a batch of RSI improvement hypotheses.
        
        Args:
            improvement_targets: Target improvements (e.g., {"accuracy": 0.05, "efficiency": 0.1})
            context: Current system context and constraints
            
        Returns:
            List of generated hypotheses sorted by potential impact
        """
        batch_id = f"batch_{int(time.time())}"
        logger.info("Generating hypothesis batch {} with targets: {}", 
                   batch_id, improvement_targets)
        
        hypotheses = []
        
        # Generate different types of hypotheses in parallel
        hypothesis_tasks = [
            self._generate_architecture_hypotheses(improvement_targets, context),
            self._generate_hyperparameter_hypotheses(improvement_targets, context),
            self._generate_algorithm_hypotheses(improvement_targets, context),
            self._generate_ensemble_hypotheses(improvement_targets, context),
            self._generate_safety_hypotheses(improvement_targets, context)
        ]
        
        try:
            # Execute hypothesis generation in parallel
            hypothesis_batches = await asyncio.gather(*hypothesis_tasks, return_exceptions=True)
            
            # Combine all hypotheses
            for batch in hypothesis_batches:
                if isinstance(batch, Exception):
                    logger.error("Hypothesis generation error: {}", str(batch))
                    continue
                hypotheses.extend(batch)
            
            # Filter and rank hypotheses
            filtered_hypotheses = await self._filter_and_rank_hypotheses(
                hypotheses, improvement_targets, context
            )
            
            # Store generated hypotheses
            self.generated_hypotheses.extend(filtered_hypotheses)
            self.hypothesis_history[batch_id] = filtered_hypotheses
            
            logger.info("Generated {} hypotheses in batch {}", 
                       len(filtered_hypotheses), batch_id)
            
            return filtered_hypotheses
            
        except Exception as e:
            logger.error("Error in hypothesis batch generation: {}", str(e))
            return []
    
    async def _generate_architecture_hypotheses(self, 
                                              targets: Dict[str, float],
                                              context: Dict[str, Any]) -> List[RSIHypothesis]:
        """Generate neural architecture modification hypotheses"""
        hypotheses = []
        
        # Architecture search space
        architecture_params = {
            'layer_count': (1, 10),
            'hidden_units': (32, 1024),
            'dropout_rate': (0.0, 0.5),
            'activation_function': ['relu', 'tanh', 'gelu', 'swish'],
            'normalization': ['batch_norm', 'layer_norm', 'none'],
            'skip_connections': [True, False],
            'attention_heads': (1, 16)
        }
        
        # Generate hypotheses using Optuna if available
        if OPTUNA_AVAILABLE and self.study:
            for i in range(min(20, self.config.max_hypotheses_per_iteration // 5)):
                trial = self.study.ask()
                
                params = {}
                for param_name, param_range in architecture_params.items():
                    if isinstance(param_range, tuple):
                        if isinstance(param_range[0], int):
                            params[param_name] = trial.suggest_int(param_name, *param_range)
                        else:
                            params[param_name] = trial.suggest_float(param_name, *param_range)
                    elif isinstance(param_range, list):
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                
                hypothesis = RSIHypothesis(
                    hypothesis_id=f"arch_{trial.number}_{int(time.time())}",
                    hypothesis_type=HypothesisType.ARCHITECTURE_CHANGE,
                    priority=HypothesisPriority.HIGH,
                    description=f"Architecture modification: {params}",
                    parameters=params,
                    expected_improvement=await self._estimate_architecture_improvement(params, targets),
                    safety_constraints=await self._get_architecture_safety_constraints(params),
                    computational_cost=await self._estimate_computational_cost(params),
                    risk_level=await self._assess_risk_level(params, "architecture")
                )
                
                hypotheses.append(hypothesis)
        else:
            # Fallback: generate random architecture hypotheses
            for i in range(5):
                params = {
                    'layer_count': np.random.randint(2, 8),
                    'hidden_units': np.random.choice([64, 128, 256, 512]),
                    'dropout_rate': np.random.uniform(0.1, 0.4),
                    'activation_function': np.random.choice(['relu', 'gelu']),
                    'normalization': np.random.choice(['batch_norm', 'layer_norm']),
                    'skip_connections': np.random.choice([True, False])
                }
                
                hypothesis = RSIHypothesis(
                    hypothesis_id=f"arch_fallback_{i}_{int(time.time())}",
                    hypothesis_type=HypothesisType.ARCHITECTURE_CHANGE,
                    priority=HypothesisPriority.MEDIUM,
                    description=f"Fallback architecture: {params}",
                    parameters=params,
                    expected_improvement={"performance": 0.05, "efficiency": 0.02},
                    safety_constraints={"max_memory": 1000, "max_compute": 100},
                    computational_cost=50.0,
                    risk_level=0.3
                )
                
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_hyperparameter_hypotheses(self, 
                                                targets: Dict[str, float],
                                                context: Dict[str, Any]) -> List[RSIHypothesis]:
        """Generate hyperparameter optimization hypotheses"""
        hypotheses = []
        
        # Hyperparameter search space
        hyperparam_space = {
            'learning_rate': (1e-5, 1e-1),
            'batch_size': [16, 32, 64, 128, 256],
            'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],
            'weight_decay': (1e-6, 1e-2),
            'momentum': (0.8, 0.99),
            'scheduler': ['cosine', 'linear', 'exponential', 'step'],
            'warmup_steps': (0, 1000)
        }
        
        if OPTUNA_AVAILABLE and self.study:
            for i in range(min(30, self.config.max_hypotheses_per_iteration // 3)):
                trial = self.study.ask()
                
                params = {}
                for param_name, param_range in hyperparam_space.items():
                    if isinstance(param_range, tuple):
                        if param_name in ['learning_rate', 'weight_decay']:
                            params[param_name] = trial.suggest_float(param_name, *param_range, log=True)
                        elif isinstance(param_range[0], int):
                            params[param_name] = trial.suggest_int(param_name, *param_range)
                        else:
                            params[param_name] = trial.suggest_float(param_name, *param_range)
                    elif isinstance(param_range, list):
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                
                hypothesis = RSIHypothesis(
                    hypothesis_id=f"hyper_{trial.number}_{int(time.time())}",
                    hypothesis_type=HypothesisType.HYPERPARAMETER_OPTIMIZATION,
                    priority=HypothesisPriority.MEDIUM,
                    description=f"Hyperparameter optimization: {params}",
                    parameters=params,
                    expected_improvement=await self._estimate_hyperparameter_improvement(params, targets),
                    safety_constraints=await self._get_hyperparameter_safety_constraints(params),
                    computational_cost=await self._estimate_computational_cost(params),
                    risk_level=await self._assess_risk_level(params, "hyperparameter")
                )
                
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_algorithm_hypotheses(self, 
                                           targets: Dict[str, float],
                                           context: Dict[str, Any]) -> List[RSIHypothesis]:
        """Generate algorithm modification hypotheses"""
        hypotheses = []
        
        algorithm_modifications = [
            {
                'type': 'loss_function',
                'modifications': ['focal_loss', 'label_smoothing', 'mixup', 'cutmix'],
                'parameters': {'alpha': (0.1, 2.0), 'smoothing': (0.01, 0.3)}
            },
            {
                'type': 'regularization',
                'modifications': ['dropout_schedule', 'spectral_norm', 'gradient_clipping'],
                'parameters': {'clip_value': (0.1, 5.0), 'schedule_type': ['linear', 'cosine']}
            },
            {
                'type': 'training_strategy',
                'modifications': ['progressive_resizing', 'curriculum_learning', 'self_training'],
                'parameters': {'progression_steps': (2, 10), 'difficulty_schedule': ['easy_first', 'hard_first']}
            }
        ]
        
        for mod_category in algorithm_modifications:
            for modification in mod_category['modifications'][:2]:  # Limit to 2 per category
                params = {
                    'modification_type': modification,
                    'category': mod_category['type']
                }
                
                # Add specific parameters for this modification
                for param_name, param_range in mod_category['parameters'].items():
                    if isinstance(param_range, tuple):
                        params[param_name] = np.random.uniform(*param_range)
                    elif isinstance(param_range, list):
                        params[param_name] = np.random.choice(param_range)
                
                hypothesis = RSIHypothesis(
                    hypothesis_id=f"algo_{modification}_{int(time.time())}",
                    hypothesis_type=HypothesisType.ALGORITHM_MODIFICATION,
                    priority=HypothesisPriority.HIGH,
                    description=f"Algorithm modification: {modification} in {mod_category['type']}",
                    parameters=params,
                    expected_improvement=await self._estimate_algorithm_improvement(params, targets),
                    safety_constraints=await self._get_algorithm_safety_constraints(params),
                    computational_cost=await self._estimate_computational_cost(params),
                    risk_level=await self._assess_risk_level(params, "algorithm")
                )
                
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_ensemble_hypotheses(self, 
                                          targets: Dict[str, float],
                                          context: Dict[str, Any]) -> List[RSIHypothesis]:
        """Generate ensemble strategy hypotheses"""
        hypotheses = []
        
        ensemble_strategies = [
            {
                'strategy': 'weighted_voting',
                'parameters': {'weight_method': ['performance', 'diversity', 'uncertainty'], 'num_models': (3, 10)}
            },
            {
                'strategy': 'stacking',
                'parameters': {'meta_learner': ['linear', 'rf', 'xgboost'], 'cv_folds': (3, 10)}
            },
            {
                'strategy': 'boosting',
                'parameters': {'boost_type': ['ada', 'gradient', 'xgboost'], 'learning_rate': (0.01, 0.3)}
            },
            {
                'strategy': 'bagging',
                'parameters': {'sample_ratio': (0.6, 0.9), 'feature_ratio': (0.7, 1.0)}
            }
        ]
        
        for strategy_config in ensemble_strategies:
            params = {'ensemble_strategy': strategy_config['strategy']}
            
            for param_name, param_range in strategy_config['parameters'].items():
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        params[param_name] = np.random.randint(*param_range)
                    else:
                        params[param_name] = np.random.uniform(*param_range)
                elif isinstance(param_range, list):
                    params[param_name] = np.random.choice(param_range)
            
            hypothesis = RSIHypothesis(
                hypothesis_id=f"ensemble_{strategy_config['strategy']}_{int(time.time())}",
                hypothesis_type=HypothesisType.ENSEMBLE_STRATEGY,
                priority=HypothesisPriority.MEDIUM,
                description=f"Ensemble strategy: {strategy_config['strategy']}",
                parameters=params,
                expected_improvement=await self._estimate_ensemble_improvement(params, targets),
                safety_constraints=await self._get_ensemble_safety_constraints(params),
                computational_cost=await self._estimate_computational_cost(params),
                risk_level=await self._assess_risk_level(params, "ensemble")
            )
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_safety_hypotheses(self, 
                                        targets: Dict[str, float],
                                        context: Dict[str, Any]) -> List[RSIHypothesis]:
        """Generate safety enhancement hypotheses"""
        hypotheses = []
        
        safety_enhancements = [
            {
                'enhancement': 'uncertainty_quantification',
                'parameters': {'method': ['mc_dropout', 'ensemble', 'bayesian'], 'samples': (10, 100)}
            },
            {
                'enhancement': 'adversarial_training',
                'parameters': {'attack_type': ['fgsm', 'pgd', 'cw'], 'epsilon': (0.01, 0.1)}
            },
            {
                'enhancement': 'input_validation',
                'parameters': {'validation_level': ['basic', 'comprehensive'], 'rejection_threshold': (0.1, 0.5)}
            },
            {
                'enhancement': 'output_monitoring',
                'parameters': {'monitoring_type': ['statistical', 'ml_based'], 'alert_threshold': (0.05, 0.2)}
            }
        ]
        
        for safety_config in safety_enhancements:
            params = {'safety_enhancement': safety_config['enhancement']}
            
            for param_name, param_range in safety_config['parameters'].items():
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        params[param_name] = np.random.randint(*param_range)
                    else:
                        params[param_name] = np.random.uniform(*param_range)
                elif isinstance(param_range, list):
                    params[param_name] = np.random.choice(param_range)
            
            hypothesis = RSIHypothesis(
                hypothesis_id=f"safety_{safety_config['enhancement']}_{int(time.time())}",
                hypothesis_type=HypothesisType.SAFETY_ENHANCEMENT,
                priority=HypothesisPriority.CRITICAL,
                description=f"Safety enhancement: {safety_config['enhancement']}",
                parameters=params,
                expected_improvement=await self._estimate_safety_improvement(params, targets),
                safety_constraints=await self._get_safety_enhancement_constraints(params),
                computational_cost=await self._estimate_computational_cost(params),
                risk_level=await self._assess_risk_level(params, "safety")
            )
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _filter_and_rank_hypotheses(self, 
                                        hypotheses: List[RSIHypothesis],
                                        targets: Dict[str, float],
                                        context: Dict[str, Any]) -> List[RSIHypothesis]:
        """Filter and rank hypotheses by potential impact and safety"""
        
        # Filter by safety constraints
        safe_hypotheses = []
        for hypothesis in hypotheses:
            if hypothesis.risk_level <= 0.8 and await self._validate_safety_constraints(hypothesis):
                safe_hypotheses.append(hypothesis)
        
        # Rank by expected improvement and priority
        def ranking_score(h: RSIHypothesis) -> float:
            priority_weights = {
                HypothesisPriority.CRITICAL: 1.0,
                HypothesisPriority.HIGH: 0.8,
                HypothesisPriority.MEDIUM: 0.6,
                HypothesisPriority.LOW: 0.4
            }
            
            improvement_score = sum(h.expected_improvement.values())
            priority_score = priority_weights[h.priority]
            safety_score = 1.0 - h.risk_level
            efficiency_score = 1.0 / (1.0 + h.computational_cost / 100.0)
            
            return improvement_score * priority_score * safety_score * efficiency_score
        
        ranked_hypotheses = sorted(safe_hypotheses, key=ranking_score, reverse=True)
        
        # Limit to max hypotheses per iteration
        return ranked_hypotheses[:self.config.max_hypotheses_per_iteration]
    
    # Helper methods for estimation and validation
    
    async def _estimate_architecture_improvement(self, params: Dict[str, Any], targets: Dict[str, float]) -> Dict[str, float]:
        """Estimate expected improvement from architecture changes"""
        # Simplified estimation - in practice, this would use more sophisticated modeling
        base_improvement = 0.02
        if params.get('skip_connections'):
            base_improvement += 0.01
        if params.get('attention_heads', 0) > 1:
            base_improvement += 0.015
        
        return {target: base_improvement * weight for target, weight in targets.items()}
    
    async def _estimate_hyperparameter_improvement(self, params: Dict[str, Any], targets: Dict[str, float]) -> Dict[str, float]:
        """Estimate expected improvement from hyperparameter changes"""
        base_improvement = 0.01
        if params.get('optimizer') in ['adamw', 'adam']:
            base_improvement += 0.005
        
        return {target: base_improvement * weight for target, weight in targets.items()}
    
    async def _estimate_algorithm_improvement(self, params: Dict[str, Any], targets: Dict[str, float]) -> Dict[str, float]:
        """Estimate expected improvement from algorithm modifications"""
        base_improvement = 0.03
        if params.get('modification_type') in ['focal_loss', 'mixup']:
            base_improvement += 0.01
        
        return {target: base_improvement * weight for target, weight in targets.items()}
    
    async def _estimate_ensemble_improvement(self, params: Dict[str, Any], targets: Dict[str, float]) -> Dict[str, float]:
        """Estimate expected improvement from ensemble strategies"""
        base_improvement = 0.025
        if params.get('ensemble_strategy') in ['stacking', 'boosting']:
            base_improvement += 0.01
        
        return {target: base_improvement * weight for target, weight in targets.items()}
    
    async def _estimate_safety_improvement(self, params: Dict[str, Any], targets: Dict[str, float]) -> Dict[str, float]:
        """Estimate expected improvement from safety enhancements"""
        # Safety improvements often trade performance for robustness
        base_improvement = -0.005  # Small performance cost
        safety_improvement = 0.1   # Large safety benefit
        
        improvements = {target: base_improvement * weight for target, weight in targets.items()}
        improvements['safety_score'] = safety_improvement
        return improvements
    
    async def _get_architecture_safety_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get safety constraints for architecture changes"""
        return {
            'max_parameters': 10_000_000,
            'max_memory_mb': 2000,
            'max_inference_time_ms': 1000,
            'requires_validation': True
        }
    
    async def _get_hyperparameter_safety_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get safety constraints for hyperparameter changes"""
        return {
            'max_learning_rate': 0.1,
            'min_batch_size': 8,
            'max_training_time_hours': 24,
            'requires_validation': False
        }
    
    async def _get_algorithm_safety_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get safety constraints for algorithm modifications"""
        return {
            'requires_extensive_testing': True,
            'max_modification_complexity': 5,
            'requires_human_review': True,
            'rollback_strategy_required': True
        }
    
    async def _get_ensemble_safety_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get safety constraints for ensemble strategies"""
        return {
            'max_ensemble_size': 10,
            'min_diversity_threshold': 0.3,
            'requires_validation': True,
            'computational_budget_multiplier': params.get('num_models', 3)
        }
    
    async def _get_safety_enhancement_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get safety constraints for safety enhancements"""
        return {
            'always_approved': True,  # Safety enhancements are always good
            'requires_integration_testing': True,
            'performance_impact_acceptable': True,
            'priority_boost': True
        }
    
    async def _estimate_computational_cost(self, params: Dict[str, Any]) -> float:
        """Estimate computational cost of implementing hypothesis"""
        base_cost = 10.0  # Base cost in arbitrary units
        
        # Add cost based on complexity
        if 'layer_count' in params:
            base_cost += params['layer_count'] * 5
        if 'num_models' in params:
            base_cost *= params['num_models']
        if 'samples' in params:
            base_cost += params['samples'] * 0.1
        
        return base_cost
    
    async def _assess_risk_level(self, params: Dict[str, Any], hypothesis_type: str) -> float:
        """Assess risk level of hypothesis (0.0 = no risk, 1.0 = maximum risk)"""
        base_risk = {
            'architecture': 0.4,
            'hyperparameter': 0.2,
            'algorithm': 0.5,
            'ensemble': 0.3,
            'safety': 0.1
        }.get(hypothesis_type, 0.3)
        
        # Adjust risk based on parameters
        if 'modification_type' in params and params['modification_type'] in ['focal_loss', 'mixup']:
            base_risk += 0.1
        
        return min(1.0, base_risk)
    
    async def _validate_safety_constraints(self, hypothesis: RSIHypothesis) -> bool:
        """Validate that hypothesis meets safety constraints"""
        if self.validator:
            try:
                validation_result = await self.validator.validate_hypothesis(hypothesis)
                return validation_result.is_valid
            except Exception as e:
                logger.error("Hypothesis validation error: {}", str(e))
                return False
        
        # Basic safety checks
        return (hypothesis.risk_level <= self.config.safety_threshold and
                hypothesis.computational_cost <= self.config.computational_budget)
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about hypothesis generation"""
        
        total_hypotheses = len(self.generated_hypotheses)
        if total_hypotheses == 0:
            return {'status': 'no_hypotheses_generated'}
        
        type_distribution = {}
        priority_distribution = {}
        risk_distribution = {'low': 0, 'medium': 0, 'high': 0}
        
        for hypothesis in self.generated_hypotheses:
            # Type distribution
            type_key = hypothesis.hypothesis_type.value
            type_distribution[type_key] = type_distribution.get(type_key, 0) + 1
            
            # Priority distribution
            priority_key = hypothesis.priority.value
            priority_distribution[priority_key] = priority_distribution.get(priority_key, 0) + 1
            
            # Risk distribution
            if hypothesis.risk_level < 0.3:
                risk_distribution['low'] += 1
            elif hypothesis.risk_level < 0.7:
                risk_distribution['medium'] += 1
            else:
                risk_distribution['high'] += 1
        
        avg_expected_improvement = np.mean([
            sum(h.expected_improvement.values()) for h in self.generated_hypotheses
        ])
        
        avg_computational_cost = np.mean([h.computational_cost for h in self.generated_hypotheses])
        avg_risk_level = np.mean([h.risk_level for h in self.generated_hypotheses])
        
        return {
            'total_hypotheses': total_hypotheses,
            'type_distribution': type_distribution,
            'priority_distribution': priority_distribution,
            'risk_distribution': risk_distribution,
            'avg_expected_improvement': avg_expected_improvement,
            'avg_computational_cost': avg_computational_cost,
            'avg_risk_level': avg_risk_level,
            'generation_batches': len(self.hypothesis_history),
            'optuna_available': OPTUNA_AVAILABLE,
            'ray_available': RAY_AVAILABLE and self.ray_initialized
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if RAY_AVAILABLE and self.ray_initialized:
            try:
                ray.shutdown()
                logger.info("Ray shutdown completed")
            except Exception as e:
                logger.warning("Error during Ray shutdown: {}", str(e))