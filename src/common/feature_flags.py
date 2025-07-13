"""
Feature flags system for Hephaestus RSI.

Provides dynamic feature toggle capabilities with support for gradual
rollouts, A/B testing, and canary deployments.
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime, timedelta

from .exceptions import FeatureFlagError, create_error_context

logger = logging.getLogger(__name__)


class FeatureStrategy(Enum):
    """Feature rollout strategies."""
    BOOLEAN = "boolean"           # Simple on/off
    PERCENTAGE = "percentage"     # Percentage-based rollout
    USER_LIST = "user_list"       # Specific user whitelist
    TIME_WINDOW = "time_window"   # Time-based activation
    CANARY = "canary"            # Canary deployment
    A_B_TEST = "a_b_test"        # A/B testing


@dataclass
class FeatureConfig:
    """Configuration for a feature flag."""
    
    name: str
    strategy: FeatureStrategy
    enabled: bool = False
    description: str = ""
    
    # Strategy-specific configuration
    percentage: float = 0.0           # For PERCENTAGE strategy
    user_list: List[str] = field(default_factory=list)  # For USER_LIST
    start_time: Optional[datetime] = None  # For TIME_WINDOW
    end_time: Optional[datetime] = None    # For TIME_WINDOW
    
    # A/B testing configuration
    variants: Dict[str, Any] = field(default_factory=dict)
    variant_weights: Dict[str, float] = field(default_factory=dict)
    
    # Canary configuration
    canary_percentage: float = 0.0
    success_threshold: float = 0.95
    failure_threshold: float = 0.05
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)


@dataclass
class FeatureEvaluation:
    """Result of feature flag evaluation."""
    
    feature_name: str
    enabled: bool
    variant: Optional[str] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=datetime.now)


class FeatureFlagManager:
    """Manages feature flags with dynamic updates and analytics."""
    
    def __init__(
        self,
        config_file: Optional[Path] = None,
        refresh_interval: float = 60.0,
        enable_analytics: bool = True
    ):
        self.config_file = config_file or Path("config/feature_flags.json")
        self.refresh_interval = refresh_interval
        self.enable_analytics = enable_analytics
        
        # Feature configurations
        self.features: Dict[str, FeatureConfig] = {}
        self.lock = threading.RLock()
        
        # Analytics data
        self.evaluations: List[FeatureEvaluation] = []
        self.evaluation_lock = threading.Lock()
        
        # Background refresh task
        self._refresh_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Ensure config directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load initial configuration
        self._load_configuration()
    
    async def start(self) -> None:
        """Start the feature flag manager."""
        if self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info("Feature flag manager started")
    
    async def stop(self) -> None:
        """Stop the feature flag manager."""
        self._shutdown_event.set()
        
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
        
        logger.info("Feature flag manager stopped")
    
    def _load_configuration(self) -> None:
        """Load feature flag configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                with self.lock:
                    self.features.clear()
                    for name, config_data in data.get('features', {}).items():
                        config = self._deserialize_config(name, config_data)
                        self.features[name] = config
                
                logger.info(f"Loaded {len(self.features)} feature flags")
            else:
                # Create default configuration
                self._create_default_configuration()
        
        except Exception as e:
            logger.error(f"Failed to load feature flags: {e}")
            raise FeatureFlagError(
                f"Configuration load failed: {e}",
                context=create_error_context("feature_flag_load")
            )
    
    def _save_configuration(self) -> None:
        """Save feature flag configuration to file."""
        try:
            data = {
                'features': {
                    name: self._serialize_config(config)
                    for name, config in self.features.items()
                },
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug("Feature flag configuration saved")
        
        except Exception as e:
            logger.error(f"Failed to save feature flags: {e}")
    
    def _create_default_configuration(self) -> None:
        """Create default feature flag configuration."""
        default_features = {
            'enhanced_learning': FeatureConfig(
                name='enhanced_learning',
                strategy=FeatureStrategy.PERCENTAGE,
                enabled=True,
                percentage=50.0,
                description='Enhanced meta-learning capabilities'
            ),
            'threat_detection': FeatureConfig(
                name='threat_detection',
                strategy=FeatureStrategy.BOOLEAN,
                enabled=True,
                description='Advanced threat detection system'
            ),
            'canary_deployment': FeatureConfig(
                name='canary_deployment',
                strategy=FeatureStrategy.CANARY,
                enabled=False,
                canary_percentage=5.0,
                description='Canary deployment system'
            ),
            'performance_optimization': FeatureConfig(
                name='performance_optimization',
                strategy=FeatureStrategy.A_B_TEST,
                enabled=True,
                variants={'control': 'baseline', 'treatment': 'optimized'},
                variant_weights={'control': 0.5, 'treatment': 0.5},
                description='Performance optimization A/B test'
            )
        }
        
        with self.lock:
            self.features.update(default_features)
        
        self._save_configuration()
        logger.info("Created default feature flag configuration")
    
    def _serialize_config(self, config: FeatureConfig) -> Dict[str, Any]:
        """Serialize feature configuration to dict."""
        return {
            'strategy': config.strategy.value,
            'enabled': config.enabled,
            'description': config.description,
            'percentage': config.percentage,
            'user_list': config.user_list,
            'start_time': config.start_time.isoformat() if config.start_time else None,
            'end_time': config.end_time.isoformat() if config.end_time else None,
            'variants': config.variants,
            'variant_weights': config.variant_weights,
            'canary_percentage': config.canary_percentage,
            'success_threshold': config.success_threshold,
            'failure_threshold': config.failure_threshold,
            'created_at': config.created_at.isoformat(),
            'updated_at': config.updated_at.isoformat(),
            'created_by': config.created_by,
            'tags': config.tags
        }
    
    def _deserialize_config(self, name: str, data: Dict[str, Any]) -> FeatureConfig:
        """Deserialize feature configuration from dict."""
        config = FeatureConfig(
            name=name,
            strategy=FeatureStrategy(data.get('strategy', 'boolean')),
            enabled=data.get('enabled', False),
            description=data.get('description', ''),
            percentage=data.get('percentage', 0.0),
            user_list=data.get('user_list', []),
            variants=data.get('variants', {}),
            variant_weights=data.get('variant_weights', {}),
            canary_percentage=data.get('canary_percentage', 0.0),
            success_threshold=data.get('success_threshold', 0.95),
            failure_threshold=data.get('failure_threshold', 0.05),
            created_by=data.get('created_by', 'system'),
            tags=data.get('tags', [])
        )
        
        # Parse datetime fields
        if data.get('start_time'):
            config.start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            config.end_time = datetime.fromisoformat(data['end_time'])
        if data.get('created_at'):
            config.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at'):
            config.updated_at = datetime.fromisoformat(data['updated_at'])
        
        return config
    
    async def _refresh_loop(self) -> None:
        """Background task to refresh configuration."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.refresh_interval)
                if not self._shutdown_event.is_set():
                    self._load_configuration()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Feature flag refresh failed: {e}")
    
    def is_enabled(
        self,
        feature_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if a feature is enabled for the given context."""
        evaluation = self.evaluate(feature_name, user_id, context)
        return evaluation.enabled
    
    def evaluate(
        self,
        feature_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> FeatureEvaluation:
        """Evaluate a feature flag and return detailed result."""
        context = context or {}
        
        with self.lock:
            config = self.features.get(feature_name)
        
        if not config:
            evaluation = FeatureEvaluation(
                feature_name=feature_name,
                enabled=False,
                reason="Feature not found"
            )
        else:
            evaluation = self._evaluate_feature(config, user_id, context)
        
        # Record evaluation for analytics
        if self.enable_analytics:
            with self.evaluation_lock:
                self.evaluations.append(evaluation)
                # Keep only recent evaluations
                cutoff = datetime.now() - timedelta(hours=24)
                self.evaluations = [
                    e for e in self.evaluations 
                    if e.evaluated_at > cutoff
                ]
        
        return evaluation
    
    def _evaluate_feature(
        self,
        config: FeatureConfig,
        user_id: Optional[str],
        context: Dict[str, Any]
    ) -> FeatureEvaluation:
        """Evaluate a single feature configuration."""
        if not config.enabled:
            return FeatureEvaluation(
                feature_name=config.name,
                enabled=False,
                reason="Feature disabled"
            )
        
        if config.strategy == FeatureStrategy.BOOLEAN:
            return FeatureEvaluation(
                feature_name=config.name,
                enabled=True,
                reason="Boolean strategy - enabled"
            )
        
        elif config.strategy == FeatureStrategy.PERCENTAGE:
            # Use deterministic hash for consistent results
            hash_input = f"{config.name}:{user_id or 'anonymous'}"
            hash_value = hash(hash_input) % 100
            enabled = hash_value < config.percentage
            
            return FeatureEvaluation(
                feature_name=config.name,
                enabled=enabled,
                reason=f"Percentage strategy - {hash_value}% < {config.percentage}%"
            )
        
        elif config.strategy == FeatureStrategy.USER_LIST:
            enabled = user_id in config.user_list if user_id else False
            return FeatureEvaluation(
                feature_name=config.name,
                enabled=enabled,
                reason=f"User list strategy - user {'in' if enabled else 'not in'} list"
            )
        
        elif config.strategy == FeatureStrategy.TIME_WINDOW:
            now = datetime.now()
            in_window = True
            
            if config.start_time and now < config.start_time:
                in_window = False
            if config.end_time and now > config.end_time:
                in_window = False
            
            return FeatureEvaluation(
                feature_name=config.name,
                enabled=in_window,
                reason=f"Time window strategy - {'in' if in_window else 'outside'} window"
            )
        
        elif config.strategy == FeatureStrategy.CANARY:
            # Canary deployment logic
            hash_input = f"{config.name}:{user_id or 'anonymous'}"
            hash_value = hash(hash_input) % 100
            enabled = hash_value < config.canary_percentage
            
            return FeatureEvaluation(
                feature_name=config.name,
                enabled=enabled,
                variant="canary" if enabled else "control",
                reason=f"Canary strategy - {config.canary_percentage}% rollout"
            )
        
        elif config.strategy == FeatureStrategy.A_B_TEST:
            # A/B testing with weighted variants
            if not config.variants or not config.variant_weights:
                return FeatureEvaluation(
                    feature_name=config.name,
                    enabled=False,
                    reason="A/B test strategy - no variants configured"
                )
            
            hash_input = f"{config.name}:{user_id or 'anonymous'}"
            hash_value = hash(hash_input) % 100
            
            cumulative_weight = 0
            for variant, weight in config.variant_weights.items():
                cumulative_weight += weight * 100
                if hash_value < cumulative_weight:
                    return FeatureEvaluation(
                        feature_name=config.name,
                        enabled=True,
                        variant=variant,
                        reason=f"A/B test strategy - variant {variant}",
                        metadata={"variant_value": config.variants.get(variant)}
                    )
            
            # Fallback to control
            return FeatureEvaluation(
                feature_name=config.name,
                enabled=False,
                reason="A/B test strategy - fallback to control"
            )
        
        else:
            return FeatureEvaluation(
                feature_name=config.name,
                enabled=False,
                reason=f"Unknown strategy: {config.strategy}"
            )
    
    def create_feature(self, config: FeatureConfig) -> None:
        """Create a new feature flag."""
        with self.lock:
            self.features[config.name] = config
        
        self._save_configuration()
        logger.info(f"Created feature flag: {config.name}")
    
    def update_feature(self, name: str, updates: Dict[str, Any]) -> None:
        """Update an existing feature flag."""
        with self.lock:
            if name not in self.features:
                raise FeatureFlagError(
                    f"Feature not found: {name}",
                    context=create_error_context("feature_update")
                )
            
            config = self.features[name]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            config.updated_at = datetime.now()
        
        self._save_configuration()
        logger.info(f"Updated feature flag: {name}")
    
    def delete_feature(self, name: str) -> None:
        """Delete a feature flag."""
        with self.lock:
            if name not in self.features:
                raise FeatureFlagError(
                    f"Feature not found: {name}",
                    context=create_error_context("feature_delete")
                )
            
            del self.features[name]
        
        self._save_configuration()
        logger.info(f"Deleted feature flag: {name}")
    
    def list_features(self) -> List[FeatureConfig]:
        """List all feature flags."""
        with self.lock:
            return list(self.features.values())
    
    def get_feature(self, name: str) -> Optional[FeatureConfig]:
        """Get a specific feature configuration."""
        with self.lock:
            return self.features.get(name)
    
    def get_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get feature flag analytics."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self.evaluation_lock:
            recent_evaluations = [
                e for e in self.evaluations
                if e.evaluated_at > cutoff
            ]
        
        # Calculate statistics
        total_evaluations = len(recent_evaluations)
        feature_stats = {}
        
        for evaluation in recent_evaluations:
            name = evaluation.feature_name
            if name not in feature_stats:
                feature_stats[name] = {
                    'total': 0,
                    'enabled': 0,
                    'variants': {}
                }
            
            feature_stats[name]['total'] += 1
            if evaluation.enabled:
                feature_stats[name]['enabled'] += 1
            
            if evaluation.variant:
                variant = evaluation.variant
                if variant not in feature_stats[name]['variants']:
                    feature_stats[name]['variants'][variant] = 0
                feature_stats[name]['variants'][variant] += 1
        
        # Calculate percentages
        for name, stats in feature_stats.items():
            if stats['total'] > 0:
                stats['enabled_percentage'] = (stats['enabled'] / stats['total']) * 100
        
        return {
            'time_period_hours': hours,
            'total_evaluations': total_evaluations,
            'feature_statistics': feature_stats,
            'last_updated': datetime.now().isoformat()
        }


# Global feature flag manager instance
_feature_manager: Optional[FeatureFlagManager] = None


def get_feature_manager() -> FeatureFlagManager:
    """Get global feature flag manager instance."""
    global _feature_manager
    if _feature_manager is None:
        _feature_manager = FeatureFlagManager()
    return _feature_manager


def set_feature_manager(manager: FeatureFlagManager) -> None:
    """Set global feature flag manager instance."""
    global _feature_manager
    _feature_manager = manager


# Convenience functions
def is_enabled(feature_name: str, user_id: Optional[str] = None, **context) -> bool:
    """Check if a feature is enabled."""
    return get_feature_manager().is_enabled(feature_name, user_id, context)


def evaluate(feature_name: str, user_id: Optional[str] = None, **context) -> FeatureEvaluation:
    """Evaluate a feature flag."""
    return get_feature_manager().evaluate(feature_name, user_id, context)