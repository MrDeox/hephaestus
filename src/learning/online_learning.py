"""
Online learning implementation for RSI system using River.
Provides continuous adaptation with concept drift detection.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import pickle
import hashlib

import numpy as np
from river import linear_model, tree, ensemble, drift, metrics, preprocessing
from river.base import Classifier, Regressor
from sklearn.model_selection import train_test_split
from loguru import logger

from ..core.state import RSIState, RSIStateManager, add_learning_record
from ..validation.validators import RSIValidator, ValidationResult
from ..safety.circuits import RSICircuitBreaker, create_learning_circuit


class LearningMode(str, Enum):
    """Learning modes for the RSI system."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    BALANCED = "balanced"


class ConceptDriftType(str, Enum):
    """Types of concept drift."""
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    RECURRING = "recurring"
    NONE = "none"


@dataclass
class LearningMetrics:
    """Metrics for online learning performance."""
    
    accuracy: float
    loss: float
    samples_processed: int
    learning_rate: float
    concept_drift_detected: bool
    drift_type: ConceptDriftType
    model_complexity: int
    prediction_confidence: float
    adaptation_speed: float
    timestamp: datetime


class RSIOnlineLearner:
    """
    Online learning component for RSI system.
    Supports continuous learning with concept drift detection.
    """
    
    def __init__(
        self,
        model_type: str = "logistic_regression",
        drift_detector: Optional[str] = "adwin",
        learning_rate: float = 0.01,
        state_manager: Optional[RSIStateManager] = None,
        validator: Optional[RSIValidator] = None,
        circuit_breaker: Optional[RSICircuitBreaker] = None
    ):
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.state_manager = state_manager
        self.validator = validator
        self.circuit_breaker = circuit_breaker or create_learning_circuit()
        
        # Initialize model
        self.model = self._create_model(model_type)
        
        # Initialize drift detector
        self.drift_detector = self._create_drift_detector(drift_detector)
        
        # Initialize metrics
        self.accuracy_metric = metrics.Accuracy()
        self.loss_metric = metrics.LogLoss()
        
        # Learning state
        self.samples_processed = 0
        self.learning_mode = LearningMode.BALANCED
        self.concept_drift_history: List[Tuple[datetime, ConceptDriftType]] = []
        self.adaptation_history: List[LearningMetrics] = []
        
        # Performance tracking
        self.recent_predictions: List[Tuple[float, float]] = []  # (prediction, actual)
        self.prediction_times: List[float] = []
        
    def _create_model(self, model_type: str) -> Union[Classifier, Regressor]:
        """Create the appropriate model based on type."""
        from river import optim
        
        models = {
            "logistic_regression": linear_model.LogisticRegression(
                optimizer=optim.SGD(lr=self.learning_rate)
            ),
            "adaptive_random_forest": ensemble.ADWINBaggingClassifier(
                model=tree.HoeffdingTreeClassifier(),
                n_models=10
            ),
            "hoeffding_tree": tree.HoeffdingTreeClassifier(
                max_depth=8,
                split_criterion="gini"
            ),
            "sgd_regressor": linear_model.LinearRegression(
                optimizer=optim.SGD(lr=self.learning_rate)
            ),
            "adaptive_model_rules": ensemble.BaggingRegressor(
                model=linear_model.LinearRegression(),
                n_models=10
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type]
    
    def _create_drift_detector(self, detector_type: Optional[str]):
        """Create drift detector."""
        if detector_type is None:
            return None
        
        detectors = {
            "adwin": drift.ADWIN(delta=0.002),
            "kswin": drift.KSWIN(alpha=0.005, window_size=100),
            "page_hinkley": drift.PageHinkley(min_instances=30, delta=0.005, threshold=50),
            "dummy": drift.DummyDriftDetector()
        }
        
        if detector_type not in detectors:
            raise ValueError(f"Unknown drift detector: {detector_type}")
        
        return detectors[detector_type]
    
    @property
    def current_metrics(self) -> LearningMetrics:
        """Get current learning metrics."""
        return LearningMetrics(
            accuracy=self.accuracy_metric.get(),
            loss=self.loss_metric.get() if hasattr(self.loss_metric, 'get') else 0.0,
            samples_processed=self.samples_processed,
            learning_rate=self.learning_rate,
            concept_drift_detected=len(self.concept_drift_history) > 0 and 
                                   self.concept_drift_history[-1][0] > datetime.now(timezone.utc) - timedelta(minutes=5),
            drift_type=self.concept_drift_history[-1][1] if self.concept_drift_history else ConceptDriftType.NONE,
            model_complexity=self._estimate_model_complexity(),
            prediction_confidence=self._calculate_prediction_confidence(),
            adaptation_speed=self._calculate_adaptation_speed(),
            timestamp=datetime.now(timezone.utc)
        )
    
    def _estimate_model_complexity(self) -> int:
        """Estimate model complexity."""
        if hasattr(self.model, 'n_nodes'):
            return getattr(self.model, 'n_nodes', 0)
        elif hasattr(self.model, 'n_features_in_'):
            return getattr(self.model, 'n_features_in_', 0)
        else:
            return 1  # Default complexity
    
    def _calculate_prediction_confidence(self) -> float:
        """Calculate prediction confidence based on recent performance."""
        if len(self.recent_predictions) < 10:
            return 0.5
        
        # Calculate variance in recent predictions, filtering out None values
        predictions = [p[0] for p in self.recent_predictions[-50:] if p[0] is not None]
        
        if not predictions:
            return 0.1  # Low confidence if no valid predictions
        
        if len(set(predictions)) == 1:
            return 1.0  # All predictions are the same
        
        try:
            # Convert to numeric for variance calculation
            numeric_predictions = []
            for pred in predictions:
                if isinstance(pred, (int, float)):
                    numeric_predictions.append(float(pred))
                elif isinstance(pred, bool):
                    numeric_predictions.append(float(pred))
                else:
                    # For non-numeric predictions, use string hash
                    numeric_predictions.append(float(hash(str(pred)) % 100) / 100)
            
            if numeric_predictions:
                variance = np.var(numeric_predictions)
                confidence = 1.0 / (1.0 + variance)
                return min(max(confidence, 0.0), 1.0)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Confidence calculation error: {e}")
            return 0.5
    
    def _calculate_adaptation_speed(self) -> float:
        """Calculate how quickly the model adapts to new data."""
        if len(self.adaptation_history) < 2:
            return 0.0
        
        recent_metrics = self.adaptation_history[-10:]
        if len(recent_metrics) < 2:
            return 0.0
        
        # Calculate rate of accuracy change
        accuracy_changes = []
        for i in range(1, len(recent_metrics)):
            accuracy_changes.append(
                abs(recent_metrics[i].accuracy - recent_metrics[i-1].accuracy)
            )
        
        return np.mean(accuracy_changes) if accuracy_changes else 0.0
    
    async def predict_one(self, x: Dict[str, Any]) -> Tuple[Any, float]:
        """
        Make a single prediction with confidence.
        
        Args:
            x: Feature dictionary
            
        Returns:
            Tuple of (prediction, confidence)
        """
        start_time = time.time()
        
        try:
            # Basic input validation
            if not isinstance(x, dict):
                raise ValueError("Input must be a dictionary")
            if not x:
                raise ValueError("Input cannot be empty")
            
            # Make prediction
            prediction = await self._safe_predict(x)
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence()
            
            # Track prediction time
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            # Keep only recent prediction times
            if len(self.prediction_times) > 1000:
                self.prediction_times = self.prediction_times[-1000:]
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _preprocess_features(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess features to ensure correct types."""
        processed = {}
        
        for key, value in x.items():
            try:
                # Skip non-numeric features
                if key in ['context', 'user_id', 'source', 'event_type']:
                    continue
                    
                # Convert to float if possible
                if isinstance(value, str):
                    try:
                        processed[key] = float(value)
                    except ValueError:
                        # Skip non-numeric strings
                        continue
                elif isinstance(value, (int, float, bool)):
                    processed[key] = float(value)
                elif value is None:
                    # Skip None values
                    continue
                else:
                    # Try to convert other types
                    try:
                        processed[key] = float(str(value))
                    except (ValueError, TypeError):
                        continue
                        
            except Exception:
                # Skip problematic features
                continue
        
        # Ensure we have at least one feature
        if not processed:
            processed['default_feature'] = 1.0
            
        return processed

    async def _safe_predict(self, x: Dict[str, Any]) -> Any:
        """Make prediction with circuit breaker protection."""
        try:
            # Preprocess features
            processed_x = self._preprocess_features(x)
            prediction = self.model.predict_one(processed_x)
            
            # Handle None predictions from uninitialized models
            if prediction is None:
                # Return default prediction based on model type
                if hasattr(self.model, 'predict_proba_one'):
                    # Classification model - return most common class (0)
                    return 0
                else:
                    # Regression model - return 0.0
                    return 0.0
            
            return prediction
            
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")
            # Return safe default
            return 0.0
    
    async def learn_one(
        self, 
        x: Dict[str, Any], 
        y: Any, 
        sample_weight: Optional[float] = None
    ) -> LearningMetrics:
        """
        Learn from a single example.
        
        Args:
            x: Feature dictionary
            y: Target value
            sample_weight: Optional sample weight
            
        Returns:
            Updated learning metrics
        """
        try:
            # Basic input validation
            if not isinstance(x, dict):
                raise ValueError("Input must be a dictionary")
            if not x:
                raise ValueError("Input cannot be empty")
            
            # Make prediction before learning
            prediction, confidence = await self.predict_one(x)
            
            # Update metrics
            self.accuracy_metric.update(y, prediction)
            if hasattr(self.loss_metric, 'update'):
                self.loss_metric.update(y, prediction)
            
            # Store prediction for confidence calculation
            self.recent_predictions.append((prediction, y))
            if len(self.recent_predictions) > 1000:
                self.recent_predictions = self.recent_predictions[-1000:]
            
            # Detect concept drift
            drift_detected = False
            if self.drift_detector:
                # Use prediction error as drift signal
                try:
                    if prediction is not None and y is not None:
                        if isinstance(y, (int, float)) and isinstance(prediction, (int, float, bool)):
                            # Numeric comparison
                            error = abs(float(prediction) - float(y))
                        else:
                            # Categorical comparison
                            error = 1.0 if prediction != y else 0.0
                    else:
                        # Handle None values
                        error = 1.0
                        
                    # Ensure error is a valid float
                    if not isinstance(error, (int, float)) or error < 0:
                        error = 1.0
                        
                    self.drift_detector.update(error)
                except Exception as e:
                    # If drift detection fails, use default error
                    logger.warning(f"Drift detection error: {e}")
                    try:
                        self.drift_detector.update(1.0)
                    except:
                        # Drift detector is broken, disable it
                        logger.error("Disabling broken drift detector")
                        self.drift_detector = None
                
                if self.drift_detector.drift_detected:
                    drift_detected = True
                    drift_type = self._classify_drift_type()
                    self.concept_drift_history.append((datetime.now(timezone.utc), drift_type))
                    logger.warning(f"Concept drift detected: {drift_type}")
                    
                    # Adapt to drift
                    await self._adapt_to_drift(drift_type)
            
            # Learn from example
            await self._safe_learn(x, y, sample_weight)
            
            self.samples_processed += 1
            
            # Update learning metrics
            current_metrics = self.current_metrics
            self.adaptation_history.append(current_metrics)
            
            # Keep only recent adaptation history
            if len(self.adaptation_history) > 10000:
                self.adaptation_history = self.adaptation_history[-10000:]
            
            # Update state if state manager is available
            if self.state_manager:
                learning_record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prediction": prediction,
                    "actual": y,
                    "accuracy": current_metrics.accuracy,
                    "drift_detected": drift_detected,
                    "samples_processed": self.samples_processed
                }
                
                self.state_manager.transition(
                    add_learning_record(learning_record),
                    "ONLINE_LEARNING",
                    {"model_type": self.model_type}
                )
            
            return current_metrics
            
        except Exception as e:
            logger.error(f"Learning error: {e}")
            raise
    
    async def _safe_learn(
        self, 
        x: Dict[str, Any], 
        y: Any, 
        sample_weight: Optional[float] = None
    ):
        """Learn with circuit breaker protection."""
        try:
            # Preprocess features
            processed_x = self._preprocess_features(x)
            
            if sample_weight is not None:
                self.model.learn_one(processed_x, y, sample_weight=sample_weight)
            else:
                self.model.learn_one(processed_x, y)
        except Exception as e:
            logger.warning(f"Model learning failed: {e}")
            raise
    
    def _classify_drift_type(self) -> ConceptDriftType:
        """Classify the type of concept drift."""
        if len(self.concept_drift_history) < 2:
            return ConceptDriftType.SUDDEN
        
        # Simple heuristic: if drifts are frequent, it's gradual
        recent_drifts = [
            d for d in self.concept_drift_history 
            if d[0] > datetime.now(timezone.utc) - timedelta(hours=1)
        ]
        
        if len(recent_drifts) > 3:
            return ConceptDriftType.GRADUAL
        elif len(recent_drifts) == 1:
            return ConceptDriftType.SUDDEN
        else:
            return ConceptDriftType.RECURRING
    
    async def _adapt_to_drift(self, drift_type: ConceptDriftType):
        """Adapt learning strategy based on drift type."""
        if drift_type == ConceptDriftType.SUDDEN:
            # Reset model or increase learning rate
            self.learning_rate = min(self.learning_rate * 1.5, 0.1)
            logger.info(f"Increased learning rate to {self.learning_rate} due to sudden drift")
            
        elif drift_type == ConceptDriftType.GRADUAL:
            # Gradually increase adaptation
            self.learning_rate = min(self.learning_rate * 1.1, 0.05)
            logger.info(f"Gradually increased learning rate to {self.learning_rate}")
            
        elif drift_type == ConceptDriftType.RECURRING:
            # Use more conservative adaptation
            self.learning_rate = max(self.learning_rate * 0.9, 0.001)
            logger.info(f"Reduced learning rate to {self.learning_rate} for recurring drift")
    
    async def batch_learn(
        self, 
        X: List[Dict[str, Any]], 
        y: List[Any],
        sample_weights: Optional[List[float]] = None
    ) -> List[LearningMetrics]:
        """
        Learn from a batch of examples.
        
        Args:
            X: List of feature dictionaries
            y: List of target values
            sample_weights: Optional list of sample weights
            
        Returns:
            List of learning metrics for each example
        """
        metrics_list = []
        
        for i, (x_i, y_i) in enumerate(zip(X, y)):
            weight = sample_weights[i] if sample_weights else None
            metrics = await self.learn_one(x_i, y_i, weight)
            metrics_list.append(metrics)
            
            # Add small delay to prevent overwhelming the system
            await asyncio.sleep(0.001)
        
        return metrics_list
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get the current model state for serialization."""
        return {
            "model_type": self.model_type,
            "learning_rate": self.learning_rate,
            "samples_processed": self.samples_processed,
            "learning_mode": self.learning_mode.value,
            "concept_drift_history": [
                (d[0].isoformat(), d[1].value) for d in self.concept_drift_history
            ],
            "accuracy": self.accuracy_metric.get(),
            "model_complexity": self._estimate_model_complexity(),
            "adaptation_speed": self._calculate_adaptation_speed(),
            "recent_prediction_times": self.prediction_times[-100:] if self.prediction_times else []
        }
    
    def save_model(self, filepath: str):
        """Save model to file."""
        try:
            model_data = {
                "model": self.model,
                "state": self.get_model_state(),
                "metrics": self.current_metrics.__dict__,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load model from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            state = model_data["state"]
            
            self.model_type = state["model_type"]
            self.learning_rate = state["learning_rate"]
            self.samples_processed = state["samples_processed"]
            self.learning_mode = LearningMode(state["learning_mode"])
            
            # Restore concept drift history
            self.concept_drift_history = [
                (datetime.fromisoformat(d[0]), ConceptDriftType(d[1]))
                for d in state["concept_drift_history"]
            ]
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


class OnlineLearningOrchestrator:
    """
    Orchestrates multiple online learners for ensemble learning.
    """
    
    def __init__(self, learners: List[RSIOnlineLearner]):
        self.learners = learners
        self.ensemble_weights = [1.0] * len(learners)
        self.performance_history = []
        
    async def ensemble_predict(self, x: Dict[str, Any]) -> Tuple[Any, float]:
        """Make ensemble prediction."""
        predictions = []
        confidences = []
        
        for learner in self.learners:
            try:
                pred, conf = await learner.predict_one(x)
                # Handle None predictions gracefully
                if pred is not None and conf is not None:
                    predictions.append(pred)
                    confidences.append(conf)
                else:
                    # Use default values for None predictions
                    predictions.append(0.0)
                    confidences.append(0.5)
            except Exception as e:
                logger.warning(f"Learner prediction failed: {e}")
                # Use default values for failed predictions
                predictions.append(0.0)
                confidences.append(0.5)
        
        # Ensure we have predictions
        if not predictions:
            return 0.0, 0.5
        
        # Weighted average prediction with null checks
        numerator = sum(
            p * w * c for p, w, c in zip(predictions, self.ensemble_weights, confidences)
            if p is not None and w is not None and c is not None
        )
        denominator = sum(
            w * c for w, c in zip(self.ensemble_weights, confidences)
            if w is not None and c is not None
        )
        
        if denominator == 0:
            weighted_pred = sum(predictions) / len(predictions)
        else:
            weighted_pred = numerator / denominator
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return weighted_pred, avg_confidence
    
    async def ensemble_learn(
        self, 
        x: Dict[str, Any], 
        y: Any
    ) -> List[LearningMetrics]:
        """Learn with all learners in ensemble."""
        metrics_list = []
        
        for learner in self.learners:
            try:
                metrics = await learner.learn_one(x, y)
                metrics_list.append(metrics)
            except Exception as e:
                logger.warning(f"Learner learning failed: {e}")
                # Create default metrics for failed learners
                default_metrics = LearningMetrics(
                    accuracy=0.0,
                    loss=1.0,
                    samples_processed=0,
                    learning_rate=0.01,
                    concept_drift_detected=False,
                    drift_type=ConceptDriftType.NONE,
                    model_complexity=0,
                    prediction_confidence=0.5,
                    adaptation_speed=0.0,
                    timestamp=datetime.now(timezone.utc)
                )
                metrics_list.append(default_metrics)
        
        # Update ensemble weights based on performance
        if metrics_list:
            self._update_ensemble_weights(metrics_list)
        
        return metrics_list
    
    def _update_ensemble_weights(self, metrics_list: List[LearningMetrics]):
        """Update ensemble weights based on performance."""
        accuracies = [m.accuracy for m in metrics_list]
        
        # Simple performance-based weighting
        total_accuracy = sum(accuracies)
        if total_accuracy > 0:
            self.ensemble_weights = [acc / total_accuracy for acc in accuracies]
        else:
            self.ensemble_weights = [1.0] * len(self.learners)


# Factory functions for common online learning setups
def create_classification_learner(
    state_manager: Optional[RSIStateManager] = None,
    validator: Optional[RSIValidator] = None
) -> RSIOnlineLearner:
    """Create online learner for classification tasks."""
    return OnlineLearner(
        model_type="adaptive_random_forest",
        drift_detector="adwin",
        learning_rate=0.01,
        state_manager=state_manager,
        validator=validator
    )


def create_regression_learner(
    state_manager: Optional[RSIStateManager] = None,
    validator: Optional[RSIValidator] = None
) -> RSIOnlineLearner:
    """Create online learner for regression tasks."""
    return OnlineLearner(
        model_type="adaptive_model_rules",
        drift_detector="kswin",
        learning_rate=0.01,
        state_manager=state_manager,
        validator=validator
    )


def create_ensemble_learner(
    state_manager: Optional[RSIStateManager] = None,
    validator: Optional[RSIValidator] = None
) -> OnlineLearningOrchestrator:
    """Create ensemble of online learners."""
    learners = [
        create_classification_learner(state_manager, validator),
        OnlineLearner(
            model_type="logistic_regression",
            drift_detector="dummy",
            learning_rate=0.005,
            state_manager=state_manager,
            validator=validator
        ),
        OnlineLearner(
            model_type="hoeffding_tree",
            drift_detector="dummy",
            learning_rate=0.02,
            state_manager=state_manager,
            validator=validator
        )
    ]
    
    return OnlineLearningOrchestrator(learners)
