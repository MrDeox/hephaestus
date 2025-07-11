"""
Advanced Uncertainty Quantification System for RSI AI.
Provides robust confidence estimation and uncertainty tracking.
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from loguru import logger

try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow Probability not available. Using fallback uncertainty estimation.")

class UncertaintyType(Enum):
    EPISTEMIC = "epistemic"  # Model uncertainty
    ALEATORIC = "aleatoric"  # Data uncertainty
    TOTAL = "total"

@dataclass
class UncertaintyEstimate:
    """Comprehensive uncertainty estimate"""
    timestamp: float
    prediction_mean: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    confidence_score: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    uncertainty_type: str
    sample_size: int
    estimation_method: str

class RSIUncertaintyEstimator:
    """Production-ready uncertainty quantification for RSI systems"""
    
    def __init__(self, input_dim: int, hidden_units: int = 50, 
                 estimation_method: str = "monte_carlo_dropout"):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.estimation_method = estimation_method
        self.uncertainty_history: List[UncertaintyEstimate] = []
        
        # Initialize appropriate model based on availability
        if TF_AVAILABLE and estimation_method == "bayesian":
            self.model = self._build_bayesian_model()
        else:
            self.model = self._build_dropout_model()
            
        self.calibration_data = []
        self.is_calibrated = False
        
    def _build_bayesian_model(self):
        """Build Bayesian neural network for uncertainty quantification"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow Probability not available for Bayesian models")
            
        model = tf.keras.Sequential([
            tfp.layers.DenseVariational(
                units=self.hidden_units,
                make_prior_fn=self._make_prior_fn,
                make_posterior_fn=self._make_posterior_fn,
                kl_weight=1e-5,
                activation='relu',
                input_shape=(self.input_dim,)
            ),
            tfp.layers.DenseVariational(
                units=self.hidden_units // 2,
                make_prior_fn=self._make_prior_fn,
                make_posterior_fn=self._make_posterior_fn,
                kl_weight=1e-5,
                activation='relu'
            ),
            tfp.layers.DenseVariational(
                units=1,
                make_prior_fn=self._make_prior_fn,
                make_posterior_fn=self._make_posterior_fn,
                kl_weight=1e-5
            )
        ])
        
        # Compile with appropriate loss for uncertainty
        def negative_log_likelihood(y_true, y_pred):
            return -y_pred.log_prob(y_true)
            
        model.compile(
            optimizer='adam',
            loss=negative_log_likelihood,
            metrics=['mae']
        )
        
        return model
    
    def _make_prior_fn(self, kernel_size, bias_size, dtype=None):
        """Create prior distribution for Bayesian layers"""
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1),
                    reinterpreted_batch_ndims=1
                )
            )
        ])
    
    def _make_posterior_fn(self, kernel_size, bias_size, dtype=None):
        """Create posterior distribution for Bayesian layers"""
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(
                        loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(c + t[..., n:])
                    ),
                    reinterpreted_batch_ndims=1
                )
            )
        ])
    
    def _build_dropout_model(self):
        """Build standard model with dropout for MC uncertainty estimation"""
        if TF_AVAILABLE:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(self.hidden_units, activation='relu', 
                                    input_shape=(self.input_dim,)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.hidden_units // 2, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        else:
            # Fallback: simple numpy-based model for demonstration
            return None
    
    async def predict_with_uncertainty(self, X: np.ndarray, 
                                     n_samples: int = 100,
                                     confidence_level: float = 0.95) -> UncertaintyEstimate:
        """Generate predictions with comprehensive uncertainty estimates"""
        
        if self.estimation_method == "bayesian" and TF_AVAILABLE:
            return await self._bayesian_prediction(X, n_samples, confidence_level)
        elif self.estimation_method == "monte_carlo_dropout" and TF_AVAILABLE:
            return await self._mc_dropout_prediction(X, n_samples, confidence_level)
        else:
            return await self._fallback_prediction(X, n_samples, confidence_level)
    
    async def _bayesian_prediction(self, X: np.ndarray, n_samples: int, 
                                 confidence_level: float) -> UncertaintyEstimate:
        """Bayesian uncertainty estimation"""
        predictions = []
        
        for _ in range(n_samples):
            pred = self.model(X, training=True)
            if hasattr(pred, 'sample'):
                predictions.append(pred.sample().numpy())
            else:
                predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        epistemic_uncertainty = np.std(predictions, axis=0)
        
        # For Bayesian models, epistemic uncertainty dominates
        aleatoric_uncertainty = epistemic_uncertainty * 0.1
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(predictions, lower_percentile, axis=0)
        ci_upper = np.percentile(predictions, upper_percentile, axis=0)
        
        confidence_score = 1 / (1 + total_uncertainty)
        
        return UncertaintyEstimate(
            timestamp=time.time(),
            prediction_mean=float(mean_pred.item() if mean_pred.size == 1 else mean_pred[0]),
            epistemic_uncertainty=float(epistemic_uncertainty.item() if epistemic_uncertainty.size == 1 else epistemic_uncertainty[0]),
            aleatoric_uncertainty=float(aleatoric_uncertainty.item() if aleatoric_uncertainty.size == 1 else aleatoric_uncertainty[0]),
            total_uncertainty=float(total_uncertainty.item() if total_uncertainty.size == 1 else total_uncertainty[0]),
            confidence_score=float(confidence_score.item() if confidence_score.size == 1 else confidence_score[0]),
            confidence_interval_lower=float(ci_lower.item() if ci_lower.size == 1 else ci_lower[0]),
            confidence_interval_upper=float(ci_upper.item() if ci_upper.size == 1 else ci_upper[0]),
            uncertainty_type="bayesian",
            sample_size=n_samples,
            estimation_method="bayesian_neural_network"
        )
    
    async def _mc_dropout_prediction(self, X: np.ndarray, n_samples: int, 
                                   confidence_level: float) -> UncertaintyEstimate:
        """Monte Carlo dropout uncertainty estimation"""
        predictions = []
        
        for _ in range(n_samples):
            pred = self.model(X, training=True)  # Keep dropout active
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate epistemic uncertainty from MC samples
        mean_pred = np.mean(predictions, axis=0)
        epistemic_uncertainty = np.std(predictions, axis=0)
        
        # Estimate aleatoric uncertainty (data noise)
        # This is a simplified estimation - in practice, this would come from data analysis
        aleatoric_uncertainty = epistemic_uncertainty * 0.3
        
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        # Confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(predictions, lower_percentile, axis=0)
        ci_upper = np.percentile(predictions, upper_percentile, axis=0)
        
        confidence_score = 1 / (1 + total_uncertainty)
        
        return UncertaintyEstimate(
            timestamp=time.time(),
            prediction_mean=float(mean_pred.item() if mean_pred.size == 1 else mean_pred[0]),
            epistemic_uncertainty=float(epistemic_uncertainty.item() if epistemic_uncertainty.size == 1 else epistemic_uncertainty[0]),
            aleatoric_uncertainty=float(aleatoric_uncertainty.item() if aleatoric_uncertainty.size == 1 else aleatoric_uncertainty[0]),
            total_uncertainty=float(total_uncertainty.item() if total_uncertainty.size == 1 else total_uncertainty[0]),
            confidence_score=float(confidence_score.item() if confidence_score.size == 1 else confidence_score[0]),
            confidence_interval_lower=float(ci_lower.item() if ci_lower.size == 1 else ci_lower[0]),
            confidence_interval_upper=float(ci_upper.item() if ci_upper.size == 1 else ci_upper[0]),
            uncertainty_type="monte_carlo_dropout",
            sample_size=n_samples,
            estimation_method="mc_dropout"
        )
    
    async def _fallback_prediction(self, X: np.ndarray, n_samples: int, 
                                 confidence_level: float) -> UncertaintyEstimate:
        """Fallback uncertainty estimation without deep learning libraries"""
        
        # Simple ensemble-based uncertainty estimation
        predictions = []
        
        for i in range(n_samples):
            # Add noise to simulate model uncertainty
            noise_scale = 0.1
            noise = np.random.normal(0, noise_scale, X.shape)
            noisy_input = X + noise
            
            # Simple linear prediction with noise
            pred = np.sum(noisy_input, axis=-1) * 0.1 + np.random.normal(0, 0.05)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions)
        epistemic_uncertainty = np.std(predictions)
        aleatoric_uncertainty = 0.05  # Fixed data noise assumption
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        # Simple confidence intervals
        ci_lower = np.percentile(predictions, 2.5)
        ci_upper = np.percentile(predictions, 97.5)
        
        confidence_score = 1 / (1 + total_uncertainty)
        
        return UncertaintyEstimate(
            timestamp=time.time(),
            prediction_mean=float(mean_pred),
            epistemic_uncertainty=float(epistemic_uncertainty),
            aleatoric_uncertainty=float(aleatoric_uncertainty),
            total_uncertainty=float(total_uncertainty),
            confidence_score=float(confidence_score),
            confidence_interval_lower=float(ci_lower),
            confidence_interval_upper=float(ci_upper),
            uncertainty_type="ensemble_fallback",
            sample_size=n_samples,
            estimation_method="fallback_ensemble"
        )
    
    async def assess_prediction_confidence(self, uncertainty_estimate: UncertaintyEstimate, 
                                         threshold: float = 0.1) -> Dict[str, Any]:
        """Assess confidence levels for RSI decision-making"""
        
        confidence_assessment = {
            'confidence_score': uncertainty_estimate.confidence_score,
            'high_confidence': uncertainty_estimate.total_uncertainty < threshold,
            'medium_confidence': threshold <= uncertainty_estimate.total_uncertainty < threshold * 2,
            'low_confidence': uncertainty_estimate.total_uncertainty >= threshold * 2,
            'requires_human_review': uncertainty_estimate.total_uncertainty > threshold * 3,
            'uncertainty_level': self._categorize_uncertainty(uncertainty_estimate.total_uncertainty, threshold),
            'epistemic_dominance': uncertainty_estimate.epistemic_uncertainty > uncertainty_estimate.aleatoric_uncertainty,
            'prediction_reliable': uncertainty_estimate.confidence_score > 0.7,
            'confidence_interval_width': uncertainty_estimate.confidence_interval_upper - uncertainty_estimate.confidence_interval_lower
        }
        
        # Add calibration assessment if available
        if self.is_calibrated:
            confidence_assessment['calibration_quality'] = await self._assess_calibration_quality(uncertainty_estimate)
        
        return confidence_assessment
    
    def _categorize_uncertainty(self, total_uncertainty: float, threshold: float) -> str:
        """Categorize uncertainty level"""
        if total_uncertainty < threshold:
            return "low"
        elif total_uncertainty < threshold * 2:
            return "medium"
        elif total_uncertainty < threshold * 3:
            return "high"
        else:
            return "very_high"
    
    async def calibrate_uncertainty(self, validation_data: List[Tuple[np.ndarray, float]]):
        """Calibrate uncertainty estimates using validation data"""
        logger.info("Calibrating uncertainty estimator with {} samples", len(validation_data))
        
        calibration_results = []
        
        for X, true_value in validation_data:
            # Get uncertainty estimate
            uncertainty_est = await self.predict_with_uncertainty(X.reshape(1, -1), n_samples=50)
            
            # Check if true value falls within confidence interval
            in_ci = (uncertainty_est.confidence_interval_lower <= true_value <= 
                    uncertainty_est.confidence_interval_upper)
            
            # Calculate prediction error
            prediction_error = abs(uncertainty_est.prediction_mean - true_value)
            
            calibration_results.append({
                'prediction_error': prediction_error,
                'uncertainty_estimate': uncertainty_est.total_uncertainty,
                'in_confidence_interval': in_ci,
                'confidence_score': uncertainty_est.confidence_score
            })
        
        self.calibration_data = calibration_results
        self.is_calibrated = True
        
        # Calculate calibration metrics
        coverage = np.mean([r['in_confidence_interval'] for r in calibration_results])
        avg_uncertainty = np.mean([r['uncertainty_estimate'] for r in calibration_results])
        avg_error = np.mean([r['prediction_error'] for r in calibration_results])
        
        logger.info("Calibration complete: Coverage={:.3f}, Avg Uncertainty={:.3f}, Avg Error={:.3f}", 
                   coverage, avg_uncertainty, avg_error)
        
        return {
            'coverage': coverage,
            'average_uncertainty': avg_uncertainty,
            'average_error': avg_error,
            'calibration_quality': 'good' if 0.9 <= coverage <= 0.98 else 'needs_improvement'
        }
    
    async def _assess_calibration_quality(self, uncertainty_estimate: UncertaintyEstimate) -> str:
        """Assess the quality of uncertainty calibration"""
        if not self.is_calibrated:
            return "not_calibrated"
        
        # Compare current uncertainty with historical calibration data
        similar_uncertainties = [
            r for r in self.calibration_data
            if abs(r['uncertainty_estimate'] - uncertainty_estimate.total_uncertainty) < 0.05
        ]
        
        if not similar_uncertainties:
            return "insufficient_data"
        
        # Check if uncertainty estimates are well-calibrated for similar cases
        coverage = np.mean([r['in_confidence_interval'] for r in similar_uncertainties])
        
        if 0.9 <= coverage <= 0.98:
            return "well_calibrated"
        elif coverage < 0.9:
            return "overconfident"
        else:
            return "underconfident"
    
    def update_uncertainty_history(self, uncertainty_estimate: UncertaintyEstimate):
        """Update uncertainty tracking history"""
        self.uncertainty_history.append(uncertainty_estimate)
        
        # Keep last 1000 estimates
        if len(self.uncertainty_history) > 1000:
            self.uncertainty_history.pop(0)
    
    def get_uncertainty_trends(self) -> Dict[str, Any]:
        """Analyze uncertainty trends over time"""
        if len(self.uncertainty_history) < 10:
            return {'status': 'insufficient_data'}
        
        recent_estimates = self.uncertainty_history[-100:]
        
        # Calculate trends
        timestamps = [e.timestamp for e in recent_estimates]
        uncertainties = [e.total_uncertainty for e in recent_estimates]
        confidences = [e.confidence_score for e in recent_estimates]
        
        # Linear trend analysis
        if len(recent_estimates) > 1:
            time_diffs = np.diff(timestamps)
            uncertainty_diffs = np.diff(uncertainties)
            uncertainty_trend = np.mean(uncertainty_diffs / time_diffs) if np.mean(time_diffs) > 0 else 0
            
            confidence_diffs = np.diff(confidences)
            confidence_trend = np.mean(confidence_diffs / time_diffs) if np.mean(time_diffs) > 0 else 0
        else:
            uncertainty_trend = 0
            confidence_trend = 0
        
        return {
            'average_uncertainty': np.mean(uncertainties),
            'uncertainty_variance': np.var(uncertainties),
            'uncertainty_trend': uncertainty_trend,
            'average_confidence': np.mean(confidences),
            'confidence_trend': confidence_trend,
            'epistemic_ratio': np.mean([e.epistemic_uncertainty / e.total_uncertainty 
                                      for e in recent_estimates if e.total_uncertainty > 0]),
            'trend_analysis': {
                'uncertainty_increasing': uncertainty_trend > 0.001,
                'confidence_improving': confidence_trend > 0.001,
                'stability': 'stable' if abs(uncertainty_trend) < 0.001 else 'changing'
            }
        }
    
    async def suggest_improvement_actions(self, uncertainty_estimate: UncertaintyEstimate) -> List[str]:
        """Suggest actions to improve prediction confidence"""
        suggestions = []
        
        if uncertainty_estimate.total_uncertainty > 0.3:
            suggestions.append("Consider collecting more training data")
        
        if uncertainty_estimate.epistemic_uncertainty > uncertainty_estimate.aleatoric_uncertainty * 2:
            suggestions.append("Model uncertainty is high - consider ensemble methods or larger model")
        
        if uncertainty_estimate.aleatoric_uncertainty > uncertainty_estimate.epistemic_uncertainty * 2:
            suggestions.append("Data uncertainty is high - review data quality and preprocessing")
        
        if uncertainty_estimate.confidence_score < 0.6:
            suggestions.append("Low confidence detected - recommend human oversight")
        
        if self.is_calibrated:
            calibration_quality = await self._assess_calibration_quality(uncertainty_estimate)
            if calibration_quality == "overconfident":
                suggestions.append("Model appears overconfident - recalibrate uncertainty estimates")
            elif calibration_quality == "underconfident":
                suggestions.append("Model appears underconfident - consider reducing uncertainty penalties")
        
        return suggestions

class UncertaintyAggregator:
    """Aggregate uncertainty estimates from multiple sources"""
    
    def __init__(self):
        self.estimators: Dict[str, RSIUncertaintyEstimator] = {}
        self.weights: Dict[str, float] = {}
    
    def add_estimator(self, name: str, estimator: RSIUncertaintyEstimator, weight: float = 1.0):
        """Add uncertainty estimator to the ensemble"""
        self.estimators[name] = estimator
        self.weights[name] = weight
    
    async def aggregate_uncertainty(self, X: np.ndarray, n_samples: int = 100) -> UncertaintyEstimate:
        """Aggregate uncertainty estimates from multiple estimators"""
        if not self.estimators:
            raise ValueError("No estimators available for aggregation")
        
        estimates = {}
        total_weight = sum(self.weights.values())
        
        # Get estimates from all estimators
        for name, estimator in self.estimators.items():
            estimates[name] = await estimator.predict_with_uncertainty(X, n_samples)
        
        # Weighted aggregation
        weighted_mean = sum(
            est.prediction_mean * self.weights[name] / total_weight
            for name, est in estimates.items()
        )
        
        weighted_epistemic = sum(
            est.epistemic_uncertainty * self.weights[name] / total_weight
            for name, est in estimates.items()
        )
        
        weighted_aleatoric = sum(
            est.aleatoric_uncertainty * self.weights[name] / total_weight
            for name, est in estimates.items()
        )
        
        # Add disagreement uncertainty
        mean_predictions = [est.prediction_mean for est in estimates.values()]
        disagreement_uncertainty = np.std(mean_predictions) if len(mean_predictions) > 1 else 0
        
        total_uncertainty = np.sqrt(
            weighted_epistemic**2 + 
            weighted_aleatoric**2 + 
            disagreement_uncertainty**2
        )
        
        confidence_score = 1 / (1 + total_uncertainty)
        
        # Aggregate confidence intervals
        all_ci_lower = [est.confidence_interval_lower for est in estimates.values()]
        all_ci_upper = [est.confidence_interval_upper for est in estimates.values()]
        
        return UncertaintyEstimate(
            timestamp=time.time(),
            prediction_mean=weighted_mean,
            epistemic_uncertainty=weighted_epistemic + disagreement_uncertainty,
            aleatoric_uncertainty=weighted_aleatoric,
            total_uncertainty=total_uncertainty,
            confidence_score=confidence_score,
            confidence_interval_lower=min(all_ci_lower),
            confidence_interval_upper=max(all_ci_upper),
            uncertainty_type="ensemble_aggregated",
            sample_size=n_samples,
            estimation_method="weighted_ensemble"
        )