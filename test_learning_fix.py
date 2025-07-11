#!/usr/bin/env python3
"""
Test the fixed learning system.
"""

import asyncio
from src.learning.online_learning import OnlineLearner, create_ensemble_learner

async def test_fixed_learning():
    """Test the corrected learning system."""
    print("🧪 Testing Fixed Learning System")
    print("=" * 40)
    
    # Test features
    test_features = {
        "feature1": 0.8,
        "feature2": 0.6,
        "feature3": 0.9,
        "context": "test"
    }
    
    print(f"Test features: {test_features}")
    
    # Test individual learner
    print("\n1. Testing Individual OnlineLearner...")
    try:
        learner = OnlineLearner(
            model_type="hoeffding_tree",
            drift_detector="dummy",
            learning_rate=0.01
        )
        
        # Test prediction before learning
        pred_before, conf_before = await learner.predict_one(test_features)
        print(f"   Prediction before learning: {pred_before} (confidence: {conf_before:.3f})")
        
        # Test learning
        metrics = await learner.learn_one(test_features, 1)
        print(f"   Learning metrics: accuracy={metrics.accuracy:.3f}, samples={metrics.samples_processed}")
        
        # Test prediction after learning
        pred_after, conf_after = await learner.predict_one(test_features)
        print(f"   Prediction after learning: {pred_after} (confidence: {conf_after:.3f})")
        
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test ensemble learner
    print("\n2. Testing Ensemble Learner...")
    try:
        ensemble = create_ensemble_learner()
        
        # Test ensemble prediction
        pred, conf = await ensemble.ensemble_predict(test_features)
        print(f"   Ensemble prediction: {pred} (confidence: {conf:.3f})")
        
        # Test ensemble learning
        result = await ensemble.ensemble_learn(test_features, 1)
        print(f"   Ensemble learning result: {result}")
        
        # Test another prediction
        pred2, conf2 = await ensemble.ensemble_predict(test_features)
        print(f"   Ensemble prediction after learning: {pred2} (confidence: {conf2:.3f})")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fixed_learning())