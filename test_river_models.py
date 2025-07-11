#!/usr/bin/env python3
"""
Test River models to diagnose prediction issues.
"""

from river import linear_model, tree, ensemble, optim

def test_river_models():
    """Test basic River model functionality."""
    print("Testing River Models")
    print("=" * 40)
    
    # Test data
    test_features = {
        "feature1": 0.8,
        "feature2": 0.6,
        "feature3": 0.9
    }
    
    print(f"Test features: {test_features}")
    
    # Test Logistic Regression
    print("\n1. Testing Logistic Regression...")
    try:
        lr_model = linear_model.LogisticRegression(
            optimizer=optim.SGD(lr=0.01)
        )
        
        # Test prediction before learning (should handle gracefully)
        pred_before = lr_model.predict_one(test_features)
        print(f"   Prediction before learning: {pred_before} (type: {type(pred_before)})")
        
        # Learn from a sample
        lr_model.learn_one(test_features, 1)
        
        # Test prediction after learning
        pred_after = lr_model.predict_one(test_features)
        print(f"   Prediction after learning: {pred_after} (type: {type(pred_after)})")
        
        # Test prediction probability
        prob = lr_model.predict_proba_one(test_features)
        print(f"   Prediction probabilities: {prob}")
        
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test Hoeffding Tree
    print("\n2. Testing Hoeffding Tree...")
    try:
        tree_model = tree.HoeffdingTreeClassifier(max_depth=8)
        
        pred_before = tree_model.predict_one(test_features)
        print(f"   Prediction before learning: {pred_before} (type: {type(pred_before)})")
        
        tree_model.learn_one(test_features, 1)
        pred_after = tree_model.predict_one(test_features)
        print(f"   Prediction after learning: {pred_after} (type: {type(pred_after)})")
        
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test ADWIN Bagging
    print("\n3. Testing ADWIN Bagging...")
    try:
        bagging_model = ensemble.ADWINBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(),
            n_models=3
        )
        
        pred_before = bagging_model.predict_one(test_features)
        print(f"   Prediction before learning: {pred_before} (type: {type(pred_before)})")
        
        bagging_model.learn_one(test_features, 1)
        pred_after = bagging_model.predict_one(test_features)
        print(f"   Prediction after learning: {pred_after} (type: {type(pred_after)})")
        
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test Linear Regression (for comparison)
    print("\n4. Testing Linear Regression...")
    try:
        reg_model = linear_model.LinearRegression(
            optimizer=optim.SGD(lr=0.01)
        )
        
        pred_before = reg_model.predict_one(test_features)
        print(f"   Prediction before learning: {pred_before} (type: {type(pred_before)})")
        
        reg_model.learn_one(test_features, 0.8)
        pred_after = reg_model.predict_one(test_features)
        print(f"   Prediction after learning: {pred_after} (type: {type(pred_after)})")
        
    except Exception as e:
        print(f"   ERROR: {e}")

if __name__ == "__main__":
    test_river_models()