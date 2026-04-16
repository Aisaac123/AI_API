"""
Unit tests for backpropagation network.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import BackpropNetwork, BackpropConfig


def test_backprop_network_initialization():
    """Test backpropagation network initialization."""
    config = BackpropConfig(hidden_layers=[10, 5], learning_rate=0.01)
    model = BackpropNetwork(config)
    
    assert model.config.hidden_layers == [10, 5]
    assert model.config.learning_rate == 0.01
    assert model.is_fitted == False
    print("Test backpropagation network initialization: PASSED")


def test_backprop_network_training():
    """Test backpropagation network training."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.random.randn(50, 1)
    
    config = BackpropConfig(hidden_layers=[10, 5], learning_rate=0.01, epochs=100, random_state=42)
    model = BackpropNetwork(config)
    model.fit(X, y)
    
    assert model.is_fitted == True
    assert len(model.layers) > 0
    assert model.n_features_ == 2
    assert model.n_outputs_ == 1
    print("Test backpropagation network training: PASSED")


def test_backprop_network_prediction():
    """Test backpropagation network prediction."""
    np.random.seed(42)
    X_train = np.random.randn(50, 2)
    y_train = np.random.randn(50, 1)
    X_test = np.random.randn(10, 2)
    
    config = BackpropConfig(hidden_layers=[10, 5], learning_rate=0.01, epochs=100, random_state=42)
    model = BackpropNetwork(config)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    assert predictions.shape == (10, 1)
    assert not np.any(np.isnan(predictions))
    print("Test backpropagation network prediction: PASSED")


def test_backprop_network_summary():
    """Test backpropagation network summary."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.random.randn(50, 1)
    
    config = BackpropConfig(hidden_layers=[10, 5], learning_rate=0.01, epochs=100, random_state=42)
    model = BackpropNetwork(config)
    model.fit(X, y)
    
    summary = model.summary()
    
    assert 'model_type' in summary
    assert 'is_fitted' in summary
    assert 'hidden_layers' in summary
    assert summary['model_type'] == 'BackpropNetwork'
    print("Test backpropagation network summary: PASSED")


def run_all_tests():
    """Run all backpropagation network tests."""
    print("Running Backpropagation Network Tests")
    print("=" * 50)
    test_backprop_network_initialization()
    test_backprop_network_training()
    test_backprop_network_prediction()
    test_backprop_network_summary()
    print("=" * 50)
    print("All backpropagation network tests passed!")


if __name__ == "__main__":
    run_all_tests()
