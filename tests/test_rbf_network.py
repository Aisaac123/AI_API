"""
Unit tests for RBF network.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import RBFNetwork, RBFConfig
from src.core.activation import GaussianActivation


def test_rbf_network_initialization():
    """Test RBF network initialization."""
    config = RBFConfig(n_centers=10, sigma=1.0)
    model = RBFNetwork(config)
    
    assert model.config.n_centers == 10
    assert model.config.sigma == 1.0
    assert model.is_fitted == False
    print("Test RBF network initialization: PASSED")


def test_rbf_network_training():
    """Test RBF network training."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.random.randn(50, 1)
    
    config = RBFConfig(n_centers=10, sigma=1.0, random_state=42)
    model = RBFNetwork(config)
    model.fit(X, y)
    
    assert model.is_fitted == True
    assert model.weights is not None
    assert model.n_features_ == 2
    assert model.n_outputs_ == 1
    print("Test RBF network training: PASSED")


def test_rbf_network_prediction():
    """Test RBF network prediction."""
    np.random.seed(42)
    X_train = np.random.randn(50, 2)
    y_train = np.random.randn(50, 1)
    X_test = np.random.randn(10, 2)
    
    config = RBFConfig(n_centers=10, sigma=1.0, random_state=42)
    model = RBFNetwork(config)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    assert predictions.shape == (10, 1)
    assert not np.any(np.isnan(predictions))
    print("Test RBF network prediction: PASSED")


def test_rbf_network_summary():
    """Test RBF network summary."""
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.random.randn(50, 1)
    
    config = RBFConfig(n_centers=10, sigma=1.0, random_state=42)
    model = RBFNetwork(config)
    model.fit(X, y)
    
    summary = model.summary()
    
    assert 'model_type' in summary
    assert 'is_fitted' in summary
    assert 'n_centers' in summary
    assert summary['model_type'] == 'RBFNetwork'
    print("Test RBF network summary: PASSED")


def run_all_tests():
    """Run all RBF network tests."""
    print("Running RBF Network Tests")
    print("=" * 50)
    test_rbf_network_initialization()
    test_rbf_network_training()
    test_rbf_network_prediction()
    test_rbf_network_summary()
    print("=" * 50)
    print("All RBF network tests passed!")


if __name__ == "__main__":
    run_all_tests()
