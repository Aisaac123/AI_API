"""
Unit tests for evaluation metrics.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import mse, mae, rmse, r2_score, accuracy


def test_mse():
    """Test mean squared error."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    
    result = mse(y_true, y_pred)
    assert result == 0.0
    
    y_pred = np.array([2, 3, 4, 5, 6])
    result = mse(y_true, y_pred)
    assert result == 1.0
    print("Test MSE: PASSED")


def test_mae():
    """Test mean absolute error."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    
    result = mae(y_true, y_pred)
    assert result == 0.0
    
    y_pred = np.array([2, 3, 4, 5, 6])
    result = mae(y_true, y_pred)
    assert result == 1.0
    print("Test MAE: PASSED")


def test_rmse():
    """Test root mean squared error."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    
    result = rmse(y_true, y_pred)
    assert result == 0.0
    
    y_pred = np.array([2, 3, 4, 5, 6])
    result = rmse(y_true, y_pred)
    assert result == 1.0
    print("Test RMSE: PASSED")


def test_r2_score():
    """Test R-squared score."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    
    result = r2_score(y_true, y_pred)
    assert result == 1.0
    
    y_pred = np.array([2, 2, 2, 2, 2])  # Predict the mean
    result = r2_score(y_true, y_pred)
    assert result <= 0.0  # Should be negative since it's worse than predicting mean
    print("Test R2 score: PASSED")


def test_accuracy():
    """Test accuracy metric."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    
    result = accuracy(y_true, y_pred)
    assert result == 1.0
    
    y_pred = np.array([2, 3, 4, 5, 6])
    result = accuracy(y_true, y_pred)
    assert result < 1.0
    assert result >= 0.0
    print("Test accuracy: PASSED")


def run_all_tests():
    """Run all metric tests."""
    print("Running Evaluation Metrics Tests")
    print("=" * 50)
    test_mse()
    test_mae()
    test_rmse()
    test_r2_score()
    test_accuracy()
    print("=" * 50)
    print("All evaluation metrics tests passed!")


if __name__ == "__main__":
    run_all_tests()
