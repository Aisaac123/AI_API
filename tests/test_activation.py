"""
Unit tests for activation functions.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.activation import (
    GaussianActivation,
    MultiquadraticActivation,
    InverseMultiquadraticActivation,
    ThinPlateSplineActivation
)


def test_gaussian_activation():
    """Test Gaussian activation function."""
    activation = GaussianActivation()
    distances = np.array([[0, 1, 2], [1, 0, 1]])
    result = activation.compute(distances, sigma=1.0)
    
    assert result.shape == distances.shape
    assert result[0, 0] == 1.0  # Distance 0 should give activation 1
    assert result[0, 1] < 1.0  # Distance 1 should give activation < 1
    assert not np.any(np.isnan(result))
    print("Test Gaussian activation: PASSED")


def test_multiquadratic_activation():
    """Test multiquadratic activation function."""
    activation = MultiquadraticActivation()
    distances = np.array([[0, 1, 2], [1, 0, 1]])
    result = activation.compute(distances, sigma=1.0)
    
    assert result.shape == distances.shape
    assert result[0, 0] == 1.0  # Distance 0 with sigma=1 gives sqrt(0+1)=1
    assert not np.any(np.isnan(result))
    print("Test multiquadratic activation: PASSED")


def test_inverse_multiquadratic_activation():
    """Test inverse multiquadratic activation function."""
    activation = InverseMultiquadraticActivation()
    distances = np.array([[0, 1, 2], [1, 0, 1]])
    result = activation.compute(distances, sigma=1.0)
    
    assert result.shape == distances.shape
    assert result[0, 0] == 1.0  # Distance 0 with sigma=1 gives 1/sqrt(0+1)=1
    assert result[0, 1] < 1.0  # Distance 1 should give activation < 1
    assert not np.any(np.isnan(result))
    print("Test inverse multiquadratic activation: PASSED")


def test_thin_plate_spline_activation():
    """Test thin plate spline activation function."""
    activation = ThinPlateSplineActivation()
    distances = np.array([[0, 1, 2], [1, 0, 1]])
    result = activation.compute(distances, sigma=1.0)
    
    assert result.shape == distances.shape
    assert result[0, 0] == 0.0  # Distance 0 should give activation 0 (by convention)
    assert not np.any(np.isnan(result))
    print("Test thin plate spline activation: PASSED")


def run_all_tests():
    """Run all activation function tests."""
    print("Running Activation Function Tests")
    print("=" * 50)
    test_gaussian_activation()
    test_multiquadratic_activation()
    test_inverse_multiquadratic_activation()
    test_thin_plate_spline_activation()
    print("=" * 50)
    print("All activation function tests passed!")


if __name__ == "__main__":
    run_all_tests()
