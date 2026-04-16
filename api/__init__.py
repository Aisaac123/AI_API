"""
API module exports.
This module provides a compact, MATLAB-style interface for neural networks.
"""

from .model_type import ModelType
from .neural_network import NeuralNetwork
from .config import NeuralNetworkConfig

__all__ = [
    'ModelType',
    'NeuralNetwork',
    'NeuralNetworkConfig'
]
