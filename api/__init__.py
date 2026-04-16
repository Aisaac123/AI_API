"""
API module exports.
This module provides a compact, MATLAB-style interface for neural networks.
"""

from .core.model_type import ModelType
from .neural_network import NeuralNetwork
from .config import NeuralNetworkConfig
from .core.results import TrainingResult, EvaluationResult, LayerWeights, ModelSummary
from .core.registry import register_default_models

# Registrar modelos por defecto al importar
register_default_models()

__all__ = [
    'ModelType',
    'NeuralNetwork',
    'NeuralNetworkConfig',
    'TrainingResult',
    'EvaluationResult',
    'LayerWeights',
    'ModelSummary'
]
