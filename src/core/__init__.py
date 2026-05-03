"""
Core module exports.
This module exposes the public interfaces and types used throughout the project.
"""

# Abstract base classes
from .interfaces import BaseModel, BaseLayer, BaseTrainer, BaseCenterInitializer

# Activation functions
from .activation import (
    ActivationFunction,
    GaussianActivation,
    MultiquadraticActivation,
    InverseMultiquadraticActivation,
    ThinPlateSplineActivation,
    ThinPlateSplineLog10Activation
)

# Distance functions
from .distance import euclidean_distance, euclidean_distance_matrix, euclidean_distance_squared_matrix

# Result dataclasses
from .results import TrainingResult, EvaluationReport

# Exceptions
from .exceptions import (
    RBFNetworkError,
    NotFittedError,
    InvalidConfigError,
    InvalidInputError,
    ConvergenceError
)

__all__ = [
    # Interfaces
    'BaseModel',
    'BaseLayer',
    'BaseTrainer',
    'BaseCenterInitializer',
    
    # Activation functions
    'ActivationFunction',
    'GaussianActivation',
    'MultiquadraticActivation',
    'InverseMultiquadraticActivation',
    'ThinPlateSplineActivation',
    'ThinPlateSplineLog10Activation',
    
    # Distance functions
    'euclidean_distance',
    'euclidean_distance_matrix',
    'euclidean_distance_squared_matrix',
    
    # Results
    'TrainingResult',
    'EvaluationReport',
    
    # Exceptions
    'RBFNetworkError',
    'NotFittedError',
    'InvalidConfigError',
    'InvalidInputError',
    'ConvergenceError'
]
