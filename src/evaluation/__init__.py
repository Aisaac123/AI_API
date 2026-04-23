"""
Evaluation module exports.
This module exposes evaluation metrics, the evaluator class, and confusion matrix functionality.
"""

from .metrics import mse, mae, rmse, r2_score, accuracy
from .evaluator import Evaluator
from .confusion_matrix import ConfusionMatrixCalculator, ConfusionMatrixResult

__all__ = [
    'mse',
    'mae',
    'rmse',
    'r2_score',
    'accuracy',
    'Evaluator',
    'ConfusionMatrixCalculator',
    'ConfusionMatrixResult'
]
