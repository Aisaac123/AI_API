"""
Resultados tipados para la API de redes neuronales.
Usa dataclasses para proporcionar tipado fuerte y autocompletado.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class TrainingResult:
    """
    Resultado del entrenamiento de la red neuronal.
    
    Proporciona acceso tipado a los resultados de entrenamiento
    en lugar de un diccionario genérico.
    """
    training_time: float
    final_error: float
    epochs: int
    error_history: List[float]
    converged: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para compatibilidad con código existente."""
        return {
            'training_time': self.training_time,
            'final_error': self.final_error,
            'epochs': self.epochs,
            'error_history': self.error_history,
            'converged': self.converged,
            'metadata': self.metadata
        }


@dataclass
class EvaluationResult:
    """
    Resultado de la evaluación del modelo.
    
    Proporciona acceso tipado a las métricas de evaluación.
    """
    mse: float
    mae: float
    rmse: float
    r2: float
    accuracy: float
    predictions: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para compatibilidad con código existente."""
        result = {
            'mse': self.mse,
            'mae': self.mae,
            'rmse': self.rmse,
            'r2': self.r2,
            'accuracy': self.accuracy
        }
        if self.predictions is not None:
            result['predictions'] = self.predictions
        if self.metadata:
            result['metadata'] = self.metadata
        return result


@dataclass
class LayerWeights:
    """
    Pesos y bias de una capa específica.
    
    Proporciona acceso tipado a los pesos de una capa.
    """
    layer_index: int
    layer_type: str  # 'hidden' o 'output'
    input_size: int
    output_size: int
    weights: np.ndarray
    bias: Optional[np.ndarray]
    activation: str
    use_bias: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para compatibilidad con código existente."""
        return {
            'layer_index': self.layer_index,
            'layer_type': self.layer_type,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'weights': self.weights,
            'bias': self.bias,
            'activation': self.activation,
            'use_bias': self.use_bias
        }


@dataclass
class ModelSummary:
    """
    Resumen completo del modelo.
    
    Proporciona acceso tipado a la información del modelo.
    """
    model_type: str
    is_fitted: bool
    configuration: Dict[str, Any]
    architecture: Optional[Dict[str, Any]] = None
    n_parameters: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para compatibilidad con código existente."""
        result = {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'configuration': self.configuration
        }
        if self.architecture is not None:
            result['architecture'] = self.architecture
        if self.n_parameters is not None:
            result['n_parameters'] = self.n_parameters
        return result


@dataclass
class ConfusionMatrixResult:
    """
    Resultado de cálculo de matriz de confusión con métricas derivadas.
    
    Este dataclass encapsula la matriz de confusión y todas las métricas
    derivadas como precision, recall, F1-score, etc.
    """
    matrix: np.ndarray
    """Matriz de confusión con valores absolutos (n_classes, n_classes)"""
    
    matrix_normalized_row: np.ndarray
    """Matriz normalizada por fila (recall por clase)"""
    
    matrix_normalized_col: np.ndarray
    """Matriz normalizada por columna (precision por clase)"""
    
    precision: Dict[str, float]
    """Precision por clase: TP / (TP + FP)"""
    
    recall: Dict[str, float]
    """Recall por clase: TP / (TP + FN)"""
    
    f1_score: Dict[str, float]
    """F1-score por clase: 2 * (precision * recall) / (precision + recall)"""
    
    support: Dict[str, int]
    """Número de muestras reales por clase"""
    
    accuracy: float
    """Accuracy global: (TP + TN) / total"""
    
    macro_avg: Dict[str, float]
    """Promedio macro de precision, recall, f1"""
    
    weighted_avg: Dict[str, float]
    """Promedio ponderado por support de precision, recall, f1"""
    
    n_classes: int
    """Número de clases"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para compatibilidad con código existente."""
        return {
            'matrix': self.matrix,
            'matrix_normalized_row': self.matrix_normalized_row,
            'matrix_normalized_col': self.matrix_normalized_col,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'support': self.support,
            'accuracy': self.accuracy,
            'macro_avg': self.macro_avg,
            'weighted_avg': self.weighted_avg,
            'n_classes': self.n_classes
        }
