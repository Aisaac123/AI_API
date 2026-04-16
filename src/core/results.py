"""
Dataclasses para resultados de entrenamiento y evaluación.
Estos proporcionan contenedores estructurados para resultados de entrenadores y evaluadores.
"""

from dataclasses import dataclass, field
from typing import Any
import time
import numpy as np


@dataclass
class TrainingResult:
    """
    Resultado de entrenamiento de un modelo.
    
    Contiene métricas y metadatos del proceso de entrenamiento.
    """
    error_history: list[float] = field(default_factory=list)
    """Historial de errores a través de épocas"""
    
    epochs: int = 0
    """Número de épocas de entrenamiento"""
    
    training_time: float = 0.0
    """Tiempo de entrenamiento en segundos"""
    
    final_error: float = 0.0
    """Error final después del entrenamiento"""
    
    converged: bool = True
    """Si el entrenamiento convergió"""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Metadatos adicionales sobre el entrenamiento"""

    def __str__(self) -> str:
        """Retornar un resumen legible por humanos del resultado de entrenamiento."""
        return (
            f"TrainingResult(epochs={self.epochs}, "
            f"final_error={self.final_error:.6f}, "
            f"time={self.training_time:.2f}s, "
            f"converged={self.converged})"
        )


@dataclass
class EvaluationReport:
    """
    Reporte de evaluación de un modelo.
    
    Contiene métricas de rendimiento en datos de prueba.
    """
    mse: float = 0.0
    """Error Cuadrático Medio: MSE = (1/n) * sum((y_true - y_pred)^2)"""
    
    mae: float = 0.0
    """Error Absoluto Medio: MAE = (1/n) * sum(|y_true - y_pred|)"""
    
    rmse: float = 0.0
    """Raíz del Error Cuadrático Medio: RMSE = sqrt(MSE)"""
    
    r2: float = 0.0
    """Coeficiente de determinación R2: R2 = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))"""
    
    accuracy: float = 0.0
    """Precisión (para clasificación): Accuracy = (correctas / total)"""
    
    predictions: np.ndarray = None
    """Predicciones del modelo"""
    
    metadata: dict = None
    """Metadatos adicionales sobre la evaluación"""

    def __str__(self) -> str:
        """Retornar un resumen legible por humanos del reporte de evaluación."""
        return (
            f"EvaluationReport(MSE={self.mse:.6f}, "
            f"MAE={self.mae:.6f}, "
            f"RMSE={self.rmse:.6f}, "
            f"R2={self.r2:.6f}, "
            f"Accuracy={self.accuracy:.4f})"
        )
