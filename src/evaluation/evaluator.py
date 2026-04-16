"""
Clase Evaluator para evaluación de modelo.
Esta clase proporciona una interfaz unificada para evaluar cualquier modelo que implementa BaseModel.
"""

import numpy as np
from src.core.interfaces import BaseModel
from src.core.results import EvaluationReport
from .metrics import mse, mae, rmse, r2_score, accuracy


class Evaluator:
    """
    Evaluador para rendimiento de modelo.
    
    Esta clase evalúa cualquier modelo que implementa la interfaz BaseModel
    y produce un reporte de evaluación comprensivo con varias métricas.
    """
    
    def __init__(self):
        """Inicializar el evaluador."""
        pass
    
    def evaluate(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> EvaluationReport:
        """
        Evaluar un modelo en datos de prueba.
        
        Args:
            model: Instancia de modelo que implementa BaseModel
            X: Matriz de entrada de prueba de forma (n_samples, n_features)
            y: Valores objetivo de prueba de forma (n_samples, n_outputs)
            
        Returns:
            EvaluationReport conteniendo varias métricas y predicciones
        """
        # Hacer predicciones
        predictions = model.predict(X)
        
        # Calcular métricas
        mse_value = mse(y, predictions)
        mae_value = mae(y, predictions)
        rmse_value = rmse(y, predictions)
        r2_value = r2_score(y, predictions)
        accuracy_value = accuracy(y, predictions)
        
        # Crear reporte de evaluación
        report = EvaluationReport(
            mse=mse_value,
            mae=mae_value,
            rmse=rmse_value,
            r2=r2_value,
            accuracy=accuracy_value,
            predictions=predictions,
            metadata={
                'model_type': type(model).__name__,
                'n_samples': X.shape[0]
            }
        )
        
        return report
    
    def evaluate_with_training(self, model: BaseModel, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluar un modelo en datos de entrenamiento y prueba.
        
        Esto es útil para verificar sobreajuste comparando rendimiento de entrenamiento y prueba.
        
        Args:
            model: Instancia de modelo que implementa BaseModel
            X_train: Matriz de entrada de entrenamiento
            y_train: Valores objetivo de entrenamiento
            X_test: Matriz de entrada de prueba
            y_test: Valores objetivo de prueba
            
        Returns:
            Diccionario con reportes de evaluación de entrenamiento y prueba
        """
        train_report = self.evaluate(model, X_train, y_train)
        test_report = self.evaluate(model, X_test, y_test)
        
        return {
            'train': train_report,
            'test': test_report,
            'overfitting_ratio': test_report.mse / train_report.mse if train_report.mse > 0 else float('inf')
        }
