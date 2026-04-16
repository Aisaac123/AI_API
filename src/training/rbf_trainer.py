"""
Implementación de entrenador RBF.
Este entrenador maneja el proceso de entrenamiento para redes RBF usando inicialización de centros.
"""

import numpy as np
import time
from src.core.interfaces import BaseTrainer, BaseCenterInitializer
from src.core.results import TrainingResult
from .initializer import RandomInitializer


class RBFTrainer(BaseTrainer):
    """
    Entrenador para redes RBF.
    
    Este entrenador maneja el proceso de entrenamiento para redes RBF, incluyendo:
    - Inicialización de centros usando una estrategia especificada
    - Entrenamiento del modelo con los centros inicializados
    - Rastreo de métricas de entrenamiento y resultados
    
    El entrenamiento RBF usa solución de forma cerrada: W = pinv(Phi) @ y
    donde Phi es la matriz de diseño de activaciones RBF.
    """
    
    def __init__(self, initializer: BaseCenterInitializer = None):
        """
        Inicializar el entrenador RBF.
        
        Args:
            initializer: Estrategia de inicialización de centros (por defecto RandomInitializer)
        """
        if initializer is None:
            initializer = RandomInitializer()
        
        self.initializer = initializer
    
    def train(self, model, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """
        Entrenar un modelo RBF con datos de entrada X y salidas objetivo y.
        
        El entrenamiento resuelve: W = pinv(Phi) @ y
        donde Phi = phi(d(X, C), sigma) es la matriz de diseño.
        
        Args:
            model: Instancia de RBFNetwork a entrenar
            X: Matriz de entrada de forma (n_samples, n_features)
            y: Matriz de salida objetivo de forma (n_samples, n_outputs)
            
        Returns:
            TrainingResult conteniendo historial de entrenamiento y métricas
        """
        start_time = time.time()
        
        # Validar entrada
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Inicializar centros usando la estrategia especificada
        n_centers = model.config.n_centers
        centers = self.initializer.initialize(X, n_centers)
        
        # Entrenar el modelo con los centros inicializados
        model.fit(X, y, centers=centers)
        
        # Calcular error final: MSE = (1/n) * sum((y_pred_i - y_i)^2)
        predictions = model.predict(X)
        final_error = np.mean((predictions - y) ** 2)
        
        training_time = time.time() - start_time
        
        # Crear resultado de entrenamiento
        result = TrainingResult(
            error_history=[final_error],
            epochs=1,
            training_time=training_time,
            final_error=final_error,
            converged=True,
            metadata={
                'initializer': str(type(self.initializer).__name__),
                'n_centers': n_centers
            }
        )
        
        return result
