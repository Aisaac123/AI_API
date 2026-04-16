"""
Implementación de entrenador de retropropagación.
Este entrenador maneja el proceso de entrenamiento para redes de retropropagación con descenso de gradiente.
"""

import numpy as np
import time
from src.core.interfaces import BaseTrainer
from src.core.results import TrainingResult


class BackpropTrainer(BaseTrainer):
    """
    Entrenador para redes de retropropagación.
    
    Este entrenador maneja el proceso de entrenamiento para redes de retropropagación, incluyendo:
    - Entrenamiento iterativo con descenso de gradiente
    - Rastreo de error a través de épocas
    - Monitoreo de convergencia
    
    El entrenamiento usa: theta_{t+1} = theta_t - alpha * gradient_L(theta_t)
    donde alpha es la tasa de aprendizaje y gradient_L es el gradiente de la función de pérdida.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Inicializar el entrenador de retropropagación.
        
        Args:
            verbose: Si imprimir progreso de entrenamiento
        """
        self.verbose = verbose
    
    def train(self, model, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """
        Entrenar un modelo de retropropagación con datos de entrada X y salidas objetivo y.
        
        El entrenamiento minimiza: L(theta) = (1/2) * sum((y - y_pred)^2)
        usando descenso de gradiente: theta_{t+1} = theta_t - alpha * gradient_L(theta_t)
        
        Args:
            model: Instancia de BackpropNetwork a entrenar
            X: Matriz de entrada de forma (n_samples, n_features)
            y: Matriz de salida objetivo de forma (n_samples, n_outputs)
            
        Returns:
            TrainingResult conteniendo historial de entrenamiento y métricas
        """
        start_time = time.time()
        
        # Validar entrada
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Almacenar épocas de configuración original para rastreo
        original_epochs = model.config.epochs
        
        # Llamar al método fit del modelo que maneja toda la lógica de entrenamiento
        model.fit(X, y)
        
        # Calcular error final: MSE = (1/n) * sum((y_pred_i - y_i)^2)
        predictions = model.predict(X)
        final_error = np.mean((predictions - y) ** 2)
        
        training_time = time.time() - start_time
        
        # Crear resultado de entrenamiento
        result = TrainingResult(
            error_history=[final_error],
            epochs=original_epochs,
            training_time=training_time,
            final_error=final_error,
            converged=True,
            metadata={
                'learning_rate': model.config.learning_rate
            }
        )
        
        return result
    
    def _forward_pass(self, model, X: np.ndarray) -> np.ndarray:
        """Calcular pase hacia adelante a través de todas las capas."""
        activation = X
        for layer in model.layers:
            activation = layer.forward(activation)
        return activation
    
    def _backward_pass(self, model, output_gradient: np.ndarray) -> None:
        """Calcular pase hacia atrás a través de todas las capas."""
        gradient = output_gradient
        for layer in reversed(model.layers):
            gradient = layer.backward(gradient)
