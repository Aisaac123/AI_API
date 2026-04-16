"""
Clases base abstractas (interfaces) para el proyecto de red RBF.
Estas interfaces definen los contratos que todas las implementaciones concretas deben seguir.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseModel(ABC):
    """
    Clase base abstracta para todos los modelos.
    Cualquier modelo (RBF, backprop, etc.) debe implementar estos métodos.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrenar el modelo con datos de entrada X y salidas objetivo y.
        
        El entrenamiento minimiza una función de pérdida L(theta) donde theta son los parámetros del modelo.
        
        Fórmula: theta* = argmin_theta L(theta, X, y)
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            y: Matriz de salida objetivo de forma (n_samples, n_outputs)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Hacer predicciones para datos de entrada X.
        
        Calcula la salida del modelo dado los parámetros entrenados.
        
        Fórmula: y_pred = f(X, theta)
        
        Donde:
        - f: función del modelo
        - theta: parámetros entrenados
        - y_pred: predicciones
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            
        Returns:
            Matriz de predicciones de forma (n_samples, n_outputs)
        """
        pass

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        """
        Retornar un diccionario con la configuración y estado del modelo.
        
        Returns:
            Diccionario que contiene información del modelo
        """
        pass


class BaseLayer(ABC):
    """
    Clase base abstracta para todas las capas.
    Cualquier capa debe implementar el pase hacia adelante (forward pass).
    """

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calcular el pase hacia adelante de la capa.
        
        Fórmula: Y = activation(X @ W + b)
        
        Donde:
        - X: matriz de entrada
        - W: matriz de pesos
        - b: vector de bias
        - activation: función de activación no lineal
        
        Args:
            X: Matriz de entrada de forma (n_samples, input_size)
            
        Returns:
            Matriz de salida de forma (n_samples, output_size)
        """
        pass


class BaseTrainer(ABC):
    """
    Clase base abstracta para todos los entrenadores.
    Cualquier entrenador debe implementar el método train siguiendo una interfaz consistente.
    """

    @abstractmethod
    def train(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Entrenar un modelo con datos de entrada X y salidas objetivo y.
        
        El objetivo es minimizar la función de pérdida: L(θ) = Σ(ŷᵢ - yᵢ)²
        
        Args:
            model: La instancia del modelo a entrenar
            X: Matriz de entrada de forma (n_samples, n_features)
            y: Matriz de salida objetivo de forma (n_samples, n_outputs)
            
        Returns:
            Resultado de entrenamiento que contiene historial y métricas
        """
        pass


class BaseCenterInitializer(ABC):
    """
    Clase base abstracta para estrategias de inicialización de centros.
    Cualquier estrategia debe implementar el método initialize.
    """

    @abstractmethod
    def initialize(self, X: np.ndarray, n_centers: int) -> np.ndarray:
        """
        Inicializar centros para red RBF.
        
        Fórmula general: C = init(X, k)
        
        Donde:
        - C: matriz de centros
        - X: datos de entrenamiento
        - k: número de centros
        
        Args:
            X: Matriz de datos de entrenamiento de forma (n_samples, n_features)
            n_centers: Número de centros a inicializar
            
        Returns:
            Matriz de centros de forma (n_centers, n_features)
        """
        pass
