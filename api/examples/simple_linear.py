"""
Ejemplo de cómo agregar un nuevo modelo al sistema de registro.

Este ejemplo muestra cómo agregar un modelo lineal simple (regresión lineal)
al sistema de registro dinámico sin modificar el código central de la API.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from src.training.base_trainer import BaseTrainer
from src.training.training_result import TrainingResult
from src.models.base_model import BaseModel
from src.models.config import BaseConfig
from api.base import ModelFactory
from api.registry import ModelRegistry


class LinearConfig(BaseConfig):
    """Configuración para el modelo lineal."""
    
    def __init__(
        self,
        use_bias: bool = True,
        random_state: int = 42
    ):
        self.use_bias = use_bias
        self.random_state = random_state
    
    def validate(self) -> None:
        """Validar configuración."""
        if not isinstance(self.use_bias, bool):
            raise ValueError("use_bias debe ser booleano")


class LinearNetwork(BaseModel):
    """
    Modelo de regresión lineal simple.
    
    y = X @ W + b
    """
    
    def __init__(self, config: LinearConfig):
        super().__init__(config)
        self.weights = None
        self.bias = None
        self.use_bias = config.use_bias
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """
        Entrenar usando solución de mínimos cuadrados.
        
        W = (X.T @ X)^(-1) @ X.T @ y
        """
        n_samples, n_features = X.shape
        
        if self.use_bias:
            # Agregar columna de unos para el bias
            X_augmented = np.column_stack([X, np.ones(n_samples)])
            weights_with_bias = np.linalg.pinv(X_augmented.T @ X_augmented) @ X_augmented.T @ y
            self.weights = weights_with_bias[:-1]
            self.bias = weights_with_bias[-1]
        else:
            self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y
            self.bias = 0
        
        # Calcular error
        predictions = self.predict(X)
        mse = np.mean((y - predictions) ** 2)
        
        return TrainingResult(
            final_error=mse,
            epochs=1,
            error_history=[mse],
            converged=True,
            metadata={'method': 'least_squares'}
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predecir con el modelo lineal."""
        predictions = X @ self.weights
        if self.use_bias:
            predictions += self.bias
        return predictions
    
    def summary(self) -> dict:
        """Obtener resumen del modelo."""
        n_params = self.weights.shape[0] + (1 if self.use_bias else 0)
        return {
            'model_type': 'linear',
            'architecture': {
                'input_size': self.weights.shape[0],
                'output_size': 1,
                'use_bias': self.use_bias
            },
            'n_parameters': n_params
        }


class LinearTrainer(BaseTrainer):
    """Entrenador para el modelo lineal."""
    
    def __init__(self):
        super().__init__()
    
    def train(self, model: LinearNetwork, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """Entrenar el modelo lineal."""
        return model.fit(X, y)


class LinearModelFactory(ModelFactory):
    """Factory para el modelo lineal."""
    
    def create_network(self, X: np.ndarray, y: np.ndarray, config):
        """Crear red lineal."""
        return LinearNetwork(config)
    
    def create_trainer(self, config):
        """Crear entrenador lineal."""
        return LinearTrainer()
    
    def get_config_class(self):
        """Obtener clase de configuración."""
        return LinearConfig


# Registrar el modelo lineal en el sistema
ModelRegistry.register('linear', LinearModelFactory())


# Ejemplo de uso
if __name__ == '__main__':
    print("=== Ejemplo de uso del modelo lineal registrado ===")
    
    # Verificar que está registrado
    print(f"Modelos registrados: {ModelRegistry.list_models()}")
    print(f"¿Linear está registrado? {ModelRegistry.is_registered('linear')}")
    
    # Crear datos de prueba
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1 * X[:, 2] + 1 + 0.1 * np.random.randn(100)
    
    # Crear y entrenar el modelo lineal
    from api import NeuralNetwork, ModelType
    
    # Nota: Para usar el nuevo modelo, necesitaríamos agregar 'LINEAR' al enum ModelType
    # o usar el sistema directamente
    factory = ModelRegistry.get_factory('linear')
    config = LinearConfig(use_bias=True, random_state=42)
    model = factory.create_network(X, y, config)
    trainer = factory.create_trainer(config)
    
    result = trainer.train(model, X, y)
    print(f"\nEntrenamiento completado:")
    print(f"Error final: {result.final_error:.6f}")
    print(f"Pesos: {model.weights}")
    print(f"Bias: {model.bias}")
    
    predictions = model.predict(X[:5])
    print(f"\nPrimeras 5 predicciones: {predictions}")
    print(f"Valores reales: {y[:5]}")
    
    print("\n✅ Modelo lineal agregado exitosamente al sistema de registro!")
