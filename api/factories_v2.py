"""
Factories que implementan la interfaz ModelFactory para el sistema de registro.
"""

import numpy as np
from typing import Any
from src.models import RBFNetwork, RBFConfig, BackpropNetwork, BackpropConfig
from src.training import RBFTrainer, BackpropTrainer, KMeansInitializer, RandomInitializer
from src.core.activation import (
    GaussianActivation,
    MultiquadraticActivation,
    InverseMultiquadraticActivation,
    ThinPlateSplineActivation
)
from .config import NeuralNetworkConfig
from .core.base import ModelFactory as BaseModelFactory


class RBFModelFactory(BaseModelFactory):
    """Factory para modelos RBF."""
    
    def create_network(self, X: np.ndarray, y: np.ndarray, config: NeuralNetworkConfig) -> RBFNetwork:
        """Crear red RBF configurada."""
        # Determinar número de centros
        n_centers = config.n_centers
        if n_centers is None:
            n_centers = min(20, X.shape[0])
        else:
            n_centers = min(n_centers, X.shape[0])
        
        # Obtener función de activación
        activation = self._create_rbf_activation(config.activation_rbf)
        
        # Crear configuración
        rbf_config = RBFConfig(
            n_centers=n_centers,
            sigma=config.sigma,
            activation=activation,
            regularization=config.regularization,
            use_bias=config.use_bias,
            random_state=config.random_state
        )
        
        return RBFNetwork(config=rbf_config)
    
    def create_trainer(self, config: NeuralNetworkConfig) -> RBFTrainer:
        """Crear entrenador RBF configurado."""
        # Crear inicializador apropiado
        if config.initializer == 'kmeans':
            init = KMeansInitializer(max_iterations=50)
        else:
            init = RandomInitializer()
        
        return RBFTrainer(initializer=init)
    
    def get_config_class(self) -> type:
        """Obtener la clase de configuración del modelo."""
        return RBFConfig
    
    def _create_rbf_activation(self, activation_name: str):
        """Crear función de activación RBF desde string."""
        activations = {
            'gaussian': GaussianActivation,
            'multiquadratic': MultiquadraticActivation,
            'inverse_multiquadratic': InverseMultiquadraticActivation,
            'thin_plate': ThinPlateSplineActivation
        }
        activation_class = activations.get(
            activation_name.lower(),
            GaussianActivation
        )
        return activation_class()


class BackpropModelFactory(BaseModelFactory):
    """Factory para modelos de retropropagación."""
    
    def create_network(self, X: np.ndarray, y: np.ndarray, config: NeuralNetworkConfig) -> BackpropNetwork:
        """Crear red de retropropagación configurada."""
        backprop_config = BackpropConfig(
            hidden_layers=config.hidden_layers,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            batch_size=config.batch_size,
            activation=config.activation_backprop,
            layer_activations=config.layer_activations,
            output_activation=config.output_activation,
            use_bias=config.use_bias,
            random_state=config.random_state
        )
        
        return BackpropNetwork(config=backprop_config)
    
    def create_trainer(self, config: NeuralNetworkConfig) -> BackpropTrainer:
        """Crear entrenador de retropropagación configurado."""
        return BackpropTrainer(verbose=False)
    
    def get_config_class(self) -> type:
        """Obtener la clase de configuración del modelo."""
        return BackpropConfig
