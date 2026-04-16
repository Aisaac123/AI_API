"""
Fábricas para crear modelos y entrenadores.
Aplicación del patrón Factory para separar la creación de objetos.
"""

from typing import Optional
import numpy as np
from src.models import RBFNetwork, RBFConfig, BackpropNetwork, BackpropConfig
from src.training import RBFTrainer, BackpropTrainer, KMeansInitializer, RandomInitializer
from src.core.activation import (
    GaussianActivation,
    MultiquadraticActivation,
    InverseMultiquadraticActivation,
    ThinPlateSplineActivation
)
from .config import NeuralNetworkConfig
from .model_type import ModelType


class ModelFactory:
    """
    Fábrica para crear modelos de redes neuronales.
    
    Aplica el patrón Factory para separar la lógica de creación de modelos
    de la clase principal de la API.
    """
    
    @staticmethod
    def create_rbf_network(
        X: np.ndarray,
        config: NeuralNetworkConfig
    ) -> RBFNetwork:
        """
        Crear red RBF configurada.
        
        Args:
            X: Datos de entrada para determinar dimensiones
            config: Configuración de la red
            
        Returns:
            Instancia de RBFNetwork configurada
        """
        # Determinar número de centros
        n_centers = config.n_centers
        if n_centers is None:
            n_centers = min(20, X.shape[0])
        else:
            n_centers = min(n_centers, X.shape[0])
        
        # Obtener función de activación
        activation = ActivationFactory.create_rbf_activation(config.activation_rbf)
        
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
    
    @staticmethod
    def create_backprop_network(
        X: np.ndarray,
        y: np.ndarray,
        config: NeuralNetworkConfig
    ) -> BackpropNetwork:
        """
        Crear red de retropropagación configurada.
        
        Args:
            X: Datos de entrada
            y: Datos de salida
            config: Configuración de la red
            
        Returns:
            Instancia de BackpropNetwork configurada
        """
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


class TrainerFactory:
    """
    Fábrica para crear entrenadores.
    
    Aplica el patrón Factory para separar la lógica de creación de entrenadores.
    """
    
    @staticmethod
    def create_rbf_trainer(config: NeuralNetworkConfig) -> RBFTrainer:
        """
        Crear entrenador RBF configurado.
        
        Args:
            config: Configuración de la red
            
        Returns:
            Instancia de RBFTrainer configurada
        """
        # Crear inicializador apropiado
        if config.initializer == 'kmeans':
            init = KMeansInitializer(max_iterations=50)
        else:
            init = RandomInitializer()
        
        return RBFTrainer(initializer=init)
    
    @staticmethod
    def create_backprop_trainer(config: NeuralNetworkConfig) -> BackpropTrainer:
        """
        Crear entrenador de retropropagación configurado.
        
        Args:
            config: Configuración de la red
            
        Returns:
            Instancia de BackpropTrainer configurada
        """
        return BackpropTrainer(verbose=False)


class ActivationFactory:
    """
    Fábrica para crear funciones de activación.
    
    Aplica el patrón Factory para la creación de funciones de activación.
    """
    
    _rbf_activations = {
        'gaussian': GaussianActivation,
        'multiquadratic': MultiquadraticActivation,
        'inverse_multiquadratic': InverseMultiquadraticActivation,
        'thin_plate': ThinPlateSplineActivation
    }
    
    @classmethod
    def create_rbf_activation(cls, activation_name: str):
        """
        Crear función de activación RBF desde string.
        
        Args:
            activation_name: Nombre de la función de activación
            
        Returns:
            Instancia de la función de activación
        """
        activation_class = cls._rbf_activations.get(
            activation_name.lower(),
            GaussianActivation
        )
        return activation_class()
