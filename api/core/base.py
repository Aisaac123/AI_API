"""
Interfaces base para el sistema de registro de modelos.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class ModelFactory(ABC):
    """Interfaz abstracta para factories de modelos."""
    
    @abstractmethod
    def create_network(self, X: np.ndarray, y: np.ndarray, config: Any):
        """Crear una instancia del modelo."""
        pass
    
    @abstractmethod
    def create_trainer(self, config: Any):
        """Crear una instancia del entrenador."""
        pass
    
    @abstractmethod
    def get_config_class(self) -> type:
        """Obtener la clase de configuración del modelo."""
        pass
