"""
Dataclass de configuración para red RBF.
Proporciona una forma tipada y declarativa de configurar parámetros de red RBF.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
from src.core.activation import ActivationFunction, GaussianActivation


@dataclass
class RBFConfig:
    """
    Clase de configuración para parámetros de red RBF.
    
    Esta dataclass encapsula todos los hiperparámetros necesarios para construir
    y entrenar una red RBF, haciendo la configuración explícita y reproducible.
    """
    n_centers: int = 10
    """Número de centros de función de base radial (neuronas en capa oculta)"""
    
    sigma: float = 1.0
    """Parámetro de ancho para la función de activación (dispersión de cada RBF)"""
    
    activation: ActivationFunction = None
    """Función de activación a usar (por defecto Gaussiana si no se especifica)"""
    
    regularization: float = 0.0
    """Parámetro de regularización para la pseudoinversa (agrega matriz identidad * λ)"""
    
    use_bias: bool = True
    """Si incluir término de bias en la capa de salida"""
    
    random_state: int = None
    """Semilla aleatoria para reproducibilidad"""
    
    def __post_init__(self):
        """Establecer función de activación por defecto a Gaussiana si no se especifica."""
        if self.activation is None:
            self.activation = GaussianActivation()
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def validate(self) -> None:
        """
        Validar los parámetros de configuración.
        
        Raises:
            InvalidConfigError: Si algún parámetro es inválido
        """
        from src.core.exceptions import InvalidConfigError
        
        if self.n_centers <= 0:
            raise InvalidConfigError(f"n_centers debe ser positivo, se obtuvo {self.n_centers}")
        
        if self.sigma <= 0:
            raise InvalidConfigError(f"sigma debe ser positivo, se obtuvo {self.sigma}")
        
        if self.regularization < 0:
            raise InvalidConfigError(f"regularization debe ser no negativo, se obtuvo {self.regularization}")
        
        if not isinstance(self.activation, ActivationFunction):
            raise InvalidConfigError(f"activation debe ser una instancia de ActivationFunction")
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convertir configuración a diccionario.
        
        Returns:
            Representación en diccionario de la configuración
        """
        return {
            'n_centers': self.n_centers,
            'sigma': self.sigma,
            'activation': str(self.activation),
            'regularization': self.regularization,
            'use_bias': self.use_bias,
            'random_state': self.random_state
        }
