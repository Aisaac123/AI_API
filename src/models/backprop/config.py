"""
Dataclass de configuración para red de retropropagación.
Proporciona una forma tipada y declarativa de configurar parámetros de red backprop.
"""

from dataclasses import dataclass
from typing import Any, List
import numpy as np


@dataclass
class BackpropConfig:
    """
    Clase de configuración para parámetros de red de retropropagación.
    
    Esta dataclass encapsula todos los hiperparámetros necesarios para construir
    y entrenar una red de retropropagación, haciendo la configuración explícita y reproducible.
    """
    hidden_layers: List[int] = None
    """Lista de tamaños de capas ocultas, ej. [10, 5] para dos capas ocultas con 10 y 5 neuronas"""
    
    learning_rate: float = 0.01
    """Tasa de aprendizaje para descenso de gradiente: theta_{t+1} = theta_t - alpha * gradient_L(theta_t)"""
    
    epochs: int = 1000
    """Número máximo de épocas de entrenamiento"""
    
    batch_size: int = 32
    """Tamaño de lote para descenso de gradiente mini-batch (usar -1 para lote completo)"""
    
    activation: str = 'sigmoid'
    """Función de activación por defecto: 'sigmoid', 'tanh', 'relu', 'linear', 'leaky_relu'"""
    
    # Funciones de activación por capa (estilo MATLAB)
    layer_activations: List[str] = None
    """Lista de funciones de activación por capa. Si es None, usa 'activation' para todas las capas.
       Ejemplo: ['tanh', 'sigmoid', 'linear'] para 3 capas"""
    
    output_activation: str = 'linear'
    """Función de activación de la capa de salida: 'sigmoid', 'tanh', 'relu', 'linear'"""
    
    use_bias: bool = True
    """Si incluir términos de bias en cada capa"""
    
    random_state: int = None
    """Semilla aleatoria para reproducibilidad"""
    
    def __post_init__(self):
        """Establecer capas ocultas por defecto si no se especifican."""
        if self.hidden_layers is None:
            self.hidden_layers = [10]
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def validate(self) -> None:
        """
        Validar los parámetros de configuración.
        
        Raises:
            InvalidConfigError: Si algún parámetro es inválido
        """
        from src.core.exceptions import InvalidConfigError
        
        if not self.hidden_layers or len(self.hidden_layers) == 0:
            raise InvalidConfigError("hidden_layers debe ser una lista no vacía")
        
        if any(size <= 0 for size in self.hidden_layers):
            raise InvalidConfigError(f"Todos los tamaños de capa oculta deben ser positivos, se obtuvo {self.hidden_layers}")
        
        if self.learning_rate <= 0:
            raise InvalidConfigError(f"learning_rate debe ser positivo, se obtuvo {self.learning_rate}")
        
        if self.epochs <= 0:
            raise InvalidConfigError(f"epochs debe ser positivo, se obtuvo {self.epochs}")
        
        if self.batch_size != -1 and self.batch_size <= 0:
            raise InvalidConfigError(f"batch_size debe ser positivo o -1, se obtuvo {self.batch_size}")
        
        valid_activations = ['sigmoid', 'tanh', 'relu', 'linear', 'leaky_relu', 
                           'logsig', 'tansig', 'purelin']  # Nombres estilo MATLAB
        if self.activation not in valid_activations:
            raise InvalidConfigError(
                f"activation debe ser uno de {valid_activations}, se obtuvo '{self.activation}'"
            )
        
        if self.layer_activations is not None:
            if len(self.layer_activations) != len(self.hidden_layers):
                raise InvalidConfigError(
                    f"layer_activations debe tener el mismo tamaño que hidden_layers. "
                    f"Se obtuvo {len(self.layer_activations)} para {len(self.hidden_layers)} capas"
                )
            for act in self.layer_activations:
                if act not in valid_activations:
                    raise InvalidConfigError(
                        f"Activación inválida en layer_activations: '{act}'. Debe ser uno de {valid_activations}"
                    )
        
        if self.output_activation not in valid_activations:
            raise InvalidConfigError(
                f"output_activation debe ser uno de {valid_activations}, se obtuvo '{self.output_activation}'"
            )
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convertir configuración a diccionario.
        
        Returns:
            Representación en diccionario de la configuración
        """
        return {
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'activation': self.activation,
            'layer_activations': self.layer_activations,
            'output_activation': self.output_activation,
            'use_bias': self.use_bias,
            'random_state': self.random_state
        }
