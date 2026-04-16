"""
Clase de configuración para la API de redes neuronales.
Proporciona una forma estructurada de configurar parámetros de la red neuronal.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class NeuralNetworkConfig:
    """
    Clase de configuración para la API de redes neuronales.
    
    Esta dataclass encapsula todos los hiperparámetros necesarios para crear
    y entrenar redes neuronales, haciendo la configuración explícita y reproducible.
    """
    hidden_layers: List[int] = None
    """Lista de tamaños de capas ocultas (para backpropagation)"""
    
    n_centers: Optional[int] = None
    """Número de centros RBF (para RBF)"""
    
    sigma: float = 1.0
    """Parámetro de ancho de RBF"""
    
    activation_rbf: str = 'gaussian'
    """Función de activación RBF: 'gaussian', 'multiquadratic', 'inverse_multiquadratic', 'thin_plate'"""
    
    activation_backprop: str = 'sigmoid'
    """Función de activación backpropagation por defecto: 'sigmoid', 'logsig', 'tanh', 'tansig', 'relu', 'linear', 'purelin', 'leaky_relu'"""
    
    # Funciones de activación por capa (estilo MATLAB)
    layer_activations: Optional[List[str]] = None
    """Lista de funciones de activación por capa para backpropagation. Si es None, usa activation_backprop para todas las capas.
       Ejemplo: ['tansig', 'logsig', 'purelin'] para 3 capas"""
    
    output_activation: str = 'linear'
    """Función de activación de la capa de salida: 'sigmoid', 'logsig', 'tanh', 'tansig', 'relu', 'linear', 'purelin'"""
    
    learning_rate: float = 0.01
    """Tasa de aprendizaje para backpropagation"""
    
    epochs: int = 1000
    """Número de épocas de entrenamiento (para backpropagation)"""
    
    batch_size: int = 32
    """Tamaño de lote para mini-batch training"""
    
    use_bias: bool = True
    """Si usar términos de bias"""
    
    regularization: float = 0.01
    """Parámetro de regularización"""
    
    random_state: int = 42
    """Semilla aleatoria para reproducibilidad"""
    
    initializer: str = 'kmeans'
    """Estrategia de inicialización de centros: 'kmeans' o 'random'"""
    
    def __post_init__(self):
        """Validar y establecer valores por defecto."""
        if self.hidden_layers is None:
            self.hidden_layers = [10]
        
        if self.n_centers is None:
            self.n_centers = 20
    
    def validate(self) -> None:
        """
        Validar los parámetros de configuración.
        
        Raises:
            ValueError: Si algún parámetro es inválido
        """
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate debe ser positivo, se obtuvo {self.learning_rate}")
        
        if self.epochs <= 0:
            raise ValueError(f"epochs debe ser positivo, se obtuvo {self.epochs}")
        
        if self.batch_size != -1 and self.batch_size <= 0:
            raise ValueError(f"batch_size debe ser positivo o -1, se obtuvo {self.batch_size}")
        
        if self.sigma <= 0:
            raise ValueError(f"sigma debe ser positivo, se obtuvo {self.sigma}")
        
        if self.regularization < 0:
            raise ValueError(f"regularización debe ser no negativo, se obtuvo {self.regularization}")
        
        valid_activations = ['sigmoid', 'logsig', 'tanh', 'tansig', 'relu', 'linear', 'purelin', 'leaky_relu']
        if self.activation_backprop not in valid_activations:
            raise ValueError(
                f"activation_backprop debe ser uno de {valid_activations}, se obtuvo '{self.activation_backprop}'"
            )
        
        if self.layer_activations is not None:
            if len(self.layer_activations) != len(self.hidden_layers):
                raise ValueError(
                    f"layer_activations debe tener el mismo tamaño que hidden_layers. "
                    f"Se obtuvo {len(self.layer_activations)} para {len(self.hidden_layers)} capas"
                )
            for act in self.layer_activations:
                if act not in valid_activations:
                    raise ValueError(
                        f"Activación inválida en layer_activations: '{act}'. Debe ser uno de {valid_activations}"
                    )
        
        if self.output_activation not in valid_activations:
            raise ValueError(
                f"output_activation debe ser uno de {valid_activations}, se obtuvo '{self.output_activation}'"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir configuración a diccionario.
        
        Returns:
            Representación en diccionario de la configuración
        """
        return {
            'hidden_layers': self.hidden_layers,
            'n_centers': self.n_centers,
            'sigma': self.sigma,
            'activation_rbf': self.activation_rbf,
            'activation_backprop': self.activation_backprop,
            'layer_activations': self.layer_activations,
            'output_activation': self.output_activation,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'use_bias': self.use_bias,
            'regularization': self.regularization,
            'random_state': self.random_state,
            'initializer': self.initializer
        }
