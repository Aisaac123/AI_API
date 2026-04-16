"""
Implementación de capa densa para red de retropropagación.
Esta capa soporta pase hacia adelante, pase hacia atrás, y actualización de pesos.
"""

import numpy as np
from src.core.interfaces import BaseLayer
from src.core.activation import (
    SigmoidActivation,
    TanhActivation,
    ReLUActivation,
    LinearActivation,
    LeakyReLUActivation
)


class DenseLayer(BaseLayer):
    """
    Capa densa (totalmente conectada) para redes de retropropagación.
    
    Esta capa computa: y = activation(X @ W + b)
    donde X es entrada, W son pesos, b es bias, y activation es una función no lineal.
    
    La capa almacena activaciones y gradientes para retropropagación.
    
    Fórmulas:
    - Pase hacia adelante: z = X @ W + b, y = phi(z)
    - Gradiente de pesos: gradient_W = X.T @ delta
    - Gradiente de bias: gradient_b = sum(delta)
    - Gradiente de entrada: gradient_X = delta @ W.T
    donde delta = gradient_y @ phi'(z)
    """
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'sigmoid', use_bias: bool = True):
        """
        Inicializar la capa densa.
        
        Args:
            input_size: Número de características de entrada
            output_size: Número de neuronas de salida
            activation: Función de activación (estilo MATLAB): 'sigmoid', 'logsig', 'tanh', 'tansig', 'relu', 'linear', 'purelin', 'leaky_relu'
            use_bias: Si incluir término de bias
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.use_bias = use_bias
        
        # Mapear nombres estilo MATLAB a clases de activación
        self.activation = self._get_activation_function()
        
        # Inicializar pesos con inicialización Xavier/Glorot
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        
        # Inicializar bias a ceros
        if self.use_bias:
            self.bias = np.zeros(output_size)
        else:
            self.bias = None
        
        # Almacenar para retropropagación
        self.last_input = None
        self.last_output = None
        self.last_z = None
    
    def _get_activation_function(self):
        """Obtener la función de activación correspondiente al nombre."""
        activation_map = {
            'sigmoid': SigmoidActivation(),
            'logsig': SigmoidActivation(),  # Nombre estilo MATLAB
            'tanh': TanhActivation(),
            'tansig': TanhActivation(),  # Nombre estilo MATLAB
            'relu': ReLUActivation(),
            'linear': LinearActivation(),
            'purelin': LinearActivation(),  # Nombre estilo MATLAB
            'leaky_relu': LeakyReLUActivation()
        }
        
        if self.activation_name not in activation_map:
            raise ValueError(
                f"Activación desconocida: {self.activation_name}. "
                f"Opciones válidas: {list(activation_map.keys())}"
            )
        
        return activation_map[self.activation_name]
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calcular el pase hacia adelante de la capa.
        
        Fórmula: z = X @ W + b, y = phi(z)
        
        Args:
            X: Matriz de entrada de forma (n_samples, input_size)
            
        Returns:
            Matriz de salida de forma (n_samples, output_size)
        """
        self.last_input = X
        
        z = X @ self.weights
        if self.use_bias:
            z += self.bias
        self.last_z = z
        
        output = self.activation.compute(z)
        self.last_output = output
        
        return output
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Calcular el pase hacia atrás (computación de gradiente).
        
        Fórmulas:
        - delta = gradient_y @ phi'(z)
        - gradient_W = X.T @ delta
        - gradient_b = sum(delta)
        - gradient_X = delta @ W.T
        
        Args:
            output_gradient: Gradiente de la siguiente capa, forma (n_samples, output_size)
            
        Returns:
            Gradiente para pasar a la capa anterior, forma (n_samples, input_size)
        """
        activation_grad = self.activation.derivative(self.last_z)
        delta = output_gradient * activation_grad
        
        self.weights_gradient = self.last_input.T @ delta
        
        if self.use_bias:
            self.bias_gradient = np.sum(delta, axis=0)
        else:
            self.bias_gradient = None
        
        input_gradient = delta @ self.weights.T
        
        return input_gradient
    
    def update_weights(self, learning_rate: float) -> None:
        """
        Actualizar pesos y bias usando gradientes calculados.
        
        Fórmula: theta_{t+1} = theta_t - alpha * gradient_L(theta_t)
        
        Args:
            learning_rate: Tasa de aprendizaje para la actualización
        """
        self.weights -= learning_rate * self.weights_gradient
        
        if self.use_bias and self.bias_gradient is not None:
            self.bias -= learning_rate * self.bias_gradient
