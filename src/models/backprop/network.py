"""
Implementación de red de retropropagación.
Esta es una red neuronal completa entrenada con descenso de gradiente y retropropagación.
"""

import numpy as np
from src.core.interfaces import BaseModel
from src.core.exceptions import NotFittedError, InvalidInputError
from .config import BackpropConfig
from .layer import DenseLayer


class BackpropNetwork(BaseModel):
    """
    Red neuronal de retropropagación.
    
    Esta es una red neuronal feedforward estándar entrenada con descenso de gradiente
    y retropropagación. Puede tener múltiples capas ocultas y soporta
    funciones de activación sigmoid, tanh y relu.
    
    El proceso de entrenamiento es iterativo:
    1. Pase hacia adelante a través de todas las capas
    2. Calcular error en la salida
    3. Pase hacia atrás para calcular gradientes
    4. Actualizar pesos usando descenso de gradiente
    
    Fórmulas de entrenamiento:
    - Error: E = (1/2) * sum((y - y_pred)^2)
    - Actualización de pesos: theta_{t+1} = theta_t - alpha * gradient_L(theta_t)
    - Retropropagación: delta_l = delta_{l+1} @ W_{l+1} @ phi'(z_l)
    """
    
    def __init__(self, config: BackpropConfig = None):
        """
        Inicializar la red de retropropagación.
        
        Args:
            config: Objeto de configuración con parámetros de red
        """
        if config is None:
            config = BackpropConfig()
        
        self.config = config
        self.config.validate()
        
        # Las capas se crearán durante fit
        self.layers = []
        
        # Rastrear si el modelo ha sido entrenado
        self.is_fitted = False
        
        # Almacenar forma de datos de entrenamiento para validación
        self.n_features_ = None
        self.n_outputs_ = None
    
    def _build_network(self, input_size: int, output_size: int) -> None:
        """
        Construir la arquitectura de red basada en configuración.
        
        Args:
            input_size: Número de características de entrada
            output_size: Número de neuronas de salida
        """
        self.layers = []
        
        # Construir capas ocultas
        layer_sizes = [input_size] + self.config.hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Determinar función de activación para esta capa
            if i < len(layer_sizes) - 2:
                # Capa oculta: usar layer_activations si está configurado, sino usar activation por defecto
                if self.config.layer_activations is not None:
                    activation = self.config.layer_activations[i]
                else:
                    activation = self.config.activation
            else:
                # Capa de salida: usar output_activation
                activation = self.config.output_activation
            
            layer = DenseLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activation,
                use_bias=self.config.use_bias
            )
            self.layers.append(layer)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrenar la red de retropropagación con datos de entrada X y salidas objetivo y.
        
        El entrenamiento minimiza la función de pérdida: L(theta) = (1/2) * sum((y - y_pred)^2)
        usando descenso de gradiente: theta_{t+1} = theta_t - alpha * gradient_L(theta_t)
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            y: Matriz de salida objetivo de forma (n_samples, n_outputs)
        
        Raises:
            InvalidInputError: Si las formas de entrada son inválidas
        """
        # Validar formas de entrada
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise InvalidInputError(f"X debe ser array 2D, se obtuvo forma {X.shape}")
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if y.ndim != 2:
            raise InvalidInputError(f"y debe ser array 1D o 2D, se obtuvo forma {y.shape}")
        
        if X.shape[0] != y.shape[0]:
            raise InvalidInputError(
                f"Discrepancia en número de muestras: X tiene {X.shape[0]}, y tiene {y.shape[0]}"
            )
        
        # Almacenar dimensiones
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        
        # Construir arquitectura de red (asegurar que se construya antes del entrenamiento)
        self._build_network(self.n_features_, self.n_outputs_)
        
        # Verificar que las capas fueron creadas
        if len(self.layers) == 0:
            raise RuntimeError("Las capas de red no fueron creadas correctamente")
        
        # Bucle de entrenamiento
        n_samples = X.shape[0]
        batch_size = self.config.batch_size if self.config.batch_size != -1 else n_samples
        
        for epoch in range(self.config.epochs):
            # Mezclar datos cada época
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Descenso de gradiente mini-batch
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # Pase hacia adelante
                output = self._forward_pass(X_batch)
                
                # Calcular gradiente de error en salida: gradient_y = y_pred - y
                error_gradient = output - y_batch
                
                # Pase hacia atrás (retropropagación)
                self._backward_pass(error_gradient)
                
                # Actualizar pesos
                for layer in self.layers:
                    layer.update_weights(self.config.learning_rate)
        
        self.is_fitted = True
    
    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """
        Calcular pase hacia adelante a través de todas las capas.
        
        Fórmula: a_l = phi(z_l), donde z_l = a_{l-1} @ W_l + b_l
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            
        Returns:
            Matriz de salida de forma (n_samples, n_outputs)
        """
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation
    
    def _backward_pass(self, output_gradient: np.ndarray) -> None:
        """
        Calcular pase hacia atrás a través de todas las capas.
        
        Fórmula: delta_l = delta_{l+1} @ W_{l+1} @ phi'(z_l)
        
        Args:
            output_gradient: Gradiente en la capa de salida
        """
        gradient = output_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Hacer predicciones para datos de entrada X.
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
        
        Returns:
            Matriz de predicciones de forma (n_samples, n_outputs)
        
        Raises:
            NotFittedError: Si el modelo no ha sido entrenado aún
            InvalidInputError: Si la forma de entrada es inválida
        """
        if not self.is_fitted:
            raise NotFittedError("El modelo debe ser ajustado antes de hacer predicciones")
        
        X = np.asarray(X)
        
        if X.ndim != 2:
            raise InvalidInputError(f"X debe ser array 2D, se obtuvo forma {X.shape}")
        
        if X.shape[1] != self.n_features_:
            raise InvalidInputError(
                f"Se esperaban {self.n_features_} características, se obtuvieron {X.shape[1]}"
            )
        
        return self._forward_pass(X)
    
    def summary(self) -> dict:
        """
        Retornar un diccionario con la configuración y estado del modelo.
        
        Returns:
            Diccionario que contiene información del modelo
        """
        info = {
            'model_type': 'BackpropNetwork',
            'is_fitted': self.is_fitted,
            'config': self.config.to_dict(),
            'n_features': self.n_features_,
            'n_outputs': self.n_outputs_,
            'hidden_layers': self.config.hidden_layers,
            'n_layers': len(self.layers)
        }
        
        if self.is_fitted:
            info['layer_shapes'] = [(layer.input_size, layer.output_size) for layer in self.layers]
        
        return info
