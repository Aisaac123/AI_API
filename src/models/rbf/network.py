"""
Implementación de red RBF.
Esta es la red RBF completa que implementa la interfaz BaseModel.
"""

import numpy as np
from src.core.interfaces import BaseModel
from src.core.exceptions import NotFittedError, InvalidInputError
from .config import RBFConfig
from .layer import RBFLayer
from .solver import solve_pseudoinverse, compute_design_matrix


class RBFNetwork(BaseModel):
    """
    Red de Función de Base Radial (RBF).
    
    Esta es una red RBF pura con:
    - Una capa oculta con neuronas de función de base radial
    - Una capa de salida lineal resuelta vía pseudoinversa
    
    El proceso de entrenamiento es:
    1. Inicializar centros (usando estrategia provista o por defecto)
    2. Calcular matriz de diseño (activaciones de capa oculta)
    3. Resolver para pesos de salida usando pseudoinversa
    
    La salida de la red se calcula como: y_pred = Phi @ W + b
    donde:
    - Phi: matriz de diseño (activaciones RBF)
    - W: matriz de pesos de salida
    - b: vector de bias
    
    Esta es una solución de forma cerrada, no iterativa, lo cual hace que las redes RBF
    sean muy eficientes para entrenar comparadas con redes neuronales tradicionales.
    """
    
    def __init__(self, config: RBFConfig = None, centers: np.ndarray = None):
        """
        Inicializar la red RBF.
        
        Args:
            config: Objeto de configuración con parámetros de red
            centers: Centros pre-inicializados (opcional, sobrescribe config.n_centers)
        """
        if config is None:
            config = RBFConfig()
        
        self.config = config
        self.config.validate()
        
        # Inicializar centros si se proporcionan
        if centers is not None:
            self.centers = centers
            self.config.n_centers = centers.shape[0]
        else:
            self.centers = None
        
        # Crear la capa RBF (se inicializará durante fit)
        self.rbflayer = None
        
        # Pesos de salida (se calcularán durante fit)
        self.weights = None
        self.bias = None
        
        # Rastrear si el modelo ha sido entrenado
        self.is_fitted = False
        
        # Almacenar forma de datos de entrenamiento para validación
        self.n_features_ = None
        self.n_outputs_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, centers: np.ndarray = None) -> None:
        """
        Entrenar la red RBF con datos de entrada X y salidas objetivo y.
        
        El entrenamiento resuelve: W = pinv(Phi) @ y
        donde Phi es la matriz de diseño calculada como Phi = phi(d(X, C), sigma)
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            y: Matriz de salida objetivo de forma (n_samples, n_outputs)
            centers: Centros pre-calculados opcionales. Si no se proporcionan,
                    se usan centros de inicialización o centros previamente establecidos.
        
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
        
        # Usar centros proporcionados o centros existentes
        if centers is not None:
            self.centers = centers
            self.config.n_centers = centers.shape[0]
        elif self.centers is None:
            # Si no se proporcionan centros, muestrear aleatoriamente de datos de entrenamiento
            n_samples = X.shape[0]
            n_centers = min(self.config.n_centers, n_samples)
            indices = np.random.choice(n_samples, n_centers, replace=False)
            self.centers = X[indices]
        
        # Crear la capa RBF
        self.rbflayer = RBFLayer(
            centers=self.centers,
            activation=self.config.activation,
            sigma=self.config.sigma
        )
        
        # Calcular matriz de diseño (activaciones de capa oculta)
        Phi = self.rbflayer.forward(X)
        
        # Agregar columna de bias si está configurado
        if self.config.use_bias:
            Phi_bias = np.column_stack([Phi, np.ones(Phi.shape[0])])
        else:
            Phi_bias = Phi
        
        # Resolver para pesos de salida usando pseudoinversa
        self.weights = solve_pseudoinverse(
            Phi_bias,
            y,
            regularization=self.config.regularization
        )
        
        # Extraer bias si se usa
        if self.config.use_bias:
            self.bias = self.weights[-1, :]
            self.weights = self.weights[:-1, :]
        else:
            self.bias = np.zeros(self.n_outputs_)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Hacer predicciones para datos de entrada X.
        
        La predicción se calcula como: y_pred = Phi @ W + b
        
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
        
        # Calcular activaciones de capa oculta
        hidden_output = self.rbflayer.forward(X)
        
        # Calcular salida: hidden_output * weights + bias
        predictions = hidden_output @ self.weights + self.bias
        
        return predictions
    
    def summary(self) -> dict:
        """
        Retornar un diccionario con la configuración y estado del modelo.
        
        Returns:
            Diccionario que contiene información del modelo
        """
        info = {
            'model_type': 'RBFNetwork',
            'is_fitted': self.is_fitted,
            'config': self.config.to_dict(),
            'n_features': self.n_features_,
            'n_outputs': self.n_outputs_,
            'n_centers': self.config.n_centers,
            'activation': str(self.config.activation)
        }
        
        if self.is_fitted:
            info['weights_shape'] = self.weights.shape if self.weights is not None else None
            info['bias_shape'] = self.bias.shape if self.bias is not None else None
        
        return info
