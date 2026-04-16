"""
Implementación de capa oculta RBF.
Esta capa computa activaciones de función de base radial basadas en distancias euclidianas.
"""

import numpy as np
from src.core.interfaces import BaseLayer
from src.core.activation import ActivationFunction
from src.core.distance import euclidean_distance_matrix


class RBFLayer(BaseLayer):
    """
    Capa oculta de Función de Base Radial (RBF).
    
    Esta capa computa activaciones basadas en la distancia desde puntos de entrada
    a centros predefinidos. Cada neurona en esta capa representa un centro,
    y su activación se computa usando una función de base radial.
    
    El pase hacia adelante computa:
    1. Distancias euclidianas desde cada entrada a cada centro
    2. Aplicar función de activación a las distancias
    
    La salida de la capa se calcula como: Y = phi(d(X, C), sigma)
    donde:
    - X: matriz de entrada
    - C: matriz de centros
    - d: función de distancia euclidiana
    - phi: función de activación
    - sigma: parámetro de ancho
    """
    
    def __init__(self, centers: np.ndarray, activation: ActivationFunction, sigma: float):
        """
        Inicializar la capa RBF.
        
        Args:
            centers: Posiciones de centros de forma (n_centers, n_features)
            activation: Función de activación a usar (ej. GaussianActivation)
            sigma: Parámetro de ancho para la función de activación
        """
        self.centers = centers
        self.activation = activation
        self.sigma = sigma
        self.n_centers = centers.shape[0]
        self.n_features = centers.shape[1]
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calcular el pase hacia adelante de la capa RBF.
        
        La salida se calcula como: Y = phi(d(X, C), sigma)
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            
        Returns:
            Matriz de activación de forma (n_samples, n_centers)
        """
        # Calcular distancias euclidianas entre todas las muestras y todos los centros
        distances = euclidean_distance_matrix(X, self.centers)
        
        # Aplicar función de activación para obtener la salida de la capa
        activations = self.activation.compute(distances, self.sigma)
        
        return activations
    
    def get_centers(self) -> np.ndarray:
        """
        Obtener las posiciones de centros.
        
        Returns:
            Matriz de centros de forma (n_centers, n_features)
        """
        return self.centers
    
    def set_centers(self, centers: np.ndarray) -> None:
        """
        Establecer nuevas posiciones de centros.
        
        Args:
            centers: Nuevas posiciones de centros de forma (n_centers, n_features)
        """
        self.centers = centers
        self.n_centers = centers.shape[0]
        self.n_features = centers.shape[1]
