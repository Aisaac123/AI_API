"""
Estrategias de inicialización de centros para redes RBF.
Estas estrategias determinan cómo posicionar los centros RBF en el espacio de entrada.
"""

import numpy as np
from src.core.interfaces import BaseCenterInitializer


class RandomInitializer(BaseCenterInitializer):
    """
    Estrategia de inicialización aleatoria.
    
    Esta estrategia selecciona aleatoriamente n_centers muestras de los datos de entrada
    para servir como centros RBF. Es simple y a menudo funciona bien en la práctica.
    """
    
    def initialize(self, X: np.ndarray, n_centers: int) -> np.ndarray:
        """
        Inicializar centros muestreando aleatoriamente de los datos de entrada.
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            n_centers: Número de centros a inicializar
            
        Returns:
            Matriz de centros de forma (n_centers, n_features)
        """
        n_samples = X.shape[0]
        n_centers = min(n_centers, n_samples)
        
        indices = np.random.choice(n_samples, n_centers, replace=False)
        centers = X[indices]
        
        return centers


class KMeansInitializer(BaseCenterInitializer):
    """
    Estrategia de inicialización K-means.
    
    Esta estrategia usa clustering k-means para encontrar centros representativos.
    Los centros son los centroides de los clusters, que tienden a estar bien distribuidos
    a través de la distribución de datos.
    
    Objetivo de k-means: minimizar J = sum_i ||x_i - mu_{c_i}||^2
    donde mu_{c_i} es el centroide del cluster asignado al punto x_i.
    
    Algoritmo:
    1. Inicializar centroides aleatoriamente
    2. Asignar cada punto al centroide más cercano
    3. Recalcular centroides como media de puntos asignados
    4. Repetir hasta convergencia
    
    Nota: Esta es una implementación simplificada de k-means sin usar sklearn.
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-4):
        """
        Inicializar la estrategia k-means.
        
        Args:
            max_iterations: Número máximo de iteraciones k-means
            tolerance: Tolerancia de convergencia para movimiento de centroides
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def initialize(self, X: np.ndarray, n_centers: int) -> np.ndarray:
        """
        Inicializar centros usando clustering k-means.
        
        El algoritmo minimiza: J = sum_i ||x_i - mu_{c_i}||^2
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            n_centers: Número de centros a inicializar
            
        Returns:
            Matriz de centros de forma (n_centers, n_features)
        """
        n_samples = X.shape[0]
        n_centers = min(n_centers, n_samples)
        
        # Inicializar centroides aleatoriamente
        indices = np.random.choice(n_samples, n_centers, replace=False)
        centroids = X[indices].copy()
        
        # Ejecutar iteraciones k-means
        for iteration in range(self.max_iterations):
            # Asignar cada muestra al centroide más cercano
            distances = self._compute_distances(X, centroids)
            assignments = np.argmin(distances, axis=1)
            
            # Calcular nuevos centroides: mu_k = (1/|C_k|) * sum_{x_i in C_k} x_i
            new_centroids = np.zeros_like(centroids)
            for k in range(n_centers):
                mask = assignments == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # Verificar convergencia: ||mu_new - mu_old|| < tolerance
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            if centroid_shift < self.tolerance:
                break
            
            centroids = new_centroids
        
        return centroids
    
    def _compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Calcular distancias euclidianas entre todas las muestras y todos los centros.
        
        Fórmula: d(x, c) = sqrt(sum((x_i - c_i)^2))
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            centers: Matriz de centros de forma (n_centers, n_features)
            
        Returns:
            Matriz de distancias de forma (n_samples, n_centers)
        """
        X_expanded = X[:, np.newaxis, :]
        centers_expanded = centers[np.newaxis, :, :]
        squared_diff = (X_expanded - centers_expanded) ** 2
        return np.sqrt(np.sum(squared_diff, axis=2))
