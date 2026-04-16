"""
Funciones de distancia euclidiana para redes RBF.
Estas son funciones puras sin estado, usadas para calcular distancias entre puntos.
"""

import numpy as np


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calcular la distancia euclidiana entre dos puntos.
    
    Fórmula: d(x1, x2) = sqrt(sum((x1 - x2)^2))
    
    Donde:
    - x1: primer punto
    - x2: segundo punto
    - sum: suma sobre todas las dimensiones
    
    Args:
        x1: Primer punto, forma (n_features,)
        x2: Segundo punto, forma (n_features,)
        
    Returns:
        Distancia euclidiana como escalar
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def euclidean_distance_matrix(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Calcular la matriz de distancia euclidiana entre todos los puntos en X y todos los centros.
    
    Esta es el cálculo de distancia euclidiana total usado en redes RBF.
    Retorna una matriz donde el elemento (i, j) es la distancia entre el punto i y el centro j.
    
    Fórmula: Para cada punto xi y centro cj:
    d(i, j) = sqrt(sum((xi - cj)^2))
    
    Donde:
    - xi: i-ésimo punto de entrada
    - cj: j-ésimo centro
    - sum: suma sobre todas las características (features)
    
    Args:
        X: Matriz de entrada de forma (n_samples, n_features)
        centers: Matriz de centros de forma (n_centers, n_features)
        
    Returns:
        Matriz de distancias de forma (n_samples, n_centers)
    """
    # Expandir dimensiones para broadcasting
    # X: (n_samples, 1, n_features)
    # centers: (1, n_centers, n_features)
    X_expanded = X[:, np.newaxis, :]
    centers_expanded = centers[np.newaxis, :, :]
    
    # Calcular diferencias al cuadrado y sumar sobre características
    squared_diff = (X_expanded - centers_expanded) ** 2
    squared_distances = np.sum(squared_diff, axis=2)
    
    # Tomar raíz cuadrada para obtener distancias euclidianas
    return np.sqrt(squared_distances)


def euclidean_distance_squared_matrix(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Calcular la matriz de distancia euclidiana al cuadrado (sin raíz cuadrada).
    
    Esto es útil para algunas funciones de activación que trabajan con distancias al cuadrado.
    
    Fórmula: d^2(i, j) = sum((xi - cj)^2)
    
    Donde:
    - xi: i-ésimo punto de entrada
    - cj: j-ésimo centro
    - sum: suma sobre todas las características
    
    Args:
        X: Matriz de entrada de forma (n_samples, n_features)
        centers: Matriz de centros de forma (n_centers, n_features)
        
    Returns:
        Matriz de distancias al cuadrado de forma (n_samples, n_centers)
    """
    X_expanded = X[:, np.newaxis, :]
    centers_expanded = centers[np.newaxis, :, :]
    
    squared_diff = (X_expanded - centers_expanded) ** 2
    return np.sum(squared_diff, axis=2)
