"""
Solucionador de pseudoinversa para la capa de salida de red RBF.
Resuelve el sistema lineal para encontrar los pesos óptimos de la capa de salida.
"""

import numpy as np


def solve_pseudoinverse(Phi: np.ndarray, y: np.ndarray, regularization: float = 0.0) -> np.ndarray:
    """
    Resolver para pesos de salida usando pseudoinversa con regularización opcional.
    
    La capa de salida de red RBF computa: y = Phi @ W
    donde Phi es la matriz de diseño (activaciones de la capa oculta)
    y W son los pesos de salida que necesitamos encontrar.
    
    La solución es: W = pinv(Phi) @ y
    Con regularización: W = pinv(Phi.T @ Phi + lambda * I) @ Phi.T @ y
    
    Esta es una solución de forma cerrada, no iterativa, lo cual es una de las
    principales ventajas de las redes RBF sobre las redes neuronales tradicionales.
    
    Donde:
    - Phi: Matriz de diseño (activaciones)
    - W: Matriz de pesos de salida
    - lambda: Parámetro de regularización
    - I: Matriz identidad
    - pinv: Pseudoinversa de Moore-Penrose
    
    Args:
        Phi: Matriz de diseño de forma (n_samples, n_centers)
             Activaciones de la capa oculta RBF
        y: Salidas objetivo de forma (n_samples, n_outputs)
        regularization: Parámetro de regularización (lambda) para agregar a la diagonal
                        Ayuda a prevenir sobreajuste y mejora estabilidad numérica
    
    Returns:
        Matriz de pesos de forma (n_centers, n_outputs)
    """
    n_samples, n_centers = Phi.shape
    
    # Agregar columna de bias (columna de unos) si es necesario
    # Esto será manejado por el llamador, pero el solucionador trabaja con cualquier Phi dada
    
    if regularization > 0:
        # Pseudoinversa regularizada: (Phi.T @ Phi + lambda * I)^-1 @ Phi.T @ y
        Phi_T_Phi = Phi.T @ Phi
        identity = np.eye(n_centers)
        regularized_matrix = Phi_T_Phi + regularization * identity
        pseudoinverse = np.linalg.inv(regularized_matrix) @ Phi.T
    else:
        # Pseudoinversa estándar
        pseudoinverse = np.linalg.pinv(Phi)
    
    # Calcular pesos
    W = pseudoinverse @ y
    
    return W


def compute_design_matrix(X: np.ndarray, centers: np.ndarray, 
                         activation, sigma: float) -> np.ndarray:
    """
    Calcular la matriz de diseño (Φ) para red RBF.
    
    La matriz de diseño contiene las activaciones de cada neurona RBF
    para cada muestra de entrada. Φ[i, j] = activation(distancia(X[i], centro[j]))
    
    La matriz de diseño se calcula como: Φ = φ(d(X, C), σ)
    donde:
    - d(X, C): matriz de distancias euclidianas
    - φ: función de activación
    - σ: parámetro de ancho
    
    Args:
        X: Matriz de entrada de forma (n_samples, n_features)
        centers: Matriz de centros de forma (n_centers, n_features)
        activation: Instancia de función de activación (ej. GaussianActivation)
        sigma: Parámetro de ancho para la función de activación
    
    Returns:
        Matriz de diseño de forma (n_samples, n_centers)
    """
    from src.core.distance import euclidean_distance_matrix
    
    # Calcular distancias euclidianas entre todas las muestras y todos los centros
    distances = euclidean_distance_matrix(X, centers)
    
    # Aplicar función de activación a las distancias
    Phi = activation.compute(distances, sigma)
    
    return Phi
