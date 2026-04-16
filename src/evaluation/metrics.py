"""
Métricas de evaluación para rendimiento de modelo.
Estas son funciones puras que computan varias métricas entre predicciones y objetivos.
"""

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcular Error Cuadrático Medio (MSE).
    
    Fórmula: MSE = (1/n) * sum((y_true - y_pred)^2)
    
    Donde:
    - y_true: valor verdadero
    - y_pred: valor predicho
    - n: número de muestras
    
    Args:
        y_true: Valores objetivo verdaderos
        y_pred: Valores predichos
        
    Returns:
        Error cuadrático medio como escalar
    """
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcular Error Absoluto Medio (MAE).
    
    Fórmula: MAE = (1/n) * sum(|y_true - y_pred|)
    
    Donde:
    - y_true: valor verdadero
    - y_pred: valor predicho
    - n: número de muestras
    
    Args:
        y_true: Valores objetivo verdaderos
        y_pred: Valores predichos
        
    Returns:
        Error absoluto medio como escalar
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcular Raíz del Error Cuadrático Medio (RMSE).
    
    Fórmula: RMSE = sqrt(MSE) = sqrt((1/n) * sum((y_true - y_pred)^2))
    
    Donde:
    - y_true: valor verdadero
    - y_pred: valor predicho
    - n: número de muestras
    
    Args:
        y_true: Valores objetivo verdaderos
        y_pred: Valores predichos
        
    Returns:
        Raíz del error cuadrático medio como escalar
    """
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcular coeficiente de determinación R2 (R-squared).
    
    Fórmula: R2 = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))
    
    Donde:
    - y_true: valor verdadero
    - y_pred: valor predicho
    - mean(y_true): media de los valores verdaderos
    
    R2 mide qué tan bien las predicciones aproximan los datos verdaderos.
    Un valor de 1 indica predicción perfecta, 0 indica que el modelo
    funciona tan bien como predecir la media, y valores negativos indican
    peor rendimiento que predecir la media.
    
    Args:
        y_true: Valores objetivo verdaderos
        y_pred: Valores predichos
        
    Returns:
        Puntuación R-squared como escalar
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcular precisión de clasificación.
    
    Para tareas de regresión, esto computa el porcentaje de predicciones
    dentro de un pequeño umbral de los valores verdaderos.
    
    Para tareas de clasificación, esto computa el porcentaje de predicciones correctas.
    
    Fórmula: Accuracy = (1/n) * sum(|y_true - y_pred| < tolerance)
    
    Args:
        y_true: Valores objetivo verdaderos
        y_pred: Valores predichos
        
    Returns:
        Precisión como escalar entre 0 y 1
    """
    # Para simplicidad, tratar como regresión y contar predicciones dentro de 5% de tolerancia
    tolerance = 0.05 * np.std(y_true)
    correct = np.abs(y_true - y_pred) < tolerance
    return np.mean(correct)
