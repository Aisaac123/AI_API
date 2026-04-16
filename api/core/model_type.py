"""
Enum de tipo de modelo para seleccionar arquitecturas de redes neuronales.
"""

from enum import Enum


class ModelType(Enum):
    """
    Enum para seleccionar el tipo de modelo de red neuronal.
    
    RBF: Red de Funciones de Base Radial - entrenamiento rápido con solución de forma cerrada
    BACKPROP: Red de retropropagación - entrenamiento iterativo con descenso de gradiente
    """
    RBF = 'rbf'
    BACKPROP = 'backprop'
