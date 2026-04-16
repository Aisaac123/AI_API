"""
Validadores de entrada para la API de redes neuronales.
Esta clase separa la responsabilidad de validación de entrada.
"""

import numpy as np
from typing import Optional


class InputValidator:
    """
    Validador de entrada para datos de redes neuronales.
    
    Responsabilidad única: Validar que los datos de entrada cumplan con
    los requisitos esperados por los modelos.
    """
    
    @staticmethod
    def validate_X(X: np.ndarray) -> np.ndarray:
        """
        Validar matriz de entrada X.
        
        Args:
            X: Datos de entrada
            
        Returns:
            X convertido a array numpy
            
        Raises:
            ValueError: Si X no es válido
        """
        X = np.asarray(X)
        
        if X.ndim != 2:
            raise ValueError(f"X debe ser array 2D, se obtuvo forma {X.shape}")
        
        return X
    
    @staticmethod
    def validate_y(y: np.ndarray) -> np.ndarray:
        """
        Validar vector de salida y.
        
        Args:
            y: Datos de salida
            
        Returns:
            y convertido a array numpy 2D
            
        Raises:
            ValueError: Si y no es válido
        """
        y = np.asarray(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim != 2:
            raise ValueError(f"y debe ser array 1D o 2D, se obtuvo forma {y.shape}")
        
        return y
    
    @staticmethod
    def validate_compatibility(X: np.ndarray, y: np.ndarray) -> None:
        """
        Validar que X e y tengan el mismo número de muestras.
        
        Args:
            X: Datos de entrada
            y: Datos de salida
            
        Raises:
            ValueError: Si no son compatibles
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Discrepancia en número de muestras: X tiene {X.shape[0]}, y tiene {y.shape[0]}"
            )
    
    @classmethod
    def validate_input_pair(cls, X: np.ndarray, y: Optional[np.ndarray] = None) -> tuple:
        """
        Validar par de entrada X e y.
        
        Args:
            X: Datos de entrada
            y: Datos de salida opcional
            
        Returns:
            Tupla (X_validado, y_validado)
        """
        X_valid = cls.validate_X(X)
        
        if y is not None:
            y_valid = cls.validate_y(y)
            cls.validate_compatibility(X_valid, y_valid)
            return X_valid, y_valid
        
        return X_valid, None
