"""
Funciones de activación para redes RBF y Backpropagation.
Cada función sigue el patrón Strategy y es intercambiable.
"""

from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    """
    Clase base abstracta para funciones de activación.
    Todas las funciones de activación deben implementar el método compute.
    """

    @abstractmethod
    def compute(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Calcular el valor de activación dado las distancias y el parámetro de ancho.
        
        Para RBF: phi(d, sigma) donde d es la distancia y sigma es el parámetro de ancho.
        Para Backprop: phi(z) donde z es la pre-activación.
        
        Args:
            x: Matriz de distancias (RBF) o pre-activaciones (Backprop)
            sigma: Parámetro de ancho (RBF) o ignorado (Backprop)
            
        Returns:
            Matriz de activación
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Calcular la derivada de la función de activación.
        
        Requerido para backpropagation.
        
        Args:
            x: Matriz de pre-activaciones
            sigma: Parámetro de ancho (ignorado en backprop)
            
        Returns:
            Matriz de derivadas
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Retornar el nombre de la función de activación."""
        pass


# Funciones de activación para RBF (requieren sigma)

class GaussianActivation(ActivationFunction):
    """
    Función de activación Gaussiana (RBF).
    
    Fórmula: phi(r, sigma) = exp(-(r/sigma)^2)
    """

    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return np.exp(-(distances / sigma) ** 2)

    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        raise NotImplementedError("Derivada no requerida para funciones RBF")

    def __str__(self) -> str:
        return "Gaussian"


class MultiquadraticActivation(ActivationFunction):
    """
    Función de activación Multicuadrática (RBF).
    
    Fórmula: phi(r, sigma) = sqrt(1 + (r/sigma)^2)
    """

    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return np.sqrt(distances ** 2 + sigma ** 2)

    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        raise NotImplementedError("Derivada no requerida para funciones RBF")

    def __str__(self) -> str:
        return "Multiquadratic"


class InverseMultiquadraticActivation(ActivationFunction):
    """
    Función de activación Multicuadrática Inversa (RBF).
    
    Fórmula: phi(r, sigma) = 1 / sqrt(r^2 + sigma^2)
    """

    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return 1.0 / np.sqrt(distances ** 2 + sigma ** 2)

    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        raise NotImplementedError("Derivada no requerida para funciones RBF")

    def __str__(self) -> str:
        return "Inverse Multiquadratic"


class ThinPlateSplineActivation(ActivationFunction):
    """
    Función de activación Thin Plate Spline (RBF).
    
    Fórmula: phi(r) = r^2 * ln(r), con phi(0) = 0
    """

    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        scaled_distances = distances / sigma
        result = np.zeros_like(scaled_distances)
        mask = scaled_distances > 0
        result[mask] = scaled_distances[mask] ** 2 * np.log(scaled_distances[mask])
        return result

    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        raise NotImplementedError("Derivada no requerida para funciones RBF")

    def __str__(self) -> str:
        return "Thin Plate Spline"


# Funciones de activación para Backpropagation (estilo MATLAB)

class SigmoidActivation(ActivationFunction):
    """
    Función de activación Sigmoid (logsig en MATLAB).
    
    Fórmula: phi(z) = 1 / (1 + exp(-z))
    Derivada: phi'(z) = phi(z) * (1 - phi(z))
    """

    def compute(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        s = self.compute(x)
        return s * (1.0 - s)

    def __str__(self) -> str:
        return "Sigmoid (logsig)"


class TanhActivation(ActivationFunction):
    """
    Función de activación Tanh (tansig en MATLAB).
    
    Fórmula: phi(z) = tanh(z)
    Derivada: phi'(z) = 1 - tanh(z)^2
    """

    def compute(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        t = self.compute(x)
        return 1.0 - t ** 2

    def __str__(self) -> str:
        return "Tanh (tansig)"


class ReLUActivation(ActivationFunction):
    """
    Función de activación ReLU (Rectified Linear Unit).
    
    Fórmula: phi(z) = max(0, z)
    Derivada: phi'(z) = 1 si z > 0, 0 si z <= 0
    """

    def compute(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return np.maximum(0.0, x)

    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return (x > 0.0).astype(float)

    def __str__(self) -> str:
        return "ReLU"


class LinearActivation(ActivationFunction):
    """
    Función de activación Lineal (purelin en MATLAB).
    
    Fórmula: phi(z) = z
    Derivada: phi'(z) = 1
    """

    def compute(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return np.ones_like(x)

    def __str__(self) -> str:
        return "Linear (purelin)"


class LeakyReLUActivation(ActivationFunction):
    """
    Función de activación Leaky ReLU.
    
    Fórmula: phi(z) = max(0.01*z, z)
    Derivada: phi'(z) = 1 si z > 0, 0.01 si z <= 0
    """

    def compute(self, x: np.ndarray, sigma: float = 1.0, alpha: float = 0.01) -> np.ndarray:
        return np.maximum(alpha * x, x)

    def derivative(self, x: np.ndarray, sigma: float = 1.0, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0.0, 1.0, alpha)

    def __str__(self) -> str:
        return "Leaky ReLU"
