"""
Excepciones personalizadas para el proyecto de red RBF.
Estas excepciones proporcionan información específica de error para diferentes escenarios de falla.
"""


class RBFNetworkError(Exception):
    """Excepción base para todos los errores relacionados con la red RBF."""
    pass


class NotFittedError(RBFNetworkError):
    """Se lanza cuando se llama a un método en un modelo que no ha sido entrenado aún."""
    pass


class InvalidConfigError(RBFNetworkError):
    """Se lanza cuando la configuración del modelo es inválida o inconsistente."""
    pass


class InvalidInputError(RBFNetworkError):
    """Se lanza cuando los datos de entrada tienen forma o tipo incorrecto."""
    pass


class ConvergenceError(RBFNetworkError):
    """Se lanza cuando el proceso de entrenamiento falla en converger."""
    pass
