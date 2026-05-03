"""
REPL interactivo para redes neuronales con contexto completo de la aplicación.
Este script carga todos los imports y configuraciones necesarios para trabajar
con la API de redes neuronales de forma interactiva.
"""

import sys
import os

# Agregar directorio raíz al path
# Manejar el caso donde __file__ no está definido (ej. cuando se ejecuta con exec())
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Si __file__ no está definido, usar el directorio actual
    project_root = os.getcwd()
    # Asumir que estamos en el directorio raíz del proyecto
    if os.path.basename(project_root) == 'repl':
        project_root = os.path.dirname(project_root)

sys.path.insert(0, project_root)

# Importar numpy y matplotlib para visualización
import numpy as np

# Intentar importar matplotlib para visualización (opcional)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib no disponible. Instálalo con: pip install matplotlib")

# Importar API principal
from api import NeuralNetwork, ModelType
from api.config import NeuralNetworkConfig

# Importar modelos internos para uso avanzado
from src.models import RBFNetwork, RBFConfig, BackpropNetwork, BackpropConfig

# Importar funciones de activación
from src.core import (
    GaussianActivation,
    MultiquadraticActivation,
    InverseMultiquadraticActivation,
    ThinPlateSplineActivation,
    ThinPlateSplineLog10Activation
)

# Importar entrenadores e inicializadores
from src.training import RBFTrainer, BackpropTrainer, KMeansInitializer, RandomInitializer

# Importar evaluación
from src.evaluation import Evaluator
from src.evaluation import metrics

# Importar excepciones
from src.core import (
    NotFittedError,
    InvalidConfigError,
    InvalidInputError,
    ConvergenceError
)

# Importar funciones de distancia
from src.core import euclidean_distance, euclidean_distance_matrix

# Configuración de numpy para mejor visualización
np.set_printoptions(precision=4, suppress=True, linewidth=100)

print("=" * 70)
print("REPL de Redes Neuronales - Contexto Cargado")
print("=" * 70)
print()
print("Componentes disponibles:")
print("  - NeuralNetwork: API principal para crear y entrenar redes")
print("  - ModelType: Enum para seleccionar tipo de modelo (RBF, BACKPROP)")
print("  - NeuralNetworkConfig: Clase de configuración estructurada")
print()
print("Modelos internos disponibles:")
print("  - RBFNetwork, RBFConfig: Red RBF y configuración")
print("  - BackpropNetwork, BackpropConfig: Red de retropropagación y configuración")
print()
print("Funciones de activación:")
print("  - GaussianActivation, MultiquadraticActivation")
print("  - InverseMultiquadraticActivation, ThinPlateSplineActivation")
print()
print("Entrenadores:")
print("  - RBFTrainer, BackpropTrainer")
print("  - KMeansInitializer, RandomInitializer")
print()
print("Evaluación:")
print("  - Evaluator: Clase para evaluar modelos")
print("  - metrics: mse, mae, rmse, r2_score, accuracy")
print()
print("Excepciones:")
print("  - NotFittedError, InvalidConfigError, InvalidInputError, ConvergenceError")
print()
print("Ejemplos de uso:")
print("  # Crear red RBF")
print("  net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)")
print("  # Entrenar")
print("  X = np.random.randn(100, 2)")
print("  y = np.random.randn(100, 1)")
print("  net.train(X, y, verbose=True)")
print("  # Predecir")
print("  predictions = net.predict(X)")
print("  # Evaluar")
print("  metrics = net.evaluate(X, y)")
print()
print("Para salir, usa exit() o Ctrl+D")
print("=" * 70)
print()

# Variables globales convenientes
__all__ = [
    'np',
    'NeuralNetwork',
    'ModelType',
    'NeuralNetworkConfig',
    'RBFNetwork',
    'RBFConfig',
    'BackpropNetwork',
    'BackpropConfig',
    'GaussianActivation',
    'MultiquadraticActivation',
    'InverseMultiquadraticActivation',
    'ThinPlateSplineActivation',
    'RBFTrainer',
    'BackpropTrainer',
    'KMeansInitializer',
    'RandomInitializer',
    'Evaluator',
    'metrics',
    'NotFittedError',
    'InvalidConfigError',
    'InvalidInputError',
    'ConvergenceError',
    'euclidean_distance',
    'euclidean_distance_matrix',
]

# Si matplotlib está disponible, agregar a __all__
if MATPLOTLIB_AVAILABLE:
    __all__.append('plt')
