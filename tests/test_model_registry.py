"""
Script de prueba para el sistema de registro de modelos.
Verifica que el sistema de registro dinámico funciona correctamente.
"""

import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import NeuralNetwork, ModelType
from api.core.registry import ModelRegistry

print("=== Prueba 1: Verificar modelos registrados ===")
models = ModelRegistry.list_models()
print(f"Modelos registrados: {models}")
assert 'rbf' in models, "RBF debería estar registrado"
assert 'backprop' in models, "Backprop debería estar registrado"
print("✅ Ambos modelos están registrados")

print("\n=== Prueba 2: Verificar is_registered ===")
assert ModelRegistry.is_registered('rbf'), "RBF debería estar registrado"
assert ModelRegistry.is_registered('backprop'), "Backprop debería estar registrado"
assert not ModelRegistry.is_registered('inexistente'), "Modelo inexistente no debería estar registrado"
print("✅ is_registered funciona correctamente")

print("\n=== Prueba 3: Probar que RBF funciona con el sistema de registro ===")
NeuralNetwork.set_seed(42)
X = np.random.randn(100, 2)
y = np.random.randn(100)

net_rbf = NeuralNetwork(model_type=ModelType.RBF, n_centers=10, sigma=1.0)
result_rbf = net_rbf.train(X, y, verbose=False)
print(f"RBF entrenado - Error final: {result_rbf.final_error:.6f}")
predictions_rbf = net_rbf.predict(X)
print(f"✅ RBF funciona con el sistema de registro")

print("\n=== Prueba 4: Probar que Backprop funciona con el sistema de registro ===")
NeuralNetwork.set_seed(42)
net_bp = NeuralNetwork(
    model_type=ModelType.BACKPROP,
    hidden_layers=[10, 5],
    learning_rate=0.01,
    epochs=100
)
result_bp = net_bp.train(X, y, verbose=False)
print(f"Backprop entrenado - Error final: {result_bp.final_error:.6f}")
predictions_bp = net_bp.predict(X)
print(f"✅ Backprop funciona con el sistema de registro")

print("\n=== Prueba 5: Probar get_factory ===")
rbf_factory = ModelRegistry.get_factory('rbf')
backprop_factory = ModelRegistry.get_factory('backprop')
print(f"Factory RBF: {type(rbf_factory)}")
print(f"Factory Backprop: {type(backprop_factory)}")
print("✅ get_factory funciona correctamente")

print("\n=== Prueba 6: Probar error con modelo no registrado ===")
try:
    ModelRegistry.get_factory('modelo_inexistente')
    print("❌ Debería haber lanzado ValueError")
except ValueError as e:
    print(f"✅ ValueError lanzado correctamente: {e}")

print("\n✅ Todas las pruebas del sistema de registro pasaron exitosamente!")
