"""
Ejemplo de uso de funciones de activación por capa (estilo MATLAB).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from api import NeuralNetwork, ModelType, NeuralNetworkConfig

# Crear datos de prueba
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.random.randn(100, 1)

print("=== Ejemplo 1: Funciones de activación por capa (estilo MATLAB) ===")
print()

# Configurar red con diferentes funciones de activación por capa
# Similar a MATLAB: net = newff([2 10 5 1], {'tansig', 'logsig', 'purelin'})
config = NeuralNetworkConfig(
    hidden_layers=[10, 5],
    layer_activations=['tansig', 'logsig'],  # Funciones por capa
    output_activation='purelin',  # Función de salida
    learning_rate=0.01,
    epochs=100
)

net = NeuralNetwork(model_type=ModelType.BACKPROP, config=config)
result = net.train(X, y, verbose=True)

print()
print("Información de las capas:")
layer_info = net.get_layer_info()
for layer in layer_info:
    print(f"  Capa {layer['layer_index']}: {layer['layer_type']}")
    print(f"    Tamaño: {layer['input_size']} -> {layer['output_size']}")
    print(f"    Activación: {layer['activation']}")
    print(f"    Pesos: {layer['weights_shape']}")
    print()

print("=== Ejemplo 2: Ver pesos de una capa específica ===")
print()
# Ver pesos de la primera capa oculta
layer_0 = net.get_layer_weights(0)
print(f"Pesos de la capa 0:")
print(f"  Forma: {layer_0['weights'].shape}")
print(f"  Bias: {layer_0['bias']}")
print(f"  Activación: {layer_0['activation']}")
print()

# Ver pesos de la capa de salida
layer_output = net.get_layer_weights(-1)  # -1 para última capa
print(f"Pesos de la capa de salida:")
print(f"  Forma: {layer_output['weights'].shape}")
print(f"  Bias: {layer_output['bias']}")
print(f"  Activación: {layer_output['activation']}")
print()

print("=== Ejemplo 3: Usar nombres estilo MATLAB ===")
print()
# También se pueden usar los nombres estándar
config2 = NeuralNetworkConfig(
    hidden_layers=[8, 4],
    layer_activations=['tanh', 'sigmoid'],  # Nombres estándar también funcionan
    output_activation='linear',
    learning_rate=0.02,
    epochs=50
)

net2 = NeuralNetwork(model_type=ModelType.BACKPROP, config=config2)
result2 = net2.train(X, y, verbose=True)

print()
print("Información de las capas:")
layer_info2 = net2.get_layer_info()
for layer in layer_info2:
    print(f"  Capa {layer['layer_index']}: {layer['layer_type']}")
    print(f"    Activación: {layer['activation']}")
print()

print("=== Ejemplo 4: Función de activación por defecto para todas las capas ===")
print()
# Si layer_activations es None, se usa activation_backprop para todas las capas
config3 = NeuralNetworkConfig(
    hidden_layers=[10],
    activation_backprop='relu',  # Se aplica a todas las capas ocultas
    output_activation='linear',
    learning_rate=0.01,
    epochs=50
)

net3 = NeuralNetwork(model_type=ModelType.BACKPROP, config=config3)
result3 = net3.train(X, y, verbose=True)

print()
print("Información de las capas:")
layer_info3 = net3.get_layer_info()
for layer in layer_info3:
    print(f"  Capa {layer['layer_index']}: {layer['layer_type']}")
    print(f"    Activación: {layer['activation']}")
print()

print("=== Ejemplo 5: Comparación con nombres MATLAB vs estándar ===")
print()
print("Nombres estilo MATLAB: logsig, tansig, purelin")
print("Nombres estándar: sigmoid, tanh, linear")
print("Ambos son equivalentes y funcionan de la misma manera.")
