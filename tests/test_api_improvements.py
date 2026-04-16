"""
Script de prueba para las nuevas mejoras de la API.
Prueba tipado fuerte, persistencia y control de reproducibilidad.
"""

import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import NeuralNetwork, ModelType, TrainingResult, EvaluationResult

# Establecer semilla para reproducibilidad
NeuralNetwork.set_seed(42)

# Generar datos de prueba
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.random.randn(100)

print("=== Prueba 1: Tipado fuerte con dataclasses ===")
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=10, sigma=1.0)
result: TrainingResult = net.train(X, y, verbose=True)

# Verificar que result tiene tipado fuerte
print(f"\nTipo de result: {type(result)}")
print(f"Tiempo de entrenamiento: {result.training_time:.4f}s")
print(f"Error final: {result.final_error:.6f}")
print(f"Épocas: {result.epochs}")
print(f"Convergido: {result.converged}")

# Probar evaluate con tipado fuerte
eval_result: EvaluationResult = net.evaluate(X, y, detailed=True)
print(f"\nTipo de eval_result: {type(eval_result)}")
print(f"MSE: {eval_result.mse:.6f}")
print(f"MAE: {eval_result.mae:.6f}")
print(f"RMSE: {eval_result.rmse:.6f}")
print(f"R2: {eval_result.r2:.6f}")
print(f"Accuracy: {eval_result.accuracy:.6f}")

print("\n=== Prueba 2: Persistencia con save/load ===")
# Crear carpeta models si no existe
os.makedirs('models', exist_ok=True)

# Guardar modelo
model_path = 'models/test_model.pkl'
net.save(model_path)
print(f"Modelo guardado en {model_path}")

# Cargar modelo
loaded_net = NeuralNetwork.load(model_path)
print(f"Modelo cargado desde {model_path}")

# Verificar que el modelo cargado funciona
loaded_predictions = loaded_net.predict(X)
original_predictions = net.predict(X)
print(f"\nPredicciones iguales: {np.allclose(loaded_predictions, original_predictions)}")

print("\n=== Prueba 3: Control de reproducibilidad ===")
# Entrenar dos veces con la misma semilla
NeuralNetwork.set_seed(42)
net1 = NeuralNetwork(model_type=ModelType.RBF, n_centers=10, sigma=1.0)
result1 = net1.train(X, y)

NeuralNetwork.set_seed(42)
net2 = NeuralNetwork(model_type=ModelType.RBF, n_centers=10, sigma=1.0)
result2 = net2.train(X, y)

print(f"Error net1: {result1.final_error:.6f}")
print(f"Error net2: {result2.final_error:.6f}")
print(f"Errores iguales: {np.isclose(result1.final_error, result2.final_error)}")

print("\n=== Prueba 4: Compatibilidad con dict (to_dict) ===")
# Verificar que las dataclasses tienen to_dict()
result_dict = result.to_dict()
print(f"Tipo de result_dict: {type(result_dict)}")
print(f"Claves: {result_dict.keys()}")

eval_dict = eval_result.to_dict()
print(f"\nTipo de eval_dict: {type(eval_dict)}")
print(f"Claves: {eval_dict.keys()}")

print("\n=== Prueba 5: get_layer_weights con tipado fuerte ===")
# Probar con backpropagation
NeuralNetwork.set_seed(42)
net_bp = NeuralNetwork(
    model_type=ModelType.BACKPROP,
    hidden_layers=[10, 5],
    learning_rate=0.01,
    epochs=100
)
result_bp = net_bp.train(X, y, verbose=False)

layer_weights = net_bp.get_layer_weights(0)
print(f"Tipo de layer_weights: {type(layer_weights)}")
print(f"Layer index: {layer_weights.layer_index}")
print(f"Layer type: {layer_weights.layer_type}")
print(f"Input size: {layer_weights.input_size}")
print(f"Output size: {layer_weights.output_size}")
print(f"Activation: {layer_weights.activation}")
print(f"Use bias: {layer_weights.use_bias}")

print("\n=== Prueba 6: summary con tipado fuerte ===")
summary = net.summary()
print(f"Tipo de summary: {type(summary)}")
print(f"Model type: {summary.model_type}")
print(f"Is fitted: {summary.is_fitted}")

print("\n✅ Todas las pruebas completadas exitosamente!")
