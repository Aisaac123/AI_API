"""
Script de entrenamiento para dataset_rbf_2 (1).json

Proceso:
1. Cargar dataset desde JSON
2. Limpiar datos (manejar valores nulos)
3. Particionar 70/15/15 random (train/validation/test)
4. Entrenar modelo RBF
5. Evaluar en validation y test
"""

import json
import numpy as np
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.neural_network import NeuralNetwork
from api.core.model_type import ModelType


def load_json_data(json_path):
    """
    Cargar datos desde archivo JSON.

    Args:
        json_path: Ruta al archivo JSON

    Returns:
        dict: Datos cargados con dataset, features y data
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def clean_data(data_dict):
    """
    Limpiar datos: manejar valores nulos, convertir a numpy arrays.

    Args:
        data_dict: Diccionario con datos del JSON

    Returns:
        tuple: (X, y) arrays numpy limpios
    """
    X = []
    y = []

    for item in data_dict['data']:
        input_data = item['input']
        output_data = item['output']

        # Manejar nulos en input
        if any(val is None for val in input_data):
            continue

        # Manejar nulo en output
        if output_data is None:
            continue

        X.append(input_data)
        y.append(output_data)

    return np.array(X), np.array(y).reshape(-1, 1)


def split_data(X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=None):
    """
    Particionar datos en train/validation/test de forma random.

    Args:
        X: Array de características
        y: Array de etiquetas
        train_ratio: Proporción para entrenamiento (default 0.70)
        val_ratio: Proporción para validación (default 0.15)
        test_ratio: Proporción para prueba (default 0.15)
        random_state: Semilla para reproducibilidad

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    # Calcular índices para cada partición
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, '..', 'jsons', 'dataset_rbf_2.json')

    print("Cargando datos...")
    data_dict = load_json_data(json_path)
    print(f"Dataset: {data_dict['dataset']}, muestras: {len(data_dict['data'])}")

    print("Limpiando datos...")
    X, y = clean_data(data_dict)
    print(f"X: {X.shape}, y: {y.shape}, clases: {np.unique(y)}")

    print("Particionando 70/15/15...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    print("Entrenando RBF...")
    net = NeuralNetwork(
        model_type=ModelType.RBF,
        n_centers=min(50, X_train.shape[0] // 10),
        sigma=1.0,
        activation_rbf='gaussian',
        regularization=0.01,
        random_state=42
    )
    result = net.train(X_train, y_train, verbose=True)
    print(f"Tiempo: {result.training_time:.4f}s")

    print("Validation:")
    val_metrics = net.evaluate(X_val, y_val)
    print(f"MSE: {val_metrics.mse:.6f}, R²: {val_metrics.r2:.6f}")
    val_pred = net.predict(X_val)
    val_result = net.confusion_matrix(y_val, val_pred)
    print(f"Matriz confusión:\n{val_result.matrix}")
    print(f"Accuracy: {val_result.accuracy:.4f}")
    print(f"Precision: {val_result.precision}")
    print(f"Recall: {val_result.recall}")
    print(f"F1-score: {val_result.f1_score}")

    print("Test:")
    test_metrics = net.evaluate(X_test, y_test)
    print(f"MSE: {test_metrics.mse:.6f}, R²: {test_metrics.r2:.6f}")
    test_pred = net.predict(X_test)
    test_result = net.confusion_matrix(y_test, test_pred)
    print(f"Matriz confusión:\n{test_result.matrix}")
    print(f"Accuracy: {test_result.accuracy:.4f}")
    print(f"Precision: {test_result.precision}")
    print(f"Recall: {test_result.recall}")
    print(f"F1-score: {test_result.f1_score}")

    print("Completado")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
