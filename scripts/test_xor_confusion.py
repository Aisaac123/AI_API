"""
Script de prueba específico para matriz de confusión con XOR.

Genera 50 patrones XOR para entrenamiento, entrena la red,
luego genera 20 patrones para prueba, predice y calcula
la matriz de confusión con las clases verdaderas.
"""

import numpy as np
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.neural_network import NeuralNetwork
from api.core.model_type import ModelType


def generate_xor_data(n_samples, noise=0.0):
    """
    Generar datos XOR con ruido opcional.
    
    Args:
        n_samples: Número de muestras a generar
        noise: Cantidad de ruido a agregar (0.0 a 1.0)
        
    Returns:
        X: Matriz de entrada (n_samples, 2)
        y: Vector de salida (n_samples, 1)
    """
    # Generar combinaciones de 0 y 1
    X = np.random.randint(0, 2, size=(n_samples, 2)).astype(float)
    
    # XOR: 0 si ambos son iguales, 1 si son diferentes
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(float).reshape(-1, 1)
    
    # Agregar ruido si se especifica
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)
    
    return X, y


def main():
    X_train, y_train = generate_xor_data(50, noise=0.0)
    X_test, y_test = generate_xor_data(20, noise=0.0)
    
    net = NeuralNetwork(
        model_type=ModelType.RBF,
        n_centers=20,
        sigma=0.5,
        random_state=42
    )
    net.train(X_train, y_train)
    
    y_pred = net.predict(X_test)
    result = net.confusion_matrix(y_test, y_pred)
    
    print("=" * 60)
    print("DATASETS")
    print("=" * 60)
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.flatten()}")
    print(f"\nX_test: {X_test.shape}")
    print(f"y_test: {y_test.flatten()}")
    print(f"y_pred: {y_pred.flatten()}")
    
    print("\n" + "=" * 60)
    print("MATRIZ DE CONFUSIÓN")
    print("=" * 60)
    print("         Pred 0   Pred 1")
    print("True 0:  [{:^6}  {:^6}]".format(result.matrix[0, 0], result.matrix[0, 1]))
    print("True 1:  [{:^6}  {:^6}]".format(result.matrix[1, 0], result.matrix[1, 1]))
    print(f"\nAccuracy: {result.accuracy:.4f}")
    print(f"Precision: {{'0': {result.precision['0.0']:.4f}, '1': {result.precision['1.0']:.4f}}}")
    print(f"Recall: {{'0': {result.recall['0.0']:.4f}, '1': {result.recall['1.0']:.4f}}}")
    print(f"F1-score: {{'0': {result.f1_score['0.0']:.4f}, '1': {result.f1_score['1.0']:.4f}}}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()
