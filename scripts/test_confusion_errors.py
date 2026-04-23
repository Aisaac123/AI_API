"""
Script de prueba para matriz de confusión con errores.

Usa un problema más complejo (círculos concéntricos) y entrenamiento
insuficiente para generar errores de clasificación.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.neural_network import NeuralNetwork
from api.core.model_type import ModelType


def generate_circles_data(n_samples, noise=0.1):
    """
    Generar datos de círculos concéntricos (problema más difícil).
    
    Args:
        n_samples: Número de muestras a generar
        noise: Cantidad de ruido a agregar
        
    Returns:
        X: Matriz de entrada (n_samples, 2)
        y: Vector de salida (n_samples, 1)
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Elegir radio aleatorio
        radius = np.random.choice([0.5, 1.5, 2.5])
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Convertir a coordenadas cartesianas
        x = radius * np.cos(angle) + np.random.normal(0, noise)
        y_coord = radius * np.sin(angle) + np.random.normal(0, noise)
        
        X.append([x, y_coord])
        # Clase basada en el radio (círculo interno vs externo)
        y.append(0 if radius < 1.5 else 1)
    
    return np.array(X), np.array(y).reshape(-1, 1)


def main():
    X_train, y_train = generate_circles_data(50, noise=0.15)
    X_test, y_test = generate_circles_data(20, noise=0.15)
    
    # Red RBF con pocos centros y entrenamiento insuficiente
    net = NeuralNetwork(
        model_type=ModelType.RBF,
        n_centers=8,  # Pocos centros para generar errores
        sigma=0.3,  # Sigma pequeño
        random_state=42
    )
    net.train(X_train, y_train)
    
    y_pred = net.predict(X_test)
    result = net.confusion_matrix(y_test, y_pred)
    
    print("=" * 60)
    print("DATASETS (Círculos Concéntricos con Ruido)")
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
    print(f"Precision: {result.precision}")
    print(f"Recall: {result.recall}")
    print(f"F1-score: {result.f1_score}")
    
    # Análisis de errores
    y_pred_discrete = (y_pred >= 0.5).astype(int)
    errors = np.where(y_pred_discrete.flatten() != y_test.flatten())[0]
    
    if len(errors) > 0:
        print(f"\n❌ {len(errors)} errores de clasificación:")
        for idx in errors:
            print(f"  Muestra {idx}: y_true={y_test[idx][0]}, y_pred={y_pred[idx][0]:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
