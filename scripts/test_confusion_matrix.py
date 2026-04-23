"""
Script de prueba para la funcionalidad de matriz de confusión con XOR.

Este script genera datos XOR, entrena una red, predice y calcula
la matriz de confusión con una sola salida binaria (0 o 1).
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
    print("=" * 70)
    print("PRUEBA DE MATRIZ DE CONFUSIÓN CON XOR (UNA SOLA SALIDA)")
    print("=" * 70)
    
    # Paso 1: Generar datos de entrenamiento (50 patrones)
    print("\n--- Paso 1: Generar datos de entrenamiento (50 patrones) ---")
    X_train, y_train = generate_xor_data(50, noise=0.0)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Distribución de clases en entrenamiento:")
    print(f"  Clase 0: {np.sum(y_train == 0)}")
    print(f"  Clase 1: {np.sum(y_train == 1)}")
    
    # Paso 2: Crear y entrenar la red
    print("\n--- Paso 2: Crear y entrenar la red ---")
    net = NeuralNetwork(
        model_type=ModelType.BACKPROP,
        hidden_layers=[4],
        learning_rate=0.01,
        epochs=5000,
        random_state=42
    )
    
    print("Entrenando red...")
    training_result = net.train(X_train, y_train)
    print(f"Entrenamiento completado:")
    print(f"  Épocas: {training_result.epochs}")
    print(f"  Error final: {training_result.final_error:.6f}")
    print(f"  Tiempo: {training_result.training_time:.2f}s")
    
    # Paso 3: Generar datos de prueba (20 patrones)
    print("\n--- Paso 3: Generar datos de prueba (20 patrones) ---")
    X_test, y_test = generate_xor_data(20, noise=0.0)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Distribución de clases en prueba:")
    print(f"  Clase 0: {np.sum(y_test == 0)}")
    print(f"  Clase 1: {np.sum(y_test == 1)}")
    
    # Mostrar algunos ejemplos de prueba
    print("\n--- Ejemplos de prueba ---")
    for i in range(min(5, len(X_test))):
        print(f"  Muestra {i}: X={X_test[i]}, y_true={y_test[i][0]}")
    
    # Paso 4: Hacer predicciones
    print("\n--- Paso 4: Hacer predicciones ---")
    y_pred = net.predict(X_test)
    print(f"y_pred shape: {y_pred.shape}")
    
    # Mostrar predicciones vs verdaderas
    print("\n--- Predicciones vs Verdaderas (primeras 5 muestras) ---")
    for i in range(min(5, len(X_test))):
        print(f"  Muestra {i}: X={X_test[i]}, y_true={y_test[i][0]}, y_pred={y_pred[i][0]:.4f}")
    
    # Paso 5: Calcular matriz de confusión
    print("\n--- Paso 5: Calcular matriz de confusión ---")
    result = net.confusion_matrix(y_test, y_pred)
    
    print("\n" + "=" * 70)
    print("RESULTADOS DE MATRIZ DE CONFUSIÓN")
    print("=" * 70)
    
    print("\n--- Matriz de Confusión (valores absolutos) ---")
    print("         Pred 0   Pred 1")
    print("True 0:  [{:^6}  {:^6}]".format(result.matrix[0, 0], result.matrix[0, 1]))
    print("True 1:  [{:^6}  {:^6}]".format(result.matrix[1, 0], result.matrix[1, 1]))
    print("\nFilas = Clases Verdaderas (y_true): 0, 1")
    print("Columnas = Clases Predichas (y_pred): 0, 1")
    print(f"\nMatriz es {result.matrix.shape[0]}x{result.matrix.shape[1]} (binario: 2 clases)")
    
    print("\n--- Matriz Normalizada por Fila (Recall) ---")
    print("         Pred 0   Pred 1")
    print("True 0:  [{:^6.2%}  {:^6.2%}]".format(result.matrix_normalized_row[0, 0], result.matrix_normalized_row[0, 1]))
    print("True 1:  [{:^6.2%}  {:^6.2%}]".format(result.matrix_normalized_row[1, 0], result.matrix_normalized_row[1, 1]))
    
    print("\n--- Matriz Normalizada por Columna (Precision) ---")
    print("         Pred 0   Pred 1")
    print("True 0:  [{:^6.2%}  {:^6.2%}]".format(result.matrix_normalized_col[0, 0], result.matrix_normalized_col[0, 1]))
    print("True 1:  [{:^6.2%}  {:^6.2%}]".format(result.matrix_normalized_col[1, 0], result.matrix_normalized_col[1, 1]))
    
    print("\n--- Métricas por Clase ---")
    for class_label in result.precision.keys():
        print(f"Clase {class_label}:")
        print(f"  Precision: {result.precision[class_label]:.4f}")
        print(f"  Recall: {result.recall[class_label]:.4f}")
        print(f"  F1-score: {result.f1_score[class_label]:.4f}")
        print(f"  Support: {result.support[class_label]}")
    
    print("\n--- Métricas Globales ---")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Macro Avg:")
    print(f"  Precision: {result.macro_avg['precision']:.4f}")
    print(f"  Recall: {result.macro_avg['recall']:.4f}")
    print(f"  F1-score: {result.macro_avg['f1-score']:.4f}")
    print(f"Weighted Avg:")
    print(f"  Precision: {result.weighted_avg['precision']:.4f}")
    print(f"  Recall: {result.weighted_avg['recall']:.4f}")
    print(f"  F1-score: {result.weighted_avg['f1-score']:.4f}")
    
    # Paso 6: Análisis detallado de errores
    print("\n--- Análisis Detallado de Errores ---")
    y_pred_discrete = (y_pred >= 0.5).astype(int)
    errors = np.where(y_pred_discrete.flatten() != y_test.flatten())[0]
    
    if len(errors) == 0:
        print("✅ ¡No hay errores de clasificación!")
    else:
        print(f"❌ {len(errors)} errores de clasificación:")
        for idx in errors:
            print(f"  Muestra {idx}: X={X_test[idx]}, y_true={y_test[idx][0]}, y_pred={y_pred[idx][0]:.4f} (pred_class={y_pred_discrete[idx][0]})")
    
    print("\n" + "=" * 70)
    print("PRUEBA COMPLETADA")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()
