"""
Script de entrenamiento para dataset_rbf_2 (1).json

Proceso:
1. Cargar dataset desde JSON
2. Limpiar datos (manejar valores nulos)
3. Particionar 70/15/15 random (train/validation/test)
4. Entrenar modelo RBF
5. Evaluar en validation y test

Flags:
--mode: train|val|test (default: all)
--random: randomizar particionamiento (default: False)
"""

import json
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

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


def show_input_parameters(X, y, data_dict, verbose_decimals=False):
    """
    Mostrar parámetros de entrada y estadísticas descriptivas del dataset.

    Args:
        X: Array de características
        y: Array de etiquetas
        data_dict: Diccionario con datos del JSON
        verbose_decimals: Si True, muestra todos los decimales
    """
    print(f"\n==== Parámetros de Entrada ====")
    print(f"Dataset: {data_dict['dataset']}")
    print(f"Features: {data_dict['features']}")
    print(f"Muestras: {X.shape[0]}")
    print(f"Características: {X.shape[1]}")
    print(f"Clases: {np.unique(y)}")
    print(f"Número de clases: {len(np.unique(y))}")
    unique, counts = np.unique(y, return_counts=True)
    balanceo = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"Balanceo: {balanceo}")

    print(f"\n==== Estadísticas Descriptivas ====")
    print(f"Media X: {np.round(np.mean(X, axis=0), 4 if not verbose_decimals else None)}")
    print(f"Std X: {np.round(np.std(X, axis=0), 4 if not verbose_decimals else None)}")
    print(f"Min X: {np.round(np.min(X, axis=0), 4 if not verbose_decimals else None)}")
    print(f"Max X: {np.round(np.max(X, axis=0), 4 if not verbose_decimals else None)}")
    print(f"Media y: {format_float(np.mean(y), verbose_decimals)}")
    print(f"Std y: {format_float(np.std(y), verbose_decimals)}")


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


def format_float(value, verbose_decimals=False):
    """
    Formatear float a 4 decimales por defecto o todos si verbose.

    Args:
        value: Valor a formatear
        verbose_decimals: Si True, muestra todos los decimales

    Returns:
        Valor formateado como float
    """
    if verbose_decimals:
        return float(value)
    return round(float(value), 4)


def format_dict_floats(d, verbose_decimals=False):
    """
    Formatear todos los floats de un diccionario.

    Args:
        d: Diccionario con valores float
        verbose_decimals: Si True, muestra todos los decimales

    Returns:
        Diccionario con floats formateados
    """
    return {k: format_float(v, verbose_decimals) for k, v in d.items()}


def evaluate_model(net, X, y, label, verbose_decimals=False):
    """
    Evaluar modelo y mostrar métricas.

    Args:
        net: Modelo entrenado
        X: Datos de evaluación
        y: Etiquetas verdaderas
        label: Etiqueta para el print (ej: "Validation", "Test")
        verbose_decimals: Si True, muestra todos los decimales
    """
    print(f"\n==== {label} ====")
    metrics = net.evaluate(X, y)
    print(f"MSE: {format_float(metrics.mse, verbose_decimals)}, R²: {format_float(metrics.r2, verbose_decimals)}")
    pred = net.predict(X)
    result = net.confusion_matrix(y, pred)
    print(f"\nMatriz de Confusión:\n{result.matrix}")
    print(f"\nAccuracy global: {format_float(result.accuracy, verbose_decimals)}")

    # Formatear diccionarios
    precision = format_dict_floats(result.precision, verbose_decimals)
    recall = format_dict_floats(result.recall, verbose_decimals)
    specificity = format_dict_floats(result.specificity, verbose_decimals)
    f1 = format_dict_floats(result.f1_score, verbose_decimals)
    macro_avg = format_dict_floats(result.macro_avg, verbose_decimals)
    weighted_avg = format_dict_floats(result.weighted_avg, verbose_decimals)

    print(f"\nMétricas por clase:")
    print(f"  Precision: {precision}")
    print(f"  Recall (Sensibilidad): {recall}")
    print(f"  Specificity (Especificidad): {specificity}")
    print(f"  F1-score: {f1}")

    print(f"\nPromedios:")
    print(f"  Macro: {macro_avg}")
    print(f"  Ponderado: {weighted_avg}")


def plot_results(net, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name, verbose_decimals=False):
    """
    Generar 4 gráficas de resultados del modelo.

    Args:
        net: Modelo entrenado
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        X_test, y_test: Datos de prueba
        dataset_name: Nombre del dataset para títulos
        verbose_decimals: Si True, muestra todos los decimales
    """
    # Crear carpeta plots si no existe
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Configurar estilo
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Matriz de confusión (heatmap)
    pred_test = net.predict(X_test)
    result_test = net.confusion_matrix(y_test, pred_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(result_test.matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Matriz de Confusión - {dataset_name}')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{dataset_name}_confusion_matrix.png'), dpi=300)
    plt.close()

    # 2. Distribución de clases (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    classes_train, counts_train = np.unique(y_train, return_counts=True)
    classes_val, counts_val = np.unique(y_val, return_counts=True)
    classes_test, counts_test = np.unique(y_test, return_counts=True)

    x = np.arange(len(classes_train))
    width = 0.25

    ax.bar(x - width, counts_train, width, label='Train', alpha=0.8)
    ax.bar(x, counts_val, width, label='Validation', alpha=0.8)
    ax.bar(x + width, counts_test, width, label='Test', alpha=0.8)

    ax.set_xlabel('Clase')
    ax.set_ylabel('Número de muestras')
    ax.set_title(f'Distribución de Clases - {dataset_name}')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in classes_train])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{dataset_name}_class_distribution.png'), dpi=300)
    plt.close()

    # 3. Métricas por clase (bar chart comparativo)
    pred_val = net.predict(X_val)
    result_val = net.confusion_matrix(y_val, pred_val)

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = list(result_val.precision.keys())
    x = np.arange(len(labels))
    width = 0.2

    precision_vals = [format_float(result_val.precision[k], verbose_decimals) for k in labels]
    recall_vals = [format_float(result_val.recall[k], verbose_decimals) for k in labels]
    specificity_vals = [format_float(result_val.specificity[k], verbose_decimals) for k in labels]
    f1_vals = [format_float(result_val.f1_score[k], verbose_decimals) for k in labels]

    ax.bar(x - 1.5*width, precision_vals, width, label='Precision', alpha=0.8)
    ax.bar(x - 0.5*width, recall_vals, width, label='Recall', alpha=0.8)
    ax.bar(x + 0.5*width, specificity_vals, width, label='Specificity', alpha=0.8)
    ax.bar(x + 1.5*width, f1_vals, width, label='F1-score', alpha=0.8)

    ax.set_xlabel('Clase')
    ax.set_ylabel('Valor')
    ax.set_title(f'Métricas por Clase (Validation) - {dataset_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{dataset_name}_metrics_by_class.png'), dpi=300)
    plt.close()

    # 4. Scatter plot de datos (si features ≤ 3)
    if X_train.shape[1] <= 3:
        fig = plt.figure(figsize=(10, 8))

        if X_train.shape[1] == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train.flatten(), cmap='viridis', alpha=0.6)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        elif X_train.shape[1] == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train.flatten(), cmap='viridis', alpha=0.6)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')

        plt.colorbar(scatter, label='Clase')
        ax.set_title(f'Distribución de Datos - {dataset_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{dataset_name}_data_scatter.png'), dpi=300)
        plt.close()

    print(f"\nGráficas guardadas en: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo RBF con dataset')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'val', 'test', 'all'],
                        help='Modo de ejecución: train (solo entrenar), val (entrenar+validar), test (entrenar+test), all (todo)')
    parser.add_argument('--random', action='store_true',
                        help='Randomizar particionamiento 70/15/15')
    parser.add_argument('--verbose-decimals', action='store_true',
                        help='Mostrar todos los decimales (default: 4 decimales)')
    parser.add_argument('--plot', action='store_true',
                        help='Generar gráficas de resultados')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, '..', 'jsons', 'dataset_rbf_2.json')

    print("Cargando datos...")
    data_dict = load_json_data(json_path)
    print(f"Dataset: {data_dict['dataset']}, muestras: {len(data_dict['data'])}")

    print("Limpiando datos...")
    X, y = clean_data(data_dict)
    show_input_parameters(X, y, data_dict, args.verbose_decimals)

    # Randomizar o no según flag
    random_state = None if args.random else 42
    print(f"\n==== Particionamiento ====")
    print(f"70/15/15 (random={args.random})")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    print(f"\n==== Entrenamiento ====")
    net = NeuralNetwork(
        model_type=ModelType.RBF,
        n_centers=min(50, X_train.shape[0] // 10),
        sigma=1.0,
        activation_rbf='gaussian',
        regularization=0.01,
        random_state=42
    )
    result = net.train(X_train, y_train, verbose=True)
    print(f"Tiempo: {format_float(result.training_time, args.verbose_decimals)}s")

    # Evaluar según modo
    if args.mode in ['val', 'all']:
        evaluate_model(net, X_val, y_val, "Validation", args.verbose_decimals)

    if args.mode in ['test', 'all']:
        evaluate_model(net, X_test, y_test, "Test", args.verbose_decimals)

    # Generar gráficas si se solicita
    if args.plot:
        plot_results(net, X_train, y_train, X_val, y_val, X_test, y_test,
                    data_dict['dataset'], args.verbose_decimals)

    print(f"\n==== Completado ====")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
