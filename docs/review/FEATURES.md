# Features

Este documento tracks all implemented features in the project.

## Confusion Matrix

**Estado**: Completo
**Fecha de implementación**: 2026-04-23
**Descripción**: Generación de matrices de confusión con métricas derivadas para tareas de clasificación. Soporta clasificación binaria y multi-clase, así como múltiples salidas (una matriz por columna).

### Concepto

La matriz de confusión es una herramienta fundamental para evaluar el rendimiento de modelos de clasificación. Es una tabla que muestra el desempeño del modelo comparando las predicciones con los valores reales.

**Estructura de la Matriz:**
```
              Predicción
              Clase 0  Clase 1
Real Clase 0  [  TP      FP   ]
Real Clase 1  [  FN      TN   ]
```

- **TP (True Positive)**: Predijo clase 1 correctamente
- **TN (True Negative)**: Predijo clase 0 correctamente
- **FP (False Positive)**: Predijo clase 1 pero era clase 0
- **FN (False Negative)**: Predijo clase 0 pero era clase 1

### Métricas Derivadas

A partir de la matriz de confusión se calculan varias métricas importantes:

**Precision (Precisión):**
```
Precision = TP / (TP + FP)
```
- De todos los que predije como clase 1, ¿cuántos eran realmente clase 1?
- Rango: 0 a 1 (mayor es mejor)

**Recall (Sensibilidad):**
```
Recall = TP / (TP + FN)
```
- De todos los que eran realmente clase 1, ¿cuántos predije correctamente?
- Rango: 0 a 1 (mayor es mejor)

**F1-Score:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- Media armónica de precision y recall
- Útil cuando hay desbalance de clases
- Rango: 0 a 1 (mayor es mejor)

**Accuracy (Exactitud):**
```
Accuracy = (TP + TN) / Total
```
- Proporción de predicciones correctas
- Rango: 0 a 1 (mayor es mejor)

### Características de Implementación

**Discretización Automática:**
Las redes neuronales devuelven valores continuos. La implementación incluye discretización automática:
- **Binario**: Umbral 0.5 (≥0.5 → clase 1, <0.5 → clase 0)
- **Multi-clase**: Asignación a la clase más cercana

**Soporte para Múltiples Salidas:**
- Si `y.shape[1] == 1`: Retorna un solo `ConfusionMatrixResult`
- Si `y.shape[1] > 1`: Retorna `Dict[int, ConfusionMatrixResult]` (una por columna)

**Matrices Normalizadas:**
- `matrix_normalized_row`: Normalizada por fila (recall por clase)
- `matrix_normalized_col`: Normalizada por columna (precision por clase)

**Promedios:**
- `macro_avg`: Promedio macro de precision, recall, f1 (sin ponderar)
- `weighted_avg`: Promedio ponderado por support de precision, recall, f1

### Endpoints / Entradas

- Método API: `NeuralNetwork.confusion_matrix(y_true, y_pred=None, X=None)`
- Método Evaluator: `Evaluator.confusion_matrix(y_true, y_pred)`
- Clase Calculadora: `ConfusionMatrixCalculator.compute(y_true, y_pred, labels=None, discretize=True)`

### Lógica Principal

1. Valida formas de y_true y y_pred
2. Discretiza predicciones continuas a clases discretas (si discretize=True)
3. Calcula matriz de confusión con valores absolutos
4. Genera matrices normalizadas por fila (recall) y columna (precision)
5. Calcula métricas por clase: precision, recall, F1-score, support
6. Calcula accuracy global
7. Calcula promedios macro y ponderados
8. Soporta múltiples salidas generando una matriz por columna

### Dependencias

- `numpy` para cálculos matriciales
- `api.core.results.ConfusionMatrixResult` para resultado tipado
- `src.evaluation.Evaluator` para integración con evaluación existente

### Archivos Principales

- `src/evaluation/confusion_matrix.py` — Implementación de `ConfusionMatrixCalculator` y `ConfusionMatrixResult`
- `api/core/results.py` — Dataclass `ConfusionMatrixResult` para resultado tipado
- `src/evaluation/evaluator.py` — Método `confusion_matrix()` en `Evaluator`
- `api/neural_network.py` — Método `confusion_matrix()` en API pública
- `src/evaluation/__init__.py` — Exportación de nuevas clases

### Scripts de Prueba

- `scripts/test_xor_confusion.py` — Prueba con XOR (50 train, 20 test) usando RBF
- `scripts/test_confusion_errors.py` — Prueba con círculos concéntricos para generar errores

### Uso Ejemplo

```python
# Una sola salida
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)
net.train(X_train, y_train)
result = net.confusion_matrix(y_test, X=X_test)
print(result.matrix)  # Matriz de confusión
print(result.precision)  # Precision por clase
print(result.recall)  # Recall por clase
print(result.accuracy)  # Accuracy global

# Múltiples salidas
results = net.confusion_matrix(y_test_multi, X=X_test)
for output_idx, result in results.items():
    print(f"Salida {output_idx}:")
    print(f"  Accuracy: {result.accuracy}")
    print(f"  F1-score: {result.f1_score}")
```

### Documentación

- Guía técnica: `docs/guide.md` — Sección "Matriz de Confusión"
- Referencia API: `docs/reference.md` — Método `confusion_matrix()` y dataclass `ConfusionMatrixResult`
