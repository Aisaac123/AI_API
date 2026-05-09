# Guía Completa del Proceso de Entrenamiento RBF Manual

Este documento explica en detalle cada paso del proceso de entrenamiento manual de una Red Neuronal de Base Radial (RBF) implementado en los notebooks de este directorio.

---

## Índice
1. [Carga y Preprocesamiento de Datos](#1-carga-y-preprocesamiento-de-datos)
2. [Separación del Dataset (70/15/15)](#2-separación-del-dataset-701515)
3. [Configuración de la Red RBF](#3-configuración-de-la-red-rbf)
4. [Inicialización de Centros Radiales](#4-inicialización-de-centros-radiales)
5. [Cálculo de Distancias Euclidianas](#5-cálculo-de-distancias-euclidianas)
6. [Aplicación de Función de Activación](#6-aplicación-de-función-de-activación)
7. [Construcción de Matriz de Interpolación](#7-construcción-de-matriz-de-interpolación)
8. [Resolución de Pesos (Pseudoinversa)](#8-resolución-de-pesos-pseudoinversa)
9. [Simulación (Cálculo de Salidas)](#9-simulación-cálculo-de-salidas)
10. [Cálculo de Errores](#10-cálculo-de-errores)
11. [Verificación de Convergencia](#11-verificación-de-convergencia)
12. [Reentrenamiento Automático](#12-reentrenamiento-automático)
13. [Matrices de Confusión](#13-matrices-de-confusión)
14. [Evaluación Final](#14-evaluación-final)

---

## 1. Carga y Preprocesamiento de Datos

### Objetivo
Cargar el dataset desde un archivo JSON y extraer las matrices de entrada (X) y salida deseada (YD).

### Proceso
```python
# Cargar el archivo JSON
with open('../jsons/dataset_rbf_1.json', 'r') as f:
    data_dict = json.load(f)

# Extraer características y etiquetas
features = data_dict['features']
X = np.array(data_dict['data'][features])
YD = np.array(data_dict['data']['output']).reshape(-1, 1)
```

### Explicación
- **JSON Loading**: Se lee el archivo JSON que contiene la estructura del dataset.
- **Features**: Se obtienen los nombres de las columnas de entrada (características).
- **X (Entradas)**: Matriz de tamaño (n_patrones, n_entradas) que contiene los valores de entrada de cada patrón.
- **YD (Salidas Deseadas)**: Vector columna de tamaño (n_patrones, n_salidas) que contiene los valores objetivo que la red debe aprender.

---

## 2. Separación del Dataset (70/15/15)

### Objetivo
Dividir el dataset en tres conjuntos: entrenamiento (70%), validación (15%) y prueba (15%).

### Proceso
```python
# Mezclar aleatoriamente los datos
np.random.seed(42)
indices = np.random.permutation(n_patrones)

# Calcular índices de separación
train_end = int(0.70 * n_patrones)
val_end = int(0.85 * n_patrones)  # 70% + 15%

# Separar índices
train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

# Crear subconjuntos
X_train, YD_train = X[train_idx], YD[train_idx]
X_val, YD_val = X[val_idx], YD[val_idx]
X_test, YD_test = X[test_idx], YD[test_idx]
```

### Explicación
- **Mezcla Aleatoria**: Se permutan los índices aleatoriamente para asegurar que los datos estén distribuidos uniformemente.
- **Train (70%)**: Conjunto utilizado para entrenar el modelo y ajustar los pesos.
- **Validation (15%)**: Conjunto utilizado para monitorear el rendimiento durante el entrenamiento y evitar overfitting.
- **Test (15%)**: Conjunto utilizado para evaluar el rendimiento final del modelo en datos nunca vistos.

---

## 3. Configuración de la Red RBF

### Objetivo
Definir los hiperparámetros de la red RBF.

### Parámetros
```python
n_entradas = X_train.shape[1]      # Número de características de entrada
n_salidas = YD_train.shape[1]      # Número de salidas (clases)
n_patrones = X_train.shape[0]      # Número de patrones de entrenamiento
num_centros = 30                    # Número inicial de centros radiales (neuronas ocultas)
error_optimo = 0.03                 # Error objetivo para convergencia
max_iteraciones = 15                # Máximo de reentrenamientos (para referencia)
incremento_centros = 10             # Incremento de centros para reentrenamiento manual
```

### Valores Recomendados por Dataset
- **Dataset 1 (Binario)**: num_centros=10-15, error_optimo=0.03-0.05
- **Dataset 2 (3 Clases)**: num_centros=30, error_optimo=0.03
- **Dataset 3 (4 Clases)**: num_centros=40-60, error_optimo=0.02-0.04

### Explicación
- **n_entradas**: Determinado por el número de características del dataset.
- **n_salidas**: 1 para clasificación binaria, N para clasificación multiclasse.
- **n_centros**: Número de neuronas en la capa oculta RBF. Más centros = mayor capacidad de representación pero mayor riesgo de overfitting.
- **error_optimo**: Umbral de error debajo del cual se considera que el modelo ha convergido.
- **max_iteraciones**: Límite de reentrenamientos automáticos para evitar bucles infinitos.

---

## 4. Inicialización de Centros Radiales

### Objetivo
Inicializar los centros radiales (R) que serán los "prototipos" de las neuronas RBF.

### Proceso
```python
# Calcular rango de valores por cada entrada
X_min = np.min(X_train, axis=0)
X_max = np.max(X_train, axis=0)

# Inicializar centros aleatoriamente dentro del rango
R = np.random.uniform(X_min, X_max, (n_centros, n_entradas))
```

### Explicación
- **Rango de Valores**: Se calcula el mínimo y máximo de cada característica para asegurar que los centros estén dentro del espacio de datos.
- **Inicialización Aleatoria**: Cada centro se inicializa con valores aleatorios uniformemente distribuidos entre el mínimo y máximo de cada característica.
- **Centros (R)**: Matriz de tamaño (n_centros, n_entradas) donde cada fila es un centro radial.

---

## 5. Cálculo de Distancias Euclidianas

### Objetivo
Calcular la distancia euclidiana entre cada patrón de entrada y cada centro radial.

### Fórmula
$$D_{ij} = \sqrt{\sum_{k=1}^{n\_entradas} (X_{ik} - R_{jk})^2}$$

### Proceso
```python
# Calcular matriz de distancias
D = np.zeros((n_patrones, n_centros))
for i in range(n_patrones):
    for j in range(n_centros):
        # Distancia euclidiana manual
        diff = X_train[i] - R[j]
        D[i, j] = np.sqrt(np.sum(diff ** 2))
```

### Explicación
- **D (Matriz de Distancias)**: Matriz de tamaño (n_patrones, n_centros) donde D[i,j] es la distancia entre el patrón i y el centro j.
- **Distancia Euclidiana**: Mide qué tan "cerca" está un patrón de un centro radial en el espacio de características.
- **Interpretación**: Valores más bajos indican que el patrón es más similar al centro.

---

## 6. Aplicación de Función de Activación

### Objetivo
Aplicar la función de activación RBF (Thin Plate Spline) a las distancias calculadas.

### Fórmula Thin Plate Spline
$$FA = \Omega^2 \cdot \ln(\Omega)$$

Donde Ω es la distancia euclidiana.

### Proceso
```python
def thin_plate_spline(omega):
    """Función de activación Thin Plate Spline"""
    return omega ** 2 * np.log(omega)

# Aplicar función de activación
FA = thin_plate_spline(D)
```

### Explicación
- **Thin Plate Spline**: Función de activación RBF que crece con el cuadrado de la distancia, suave y diferenciable.
- **FA (Matriz de Activaciones)**: Matriz de tamaño (n_patrones, n_centros) con los valores de activación.
- **Interpretación**: Valores más altos indican que el patrón está más lejos del centro (a diferencia de la Gaussiana donde es lo contrario).

---

## 7. Construcción de Matriz de Interpolación

### Objetivo
Construir la matriz de diseño A que incluye las activaciones y un término de bias.

### Proceso
```python
# Construir matriz A = [1, FA]
A = np.column_stack([np.ones(n_patrones), FA])
```

### Explicación
- **A (Matriz de Diseño)**: Matriz de tamaño (n_patrones, n_centros + 1).
- **Columna de Unos**: Primer columna de A es un vector de unos para el bias (W0).
- **FA**: Columnas restantes son las activaciones de cada centro radial.
- **Ecuación Lineal**: El sistema A·W = YD representa un sistema de ecuaciones lineales donde W son los pesos desconocidos.

---

## 8. Resolución de Pesos (Pseudoinversa)

### Objetivo
Calcular los pesos de salida W usando la pseudoinversa de Moore-Penrose.

### Fórmula
$$W = A^+ \cdot YD$$

Donde A⁺ es la pseudoinversa de A.

### Proceso
```python
# Calcular pseudoinversa de A
A_pinv = np.linalg.pinv(A)

# Resolver pesos: W = A^+ * YD
W = np.dot(A_pinv, YD_train)
```

### Explicación
- **Pseudoinversa**: Generaliza la inversa de una matriz para matrices no cuadradas o singulares.
- **W (Pesos de Salida)**: Vector columna de tamaño (n_centros + 1, n_salidas).
- **W0**: Primer peso es el bias (intercepto).
- **W1, W2, ...**: Pesos asociados a cada centro radial.

---

## 9. Simulación (Cálculo de Salidas)

### Objetivo
Calcular las salidas de la red (YR) para un conjunto de datos.

### Proceso
```python
# Para conjunto de entrenamiento
A_train = np.column_stack([np.ones(len(X_train)), 
                            thin_plate_spline(calcular_distancias(X_train, R))])
YR_train = np.dot(A_train, W)

# Para validación
A_val = np.column_stack([np.ones(len(X_val)), 
                         thin_plate_spline(calcular_distancias(X_val, R))])
YR_val = np.dot(A_val, W)

# Para test
A_test = np.column_stack([np.ones(len(X_test)), 
                          thin_plate_spline(calcular_distancias(X_test, R))])
YR_test = np.dot(A_test, W)
```

### Explicación
- **Simulación**: Proceso de propagación hacia adelante (forward pass).
- **YR (Salidas de la Red)**: Predicciones de la red para cada patrón.
- **Proceso**: Se calculan distancias → activaciones → matriz de diseño → multiplicación por pesos.

---

## 10. Cálculo de Errores

### Objetivo
Calcular el error entre las salidas deseadas (YD) y las calculadas (YR).

### Métricas
```python
# Error por patrón (EL)
EL = YD - YR

# Error General (EG) - Promedio de errores absolutos
EG = np.mean(np.abs(EL))

# Error Cuadrático Medio (MSE)
MSE = np.mean((YD - YR) ** 2)
```

### Explicación
- **EL (Error Local)**: Error individual de cada patrón.
- **EG (Error General)**: Métrica global de rendimiento, promedio de errores absolutos.
- **MSE**: Métrica más sensible a errores grandes, útil para comparación.

---

## 11. Verificación de Convergencia

### Objetivo
Determinar si el modelo ha alcanzado el nivel de error deseado.

### Proceso
```python
if EG <= error_optimo:
    print("✓ CONVERGE: El error está dentro del límite aceptable")
    converge = True
else:
    print("✗ NO CONVERGE: El error excede el límite aceptable")
    converge = False
```

### Explicación
- **Convergencia**: El modelo converge cuando el error de entrenamiento es menor o igual al error óptimo.
- **No Convergencia**: Si el error es mayor, se debe reentrenar con diferentes parámetros.

---

## 12. Ajuste Manual de Parámetros y Reentrenamiento

### Objetivo
Ajustar manualmente los hiperparámetros si el modelo no converge y reentrenar con configuración mejorada.

### Proceso de Ajuste Manual
```python
# Sección opcional para ajuste manual
# num_centros = 30
# error_optimo = 0.03
# max_iteraciones = 15
# incremento_centros = 10
```

### Estrategia de Reentrenamiento
```python
# Guardar métricas del entrenamiento anterior
EG_anterior = mejor_EG
centros_anteriores = mejor_R.shape[0]

# Configuración mejorada para reentrenamiento
num_centros_re = num_centros * 3  # Multiplicador de centros (x2, x3, etc.)
error_optimo_re = error_optimo * 0.33  # Error más estricto (x0.33, x0.5, etc.)

# Reentrenar con nueva configuración
np.random.seed(100)  # Semilla diferente para nueva inicialización
R_re = inicializar_centros(X_train, num_centros_re)
D_re = calcular_distancias(X_train, R_re)
A_re = calcular_matriz_interpolacion(D_re)
W_re = resolver_pesos(A_re, YD_train)

# Evaluar reentrenamiento
YR_train_re = simular_red(X_train, R_re, W_re)
EG_re = np.mean(np.abs(YD_train - YR_train_re))

converge_re = EG_re <= error_optimo_re
```

### Explicación
- **Ajuste Manual**: Permite modificar parámetros antes del reentrenamiento sin ejecutar todo el notebook desde el inicio.
- **Multiplicador de Centros**: Aumenta el número de centros (x2, x3) para mejorar la capacidad de representación.
- **Error Más Estricto**: Reduce el error objetivo (x0.33, x0.5) para mayor precisión.
- **Nueva Semilla**: Reinicializa con semilla diferente para evitar mínimos locales.
- **Comparación**: Se comparan métricas entre entrenamiento y reentrenamiento para verificar mejoras.

---

## 13. Matrices de Confusión

### Objetivo
Evaluar el rendimiento de clasificación del modelo.

### Proceso (Multiclase)
```python
def calcular_matriz_confusion(YD, YR):
    """Calcula matriz de confusión para clasificación multiclase"""
    # Redondear y convertir a enteros
    YD_c = np.round(YD.flatten()).astype(int)
    YR_c = np.round(YR.flatten()).astype(int)
    
    # Determinar número de clases
    n_clases = max(np.max(YD_c), np.max(YR_c)) + 1
    
    # Construir matriz de confusión
    mat = np.zeros((n_clases, n_clases), dtype=int)
    for i in range(len(YD_c)):
        mat[YD_c[i], YR_c[i]] += 1
    
    # Calcular accuracy global
    accuracy = np.sum(np.diag(mat)) / np.sum(mat) if np.sum(mat) > 0 else 0
    
    # Calcular métricas por clase y promediar (macro average)
    precisions, recalls, specificities, f1s = [], [], [], []
    for c in range(n_clases):
        TP = mat[c, c]
        FP = np.sum(mat[:, c]) - TP
        FN = np.sum(mat[c, :]) - TP
        TN = np.sum(mat) - TP - FP - FN
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1   = 2*(prec*rec)/(prec+rec) if (prec+rec) > 0 else 0
        precisions.append(prec); recalls.append(rec)
        specificities.append(spec); f1s.append(f1)
    
    return mat, accuracy, np.mean(precisions), np.mean(recalls), np.mean(specificities), np.mean(f1s)
```

### Explicación
- **Matriz de Confusión**: Matriz de tamaño (n_clases, n_clases) donde mat[i,j] es el número de patrones de clase i predichos como clase j.
- **TP (True Positive)**: Predicción correcta de la clase c.
- **TN (True Negative)**: Predicción correcta de clases diferentes a c.
- **FP (False Positive)**: Patrones de otras clases predichos como clase c.
- **FN (False Negative)**: Patrones de clase c predichos como otras clases.
- **Accuracy**: Proporción de predicciones correctas (global).
- **Precision (macro)**: Promedio de precision por clase.
- **Recall (macro)**: Promedio de recall por clase.
- **Specificity (macro)**: Promedio de specificity por clase.
- **F1-Score (macro)**: Media armónica promedio de precision y recall por clase.

---

## 14. Evaluación Final

### Objetivo
Evaluar el modelo en los tres conjuntos (train, val, test) para obtener métricas completas.

### Proceso
```python
# Evaluación en Train
conf_train, acc_train, prec_train, rec_train, f1_train = calcular_matriz_confusion(YD_train, YR_train)

# Evaluación en Validation
conf_val, acc_val, prec_val, rec_val, f1_val = calcular_matriz_confusion(YD_val, YR_val)

# Evaluación en Test
conf_test, acc_test, prec_test, rec_test, f1_test = calcular_matriz_confusion(YD_test, YR_test)

# Imprimir resumen
print("=== EVALUACIÓN FINAL ===")
print(f"Train - Accuracy: {acc_train:.4f}, F1: {f1_train:.4f}")
print(f"Val   - Accuracy: {acc_val:.4f}, F1: {f1_val:.4f}")
print(f"Test  - Accuracy: {acc_test:.4f}, F1: {f1_test:.4f}")
```

### Explicación
- **Train**: Mide qué tan bien el modelo aprendió los datos de entrenamiento.
- **Validation**: Mide la capacidad de generalización durante el entrenamiento.
- **Test**: Mide el rendimiento final en datos nunca vistos.
- **Comparación**: Si train >> test, hay overfitting. Si train ≈ test ≈ val, el modelo generaliza bien.

---

## Notas Adicionales

### Ajuste de Hiperparámetros
Si el modelo no converge después del reentrenamiento automático, puedes ajustar manualmente:
- **n_centros**: Aumentar si el modelo no converge, disminuir si hay overfitting.
- **error_optimo**: Ajustar según la precisión requerida.
- **Función de activación**: Probar con Gaussiana, Multicuádrica, etc.

### Multiclasse
Para datasets con más de 2 clases (como Dataset 2 con 3 clases y Dataset 3 con 4 clases):
- La matriz de confusión será de tamaño (n_clases, n_clases).
- Las salidas son valores enteros (0, 1, 2, ..., n_clases-1) sin one-hot encoding.
- Las métricas se calculan por clase y se promedian (macro average): precision, recall, specificity, F1-score.
- La función `calcular_matriz_confusion` detecta automáticamente el número de clases desde los datos.

### Visualizaciones
Los notebooks incluyen:
- Gráficos de YD vs YR para visualizar el ajuste.
- Gráficos de errores por patrón.
- Heatmaps de matrices de confusión.
