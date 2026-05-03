# Flujo Completo de Red RBF

Este documento describe el flujo completo de implementación y uso de redes RBF en el proyecto, desde la configuración de parámetros hasta la evaluación del modelo.

## Tabla de Contenidos

1. [Parámetros de Activación/Entrenamiento](#1-parámetros-de-activaciónentrenamiento)
2. [Inicialización de Centros Radiales](#2-inicialización-de-centros-radiales)
3. [Entrenamiento del Modelo](#3-entrenamiento-del-modelo)
4. [Construcción de la Matriz de Interpolación](#4-construcción-de-la-matriz-de-interpolación)
5. [Cálculo de la Matriz de Pesos](#5-cálculo-de-la-matriz-de-pesos)
6. [Evaluación del Modelo (Simulación)](#6-evaluación-del-modelo-simulación)
7. [Métricas de Evaluación / Matriz de Confusión](#7-métricas-de-evaluación--matriz-de-confusión)
8. [Scripts de Entrenamiento con Flags](#8-scripts-de-entrenamiento-con-flags)

---

## 1. Parámetros de Activación/Entrenamiento

### Descripción

Configurar la arquitectura y parámetros clave del modelo:
- **Número de neuronas**: Determina la capacidad de la red (mínimo recomendado igual al número de entradas)
- **Tipo de función de activación**: Usualmente gaussiana, pero soporta múltiples opciones
- **Centros radiales**: Valores aleatorios en el intervalo de los datos de entrenamiento
- **Dimensión de centros**: Cada neurona tiene un centro radial con dimensión igual al número de entradas
- **Cantidad de centros**: Debe coincidir con el número de neuronas
- **Estructura de red**: Unicapa (capa oculta de neuronas RBF + capa de salida lineal)

### Código Profundo - Configuración

**Path:** `src/models/rbf/config.py`

```python
@dataclass
class RBFConfig:
    """Clase de configuración para parámetros de red RBF."""
    
    n_centers: int = 10
    """Número de centros de función de base radial (neuronas en capa oculta)"""
    
    sigma: float = 1.0
    """Parámetro de ancho para la función de activación (dispersión de cada RBF)"""
    
    activation: ActivationFunction = None
    """Función de activación a usar (por defecto Gaussiana si no se especifica)"""
    
    regularization: float = 0.0
    """Parámetro de regularización para la pseudoinversa (agrega matriz identidad * λ)"""
    
    use_bias: bool = True
    """Si incluir término de bias en la capa de salida"""
    
    random_state: int = None
    """Semilla aleatoria para reproducibilidad"""
```

**Validación de parámetros:**

```python
def validate(self) -> None:
    """Validar los parámetros de configuración."""
    if self.n_centers <= 0:
        raise InvalidConfigError(f"n_centers debe ser positivo, se obtuvo {self.n_centers}")
    
    if self.sigma <= 0:
        raise InvalidConfigError(f"sigma debe ser positivo, se obtuvo {self.sigma}")
    
    if self.regularization < 0:
        raise InvalidConfigError(f"regularization debe ser no negativo, se obtuvo {self.regularization}")
```

### Código API - Configuración

**Path:** `api/neural_network.py`

```python
# Método 1: Parámetros individuales
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=20,           # Número de neuronas/centros
    sigma=0.5,              # Parámetro de ancho
    activation_rbf='gaussian',  # Función de activación
    regularization=0.01,    # Regularización
    use_bias=True,          # Usar bias
    random_state=42         # Semilla
)

# Método 2: Objeto de configuración
from api.core.config import NeuralNetworkConfig

config = NeuralNetworkConfig(
    n_centers=20,
    sigma=0.5,
    activation_rbf='gaussian',
    regularization=0.01
)

net = NeuralNetwork(model_type=ModelType.RBF, config=config)
```

### Funciones de Activación Disponibles

**Path:** `src/core/activation.py`

| Función | Descripción | Parámetro API |
|---------|-------------|---------------|
| `GaussianActivation` | φ(d) = exp(-d²/2σ²) | `'gaussian'` |
| `MultiquadraticActivation` | φ(d) = sqrt(d² + σ²) | `'multiquadratic'` |
| `InverseMultiquadraticActivation` | φ(d) = 1/sqrt(d² + σ²) | `'inverse_multiquadratic'` |
| `ThinPlateActivation` | φ(d) = d² ln(d) (forma original) | `'thin_plate'` |
| `ThinPlateLog10Activation` | φ(d) = d² log₁₀(d) (variante) | `'thin_plate_log10'` |

### Output Ejemplo

```python
# Configuración válida
config = RBFConfig(n_centers=20, sigma=0.5)
print(config.to_dict())
# Output:
# {
#     'n_centers': 20,
#     'sigma': 0.5,
#     'activation': 'GaussianActivation',
#     'regularization': 0.0,
#     'use_bias': True,
#     'random_state': None
# }

# Configuración inválida (lanza excepción)
config = RBFConfig(n_centers=-5)  # InvalidConfigError
```

---

## 2. Inicialización de Centros Radiales

### Descripción

Asignar los centros radiales de forma aleatoria o mediante alguna estrategia (ej. selección de patrones), asegurando que cada centro tenga la misma dimensión que las entradas.

**Estrategias disponibles:**
1. **RandomInitializer**: Muestreo aleatorio de datos de entrenamiento
2. **KMeansInitializer**: Clustering k-means para encontrar centros representativos

### Código Profundo - Inicialización

**Path:** `src/training/initializer.py`

#### Estrategia RandomInitializer

```python
class RandomInitializer(BaseCenterInitializer):
    """Estrategia de inicialización aleatoria."""
    
    def initialize(self, X: np.ndarray, n_centers: int) -> np.ndarray:
        """
        Inicializar centros muestreando aleatoriamente de los datos de entrada.
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            n_centers: Número de centros a inicializar
            
        Returns:
            Matriz de centros de forma (n_centers, n_features)
        """
        n_samples = X.shape[0]
        n_centers = min(n_centers, n_samples)
        
        # Seleccionar índices aleatorios sin reemplazo
        indices = np.random.choice(n_samples, n_centers, replace=False)
        centers = X[indices]
        
        return centers
```

#### Estrategia KMeansInitializer

```python
class KMeansInitializer(BaseCenterInitializer):
    """Estrategia de inicialización K-means."""
    
    def initialize(self, X: np.ndarray, n_centers: int) -> np.ndarray:
        """
        Inicializar centros usando clustering k-means.
        
        Algoritmo:
        1. Inicializar centroides aleatoriamente
        2. Asignar cada punto al centroide más cercano
        3. Recalcular centroides como media de puntos asignados
        4. Repetir hasta convergencia
        
        Returns:
            Matriz de centros de forma (n_centers, n_features)
        """
        n_samples = X.shape[0]
        n_centers = min(n_centers, n_samples)
        
        # Inicializar centroides aleatoriamente
        indices = np.random.choice(n_samples, n_centers, replace=False)
        centroids = X[indices].copy()
        
        # Ejecutar iteraciones k-means
        for iteration in range(self.max_iterations):
            # Asignar cada muestra al centroide más cercano
            distances = self._compute_distances(X, centroids)
            assignments = np.argmin(distances, axis=1)
            
            # Calcular nuevos centroides: mu_k = (1/|C_k|) * sum_{x_i in C_k} x_i
            new_centroids = np.zeros_like(centroids)
            for k in range(n_centers):
                mask = assignments == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # Verificar convergencia
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            if centroid_shift < self.tolerance:
                break
            
            centroids = new_centroids
        
        return centroids
```

### Código API - Inicialización

**Path:** `api/neural_network.py`

```python
# Método 1: Inicialización automática (random por defecto)
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=20,
    initializer='random'  # o 'kmeans'
)
net.train(X_train, y_train)

# Método 2: Centros manuales
mis_centros = np.array([[0.2, 0.3], [0.5, 0.7], [0.8, 0.1]])
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=3)
net.train(X_train, y_train, centers=mis_centros)
```

### Código Profundo - Aplicación de Centros

**Path:** `src/models/rbf/network.py`

```python
def fit(self, X: np.ndarray, y: np.ndarray, centers: np.ndarray = None) -> None:
    """Entrenar la red RBF."""
    
    # Usar centros proporcionados o centros existentes
    if centers is not None:
        self.centers = centers
        self.config.n_centers = centers.shape[0]
    elif self.centers is None:
        # Si no se proporcionan centros, muestrear aleatoriamente
        n_samples = X.shape[0]
        n_centers = min(self.config.n_centers, n_samples)
        indices = np.random.choice(n_samples, n_centers, replace=False)
        self.centers = X[indices]
    
    # Crear la capa RBF con los centros
    self.rbflayer = RBFLayer(
        centers=self.centers,
        activation=self.config.activation,
        sigma=self.config.sigma
    )
```

### Output Ejemplo

```python
# Datos de ejemplo
X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])

# Inicialización random
initializer = RandomInitializer()
centers = initializer.initialize(X, n_centers=2)
print(f"Centros random: {centers}")
# Output:
# Centros random: [[0.3 0.4]
#                  [0.1 0.2]]

# Inicialización k-means
initializer = KMeansInitializer(max_iterations=100)
centers = initializer.initialize(X, n_centers=2)
print(f"Centros k-means: {centers}")
# Output:
# Centros k-means: [[0.2 0.3]
#                   [0.6 0.7]]
```

---

## 3. Entrenamiento del Modelo

### Descripción

Calcular la respuesta de la red:
1. Medir la distancia (usualmente euclidiana) entre cada patrón de entrada y cada centro radial
2. Aplicar la función de activación sobre dichas distancias para obtener la salida de las neuronas ocultas

### Código Profundo - Cálculo de Distancias

**Path:** `src/core/distance.py`

```python
def euclidean_distance_matrix(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Calcular matriz de distancias euclidianas.
    
    Fórmula: d(x, c) = sqrt(sum((x_i - c_i)^2))
    
    Args:
        X: Matriz de entrada de forma (n_samples, n_features)
        centers: Matriz de centros de forma (n_centers, n_features)
        
    Returns:
        Matriz de distancias de forma (n_samples, n_centers)
    """
    # Expandir dimensiones para broadcasting
    X_expanded = X[:, np.newaxis, :]      # (n_samples, 1, n_features)
    centers_expanded = centers[np.newaxis, :, :]  # (1, n_centers, n_features)
    
    # Calcular diferencias al cuadrado
    squared_diff = (X_expanded - centers_expanded) ** 2
    
    # Sumar sobre características y tomar raíz cuadrada
    distances = np.sqrt(np.sum(squared_diff, axis=2))
    
    return distances
```

### Código Profundo - Capa RBF Forward

**Path:** `src/models/rbf/layer.py`

```python
class RBFLayer(BaseLayer):
    """Capa oculta de Función de Base Radial (RBF)."""
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calcular el pase hacia adelante de la capa RBF.
        
        La salida se calcula como: Y = phi(d(X, C), sigma)
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            
        Returns:
            Matriz de activación de forma (n_samples, n_centers)
        """
        # Paso 1: Calcular distancias euclidianas
        distances = euclidean_distance_matrix(X, self.centers)
        
        # Paso 2: Aplicar función de activación
        activations = self.activation.compute(distances, self.sigma)
        
        return activations
```

### Código Profundo - Función de Activación Gaussiana

**Path:** `src/core/activation.py`

```python
class GaussianActivation(ActivationFunction):
    """Función de activación gaussiana."""
    
    def compute(self, distances: np.ndarray, sigma: float) -> np.ndarray:
        """
        Calcular activación gaussiana.
        
        Fórmula: φ(d) = exp(-d² / (2σ²))
        
        Args:
            distances: Matriz de distancias de forma (n_samples, n_centers)
            sigma: Parámetro de ancho
            
        Returns:
            Matriz de activaciones de forma (n_samples, n_centers)
        """
        return np.exp(-(distances ** 2) / (2 * sigma ** 2))
```

### Código API - Entrenamiento

**Path:** `api/neural_network.py`

```python
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=20,
    sigma=0.5
)

# Entrenar
result = net.train(X_train, y_train, verbose=True)

# Resultado tipado
print(f"Épocas: {result.epochs}")
print(f"Error final: {result.final_error}")
print(f"Tiempo: {result.training_time}")
```

### Output Ejemplo

```python
# Datos de ejemplo
X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
centers = np.array([[0.2, 0.3], [0.4, 0.5]])

# Calcular distancias
distances = euclidean_distance_matrix(X, centers)
print("Matriz de distancias:")
print(distances)
# Output:
# Matriz de distancias:
# [[0.14142136 0.2236068 ]
#  [0.14142136 0.14142136]
#  [0.42426407 0.14142136]]

# Aplicar activación gaussiana (sigma=0.5)
activation = GaussianActivation()
activations = activation.compute(distances, sigma=0.5)
print("Matriz de activaciones:")
print(activations)
# Output:
# Matriz de activaciones:
# [[0.92311635 0.81058425]
#  [0.92311635 0.92311635]
#  [0.60653066 0.92311635]]
```

---

## 4. Construcción de la Matriz de Interpolación

### Descripción

Generar la matriz Φ (phi):
- **Filas**: número de patrones (n_samples)
- **Columnas**: número de neuronas RBF (n_centers)
- **Cada elemento**: corresponde a la activación de una neurona para un patrón dado

La matriz de diseño Φ es el resultado de aplicar la función de activación a todas las distancias.

### Código Profundo - Matriz de Diseño

**Path:** `src/models/rbf/solver.py`

```python
def compute_design_matrix(X: np.ndarray, centers: np.ndarray, 
                        activation: ActivationFunction, sigma: float) -> np.ndarray:
    """
    Calcular la matriz de diseño (matriz de interpolación) Φ.
    
    Fórmula: Φ_ij = φ(||x_i - c_j||, σ)
    
    donde:
    - Φ_ij: activación de la neurona j para el patrón i
    - x_i: i-ésimo patrón de entrada
    - c_j: j-ésimo centro radial
    - φ: función de activación
    - σ: parámetro de ancho
    
    Args:
        X: Matriz de entrada de forma (n_samples, n_features)
        centers: Matriz de centros de forma (n_centers, n_features)
        activation: Función de activación a usar
        sigma: Parámetro de ancho
        
    Returns:
        Matriz de diseño Φ de forma (n_samples, n_centers)
    """
    # Calcular distancias euclidianas
    distances = euclidean_distance_matrix(X, centers)
    
    # Aplicar función de activación
    Phi = activation.compute(distances, sigma)
    
    return Phi
```

### Código Profundo - Aplicación en Entrenamiento

**Path:** `src/models/rbf/network.py`

```python
def fit(self, X: np.ndarray, y: np.ndarray, centers: np.ndarray = None) -> None:
    """Entrenar la red RBF."""
    
    # ... inicialización de centros ...
    
    # Crear la capa RBF
    self.rbflayer = RBFLayer(
        centers=self.centers,
        activation=self.config.activation,
        sigma=self.config.sigma
    )
    
    # Calcular matriz de diseño (activaciones de capa oculta)
    Phi = self.rbflayer.forward(X)
    
    # Agregar columna de bias si está configurado
    if self.config.use_bias:
        Phi_bias = np.column_stack([Phi, np.ones(Phi.shape[0])])
    else:
        Phi_bias = Phi
    
    # Phi_bias tiene forma (n_samples, n_centers + 1) si usa bias
    # o (n_samples, n_centers) si no usa bias
```

### Output Ejemplo

```python
# Datos de ejemplo
X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
centers = np.array([[0.2, 0.3], [0.4, 0.5]])

# Calcular matriz de diseño
Phi = compute_design_matrix(X, centers, GaussianActivation(), sigma=0.5)
print("Matriz de diseño Φ:")
print(Phi)
print(f"Forma: {Phi.shape}")  # (3 patrones, 2 centros)
# Output:
# Matriz de diseño Φ:
# [[0.92311635 0.81058425]
#  [0.92311635 0.92311635]
#  [0.60653066 0.92311635]]
# Forma: (3, 2)

# Con bias
Phi_bias = np.column_stack([Phi, np.ones(Phi.shape[0])])
print("Matriz de diseño con bias:")
print(Phi_bias)
print(f"Forma: {Phi_bias.shape}")  # (3, 3)
# Output:
# Matriz de diseño con bias:
# [[0.92311635 0.81058425 1.        ]
#  [0.92311635 0.92311635 1.        ]
#  [0.60653066 0.92311635 1.        ]]
# Forma: (3, 3)
```

---

## 5. Cálculo de la Matriz de Pesos

### Descripción

Determinar los pesos de salida resolviendo el sistema ΦW = Y.

**Métodos disponibles:**
1. **Pseudoinversa**: W = pinv(Φ) @ Y (método por defecto)
2. **Pseudoinversa con regularización**: W = pinv(ΦᵀΦ + λI) @ Φᵀ @ Y

### Código Profundo - Pseudoinversa

**Path:** `src/models/rbf/solver.py`

```python
def solve_pseudoinverse(Phi: np.ndarray, y: np.ndarray, 
                      regularization: float = 0.0) -> np.ndarray:
    """
    Resolver para pesos de salida usando pseudoinversa.
    
    Sistema: ΦW = Y
    Solución: W = pinv(Φ) @ Y
    
    Con regularización: W = pinv(ΦᵀΦ + λI) @ Φᵀ @ Y
    
    Args:
        Phi: Matriz de diseño de forma (n_samples, n_centers)
        y: Matriz de salida objetivo de forma (n_samples, n_outputs)
        regularization: Parámetro de regularización λ
        
    Returns:
        Matriz de pesos de forma (n_centers, n_outputs)
    """
    n_centers = Phi.shape[1]
    
    if regularization > 0:
        # Pseudoinversa con regularización (Tikhonov)
        # W = (ΦᵀΦ + λI)⁻¹ @ Φᵀ @ Y
        Phi_T_Phi = Phi.T @ Phi
        identity = np.eye(n_centers)
        regularized_matrix = Phi_T_Phi + regularization * identity
        pseudoinverse = np.linalg.inv(regularized_matrix) @ Phi.T
    else:
        # Pseudoinversa estándar
        # W = pinv(Φ) @ Y
        pseudoinverse = np.linalg.pinv(Phi)
    
    W = pseudoinverse @ y
    return W
```

### Código Profundo - Aplicación en Entrenamiento

**Path:** `src/models/rbf/network.py`

```python
def fit(self, X: np.ndarray, y: np.ndarray, centers: np.ndarray = None) -> None:
    """Entrenar la red RBF."""
    
    # ... cálculo de matriz de diseño Phi_bias ...
    
    # Resolver para pesos de salida usando pseudoinversa
    self.weights = solve_pseudoinverse(
        Phi_bias,
        y,
        regularization=self.config.regularization
    )
    
    # Extraer bias si se usa
    if self.config.use_bias:
        self.bias = self.weights[-1, :]  # Última fila es el bias
        self.weights = self.weights[:-1, :]  # Resto son pesos de capa oculta
    else:
        self.bias = np.zeros(self.n_outputs_)
    
    self.is_fitted = True
```

### Código API - Entrenamiento con Regularización

```python
# Sin regularización
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=20,
    regularization=0.0
)
net.train(X_train, y_train)

# Con regularización (para evitar overfitting)
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=20,
    regularization=0.01
)
net.train(X_train, y_train)
```

### Output Ejemplo

```python
# Datos de ejemplo
Phi = np.array([[0.9, 0.8, 1.0],
                [0.9, 0.9, 1.0],
                [0.6, 0.9, 1.0]])  # (3, 3) con bias
y = np.array([[1.0], [0.0], [1.0]])  # (3, 1)

# Resolver sin regularización
W = solve_pseudoinverse(Phi, y, regularization=0.0)
print("Pesos sin regularización:")
print(W)
# Output:
# Pesos sin regularización:
# [[ 0.5]
#  [-0.5]
#  [ 0.5]]

# Resolver con regularización
W_reg = solve_pseudoinverse(Phi, y, regularization=0.1)
print("Pesos con regularización:")
print(W_reg)
# Output:
# Pesos con regularización:
# [[ 0.48]
#  [-0.48]
#  [ 0.48]]
```

---

## 6. Evaluación del Modelo (Simulación)

### Descripción

Aplicar el modelo entrenado sobre datos no vistos:
1. Repetir el cálculo de distancias y activaciones
2. Generar predicciones usando los pesos obtenidos

Nota: "simular" y "evaluar" corresponden al mismo proceso en este contexto (inferencia sobre nuevos datos).

### Código Profundo - Predicción

**Path:** `src/models/rbf/network.py`

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Hacer predicciones para datos de entrada X.
    
    La predicción se calcula como: y_pred = Phi @ W + b
    
    Args:
        X: Matriz de entrada de forma (n_samples, n_features)
        
    Returns:
        Matriz de predicciones de forma (n_samples, n_outputs)
        
    Raises:
        NotFittedError: Si el modelo no ha sido entrenado aún
        InvalidInputError: Si la forma de entrada es inválida
    """
    if not self.is_fitted:
        raise NotFittedError("El modelo debe ser ajustado antes de hacer predicciones")
    
    X = np.asarray(X)
    
    if X.ndim != 2:
        raise InvalidInputError(f"X debe ser array 2D, se obtuvo forma {X.shape}")
    
    if X.shape[1] != self.n_features_:
        raise InvalidInputError(
            f"Se esperaban {self.n_features_} características, se obtuvieron {X.shape[1]}"
        )
    
    # Paso 1: Calcular activaciones de capa oculta
    hidden_output = self.rbflayer.forward(X)
    
    # Paso 2: Calcular salida: hidden_output * weights + bias
    predictions = hidden_output @ self.weights + self.bias
    
    return predictions
```

### Código API - Predicción

**Path:** `api/neural_network.py`

```python
# Entrenar modelo
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)
net.train(X_train, y_train)

# Hacer predicciones
y_pred = net.predict(X_test)
print(f"Predicciones: {y_pred}")
print(f"Forma: {y_pred.shape}")  # (n_test_samples, n_outputs)
```

### Output Ejemplo

```python
# Datos de entrenamiento
X_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
y_train = np.array([[1.0], [0.0], [1.0]])

# Entrenar
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=2)
net.train(X_train, y_train)

# Datos de prueba
X_test = np.array([[0.15, 0.25], [0.35, 0.45]])

# Predecir
y_pred = net.predict(X_test)
print("Predicciones:")
print(y_pred)
# Output:
# Predicciones:
# [[0.95]
#  [0.12]]

# Interpretación:
# - Patrón [0.15, 0.25] → predicción 0.95 (cerca de 1.0)
# - Patrón [0.35, 0.45] → predicción 0.12 (cerca de 0.0)
```

---

## 7. Métricas de Evaluación / Matriz de Confusión

### Descripción

Implementar métricas para medir el desempeño del modelo:
- **Error (MSE, RMSE, etc.)**: Para regresión
- **Exactitud, precisión, recall, F1-score**: Para clasificación
- **Matriz de confusión**: Para clasificación

### Código Profundo - Evaluación Básica

**Path:** `src/evaluation/evaluator.py`

```python
class Evaluator:
    """Evaluador de modelos de redes neuronales."""
    
    def evaluate(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> EvaluationResult:
        """
        Evaluar el rendimiento del modelo.
        
        Calcula métricas como MSE, RMSE, MAE, R².
        
        Args:
            model: Modelo entrenado
            X: Datos de entrada
            y: Valores verdaderos
            
        Returns:
            EvaluationResult con métricas calculadas
        """
        y_pred = model.predict(X)
        
        # Calcular métricas de error
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        
        # Calcular R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return EvaluationResult(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2
        )
```

### Código Profundo - Matriz de Confusión

**Path:** `src/evaluation/confusion_matrix.py`

```python
class ConfusionMatrixCalculator:
    """Calculadora de matriz de confusión y métricas derivadas."""
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, 
                labels=None, discretize: bool = True) -> ConfusionMatrixResult:
        """
        Calcular matriz de confusión y métricas.
        
        Args:
            y_true: Valores verdaderos
            y_pred: Valores predichos
            labels: Etiquetas de clases (opcional)
            discretize: Si discretizar predicciones continuas
            
        Returns:
            ConfusionMatrixResult con matriz y métricas
        """
        # Discretizar predicciones si es necesario
        if discretize:
            y_pred = self._discretize_predictions(y_pred, y_true)
        
        # Calcular matriz de confusión
        matrix = self._compute_matrix(y_true, y_pred, labels)
        
        # Calcular matrices normalizadas
        matrix_normalized_row = self._normalize_by_row(matrix)  # Recall
        matrix_normalized_col = self._normalize_by_column(matrix)  # Precision
        
        # Calcular métricas por clase
        precision, recall, f1, support = self._compute_class_metrics(matrix, labels)
        
        # Calcular métricas globales
        accuracy = np.trace(matrix) / np.sum(matrix)
        macro_avg = self._compute_macro_avg(precision, recall, f1)
        weighted_avg = self._compute_weighted_avg(precision, recall, f1, support)
        
        return ConfusionMatrixResult(
            matrix=matrix,
            matrix_normalized_row=matrix_normalized_row,
            matrix_normalized_col=matrix_normalized_col,
            precision=precision,
            recall=recall,
            f1_score=f1,
            support=support,
            accuracy=accuracy,
            macro_avg=macro_avg,
            weighted_avg=weighted_avg,
            n_classes=len(labels) if labels else matrix.shape[0]
        )
```

### Código API - Evaluación

**Path:** `api/neural_network.py`

```python
# Evaluación básica (regresión)
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)
net.train(X_train, y_train)

metrics = net.evaluate(X_test, y_test)
print(f"MSE: {metrics.mse}")
print(f"RMSE: {metrics.rmse}")
print(f"R²: {metrics.r2}")

# Matriz de confusión (clasificación)
result = net.confusion_matrix(y_test, X=X_test)
print(f"Accuracy: {result.accuracy}")
print(f"Precision: {result.precision}")
print(f"Recall: {result.recall}")
print(f"F1-score: {result.f1_score}")
print(f"Matriz de confusión:\n{result.matrix}")
```

### Output Ejemplo

```python
# Evaluación de regresión
X_test = np.array([[0.15, 0.25], [0.35, 0.45]])
y_test = np.array([[1.0], [0.0]])
y_pred = net.predict(X_test)

metrics = net.evaluate(X_test, y_test)
print("Métricas de regresión:")
print(f"MSE: {metrics.mse:.4f}")
print(f"RMSE: {metrics.rmse:.4f}")
print(f"MAE: {metrics.mae:.4f}")
print(f"R²: {metrics.r2:.4f}")
# Output:
# Métricas de regresión:
# MSE: 0.0025
# RMSE: 0.0500
# MAE: 0.0350
# R²: 0.9750

# Matriz de confusión (clasificación)
y_test = np.array([[0], [1], [0], [1]])
y_pred = net.predict(X_test)
result = net.confusion_matrix(y_test, y_pred)

print("Matriz de confusión:")
print(result.matrix)
# Output:
# Matriz de confusión:
# [[2 0]
#  [0 2]]

print(f"Accuracy: {result.accuracy:.4f}")
print(f"Precision: {result.precision}")
print(f"Recall: {result.recall}")
print(f"F1-score: {result.f1_score}")
# Output:
# Accuracy: 1.0000
# Precision: {'0.0': 1.0, '1.0': 1.0}
# Recall: {'0.0': 1.0, '1.0': 1.0}
# F1-score: {'0.0': 1.0, '1.0': 1.0}
```

---

## Resumen del Flujo Completo

### Diagrama de Flujo

```
1. Configuración
   └─> RBFConfig (n_centers, sigma, activation, regularization)

2. Inicialización de Centros
   └─> RandomInitializer o KMeansInitializer
       └─> Centros de forma (n_centers, n_features)

3. Entrenamiento
   └─> Calcular distancias euclidianas
   └─> Aplicar función de activación
   └─> Construir matriz de diseño Φ

4. Cálculo de Pesos
   └─> Resolver ΦW = Y usando pseudoinversa
   └─> W = pinv(Φ) @ Y

5. Predicción
   └─> Calcular distancias
   └─> Aplicar activación
   └─> y_pred = Φ @ W + b

6. Evaluación
   └─> Métricas de error (MSE, RMSE, R²)
   └─> Matriz de confusión (clasificación)
```

### Ejemplo Completo

```python
import numpy as np
from api.neural_network import NeuralNetwork
from api.core.model_type import ModelType

# 1. Configuración
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=20,
    sigma=0.5,
    activation_rbf='gaussian',
    regularization=0.01,
    random_state=42
)

# 2. Entrenamiento (incluye inicialización de centros)
result = net.train(X_train, y_train)

# 3. Predicción
y_pred = net.predict(X_test)

# 4. Evaluación
metrics = net.evaluate(X_test, y_test)
print(f"MSE: {metrics.mse}")

# 5. Matriz de confusión (si es clasificación)
confusion = net.confusion_matrix(y_test, X=X_test)
print(f"Accuracy: {confusion.accuracy}")
```

---

## 8. Scripts de Entrenamiento con Flags

### Descripción

El proyecto incluye scripts de entrenamiento para datasets JSON con flags de línea de comandos para controlar el modo de ejecución y la randomización del particionamiento.

### Scripts Disponibles

- `scripts/train_dataset_rbf_1.py` - Para `dataset_rbf_1.json`
- `scripts/train_dataset_rbf_2.py` - Para `dataset_rbf_2.json`
- `scripts/train_dataset_rbf_3.py` - Para `dataset_rbf_3.json`

### Flags Disponibles

#### `--mode`

Controla qué evaluaciones ejecutar después del entrenamiento.

| Valor | Descripción |
|-------|-------------|
| `train` | Solo entrena el modelo (sin evaluación) |
| `val` | Entrena y evalúa en validation |
| `test` | Entrena y evalúa en test |
| `all` (default) | Entrena y evalúa en validation y test |

#### `--random`

Controla si el particionamiento 70/15/15 es random o reproducible.

| Presente | Comportamiento |
|----------|---------------|
| No (default) | Usa semilla 42 (reproducible) |
| Sí | Randomiza sin semilla (cada ejecución diferente) |

### Ejemplos de Uso

```bash
# Ejecutar todo (entrenar + validation + test)
python scripts/train_dataset_rbf_1.py

# Solo entrenar
python scripts/train_dataset_rbf_1.py --mode train

# Entrenar + validar
python scripts/train_dataset_rbf_1.py --mode val

# Entrenar + test
python scripts/train_dataset_rbf_1.py --mode test

# Randomizar particionamiento
python scripts/train_dataset_rbf_1.py --random

# Combinar flags
python scripts/train_dataset_rbf_1.py --mode val --random
```

### Flujo de los Scripts

1. **Cargar datos** desde JSON
2. **Limpiar datos** (manejar valores nulos)
3. **Particionar** 70/15/15 (train/validation/test)
4. **Entrenar** modelo RBF
5. **Evaluar** según modo (validation/test/all)

### Funciones Auxiliares

**`evaluate_model(net, X, y, label)`**
- Función separada para mantener `main()` limpio
- Calcula MSE, R² y matriz de confusión
- Muestra todas las métricas de forma organizada

### Código de Ejemplo

```python
def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo RBF con dataset')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'val', 'test', 'all'],
                        help='Modo de ejecución: train (solo entrenar), val (entrenar+validar), test (entrenar+test), all (todo)')
    parser.add_argument('--random', action='store_true',
                        help='Randomizar particionamiento 70/15/15')
    args = parser.parse_args()

    # Randomizar o no según flag
    random_state = None if args.random else 42

    # Particionar datos
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=random_state
    )

    # Entrenar
    net = NeuralNetwork(model_type=ModelType.RBF, ...)
    result = net.train(X_train, y_train, verbose=True)

    # Evaluar según modo
    if args.mode in ['val', 'all']:
        evaluate_model(net, X_val, y_val, "Validation")

    if args.mode in ['test', 'all']:
        evaluate_model(net, X_test, y_test, "Test")
```

---

## Archivos Principales

| Archivo | Descripción |
|---------|-------------|
| `src/models/rbf/config.py` | Configuración de parámetros RBF |
| `src/models/rbf/layer.py` | Capa oculta RBF (cálculo de activaciones) |
| `src/models/rbf/network.py` | Red RBF completa (entrenamiento y predicción) |
| `src/models/rbf/solver.py` | Resolución de pesos (pseudoinversa) |
| `src/training/initializer.py` | Estrategias de inicialización de centros |
| `src/core/activation.py` | Funciones de activación (gaussiana, etc.) |
| `src/core/distance.py` | Cálculo de distancias euclidianas |
| `src/evaluation/evaluator.py` | Evaluación de modelos (MSE, RMSE, etc.) |
| `src/evaluation/confusion_matrix.py` | Matriz de confusión y métricas |
| `api/neural_network.py` | API pública para usuarios |
| `scripts/train_dataset_rbf_1.py` | Script de entrenamiento para dataset 1 |
| `scripts/train_dataset_rbf_2.py` | Script de entrenamiento para dataset 2 |
| `scripts/train_dataset_rbf_3.py` | Script de entrenamiento para dataset 3 |

---

## 9. Documentación de Sustentación - Flujo Completo RBF

Esta sección documenta el flujo completo de implementación de redes RBF siguiendo el formato de sustentación académica. Cada sección incluye la descripción detallada, las rutas de código, el código fuente (tanto del script como de la API y código interno), ejemplos de salida, y opciones de personalización.

---

### 9.0 Flags y Parámetros del Script de Entrenamiento

**Descripción:**

El script de entrenamiento `train_dataset_rbf_*.py` acepta varios flags de línea de comandos que controlan el comportamiento del entrenamiento, la visualización de resultados y la reproducibilidad. Estos parámetros permiten personalizar la ejecución sin modificar el código fuente.

**Flags disponibles:**

| Flag | Tipo | Valores | Descripción | Uso típico |
|------|------|---------|-------------|------------|
| `--mode` | string | `train`, `val`, `test`, `all` | Modo de ejecución: solo entrenar, entrenar+validar, entrenar+test, o todo | `python script.py --mode val` |
| `--random` | flag | - | Randomizar particionamiento 70/15/15 (sin semilla fija) | `python script.py --random` |
| `--verbose-decimals` | flag | - | Mostrar todos los decimales (default: 4 decimales redondeados) | `python script.py --verbose-decimals` |
| `--plot` | flag | - | Generar gráficas de resultados (4 gráficas PNG) | `python script.py --plot` |

**Path:**
- `scripts/train_dataset_rbf_1.py` - Función `main()` (líneas 322-333)

**IMG (Código - Definición de flags):**

```python
# Path: scripts/train_dataset_rbf_1.py
def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo RBF con dataset')
    
    # --mode: Controla qué fases de evaluación ejecutar
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'val', 'test', 'all'],
                        help='Modo de ejecución: train (solo entrenar), val (entrenar+validar), test (entrenar+test), all (todo)')
    
    # --random: Controla reproducibilidad del particionamiento
    parser.add_argument('--random', action='store_true',
                        help='Randomizar particionamiento 70/15/15')
    
    # --verbose-decimals: Controla precisión de salida numérica
    parser.add_argument('--verbose-decimals', action='store_true',
                        help='Mostrar todos los decimales (default: 4 decimales)')
    
    # --plot: Controla generación de visualizaciones
    parser.add_argument('--plot', action='store_true',
                        help='Generar gráficas de resultados')
    
    args = parser.parse_args()
```

**IMG (Código - Uso de flags en el flujo):**

```python
# Path: scripts/train_dataset_rbf_1.py
def main():
    # ... parse de argumentos
    
    # Uso de --random: Controla semilla para particionamiento
    random_state = None if args.random else 42
    print(f"\n==== Particionamiento ====")
    print(f"70/15/15 (random={args.random})")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=random_state
    )
    
    # Uso de --verbose-decimals: Controla formato de salida
    result = net.train(X_train, y_train, verbose=True)
    print(f"Tiempo: {format_float(result.training_time, args.verbose_decimals)}s")
    
    # Uso de --mode: Controla qué evaluaciones ejecutar
    if args.mode in ['val', 'all']:
        evaluate_model(net, X_val, y_val, "Validation", args.verbose_decimals)
    
    if args.mode in ['test', 'all']:
        evaluate_model(net, X_test, y_test, "Test", args.verbose_decimals)
    
    # Uso de --plot: Controla generación de gráficas
    if args.plot:
        plot_results(net, X_train, y_train, X_val, y_val, X_test, y_test,
                    data_dict['dataset'], args.verbose_decimals)
```

**Ejemplos de uso:**

```bash
# Ejecución estándar (entrenar + validar + test)
python scripts/train_dataset_rbf_1.py

# Solo entrenar (sin evaluación)
python scripts/train_dataset_rbf_1.py --mode train

# Entrenar y validar solo
python scripts/train_dataset_rbf_1.py --mode val

# Entrenar y testear solo
python scripts/train_dataset_rbf_1.py --mode test

# Con particionamiento aleatorio (no reproducible)
python scripts/train_dataset_rbf_1.py --random

# Con todos los decimales (sin redondeo)
python scripts/train_dataset_rbf_1.py --verbose-decimals

# Generar gráficas
python scripts/train_dataset_rbf_1.py --plot

# Combinación de flags
python scripts/train_dataset_rbf_1.py --mode all --plot --verbose-decimals
```

**Output (Ejemplo con diferentes flags):**

```
# Sin flags (default):
==== Particionamiento ====
70/15/15 (random=False)
Train: 700, Val: 150, Test: 150
Tiempo: 0.0248s

# Con --random:
==== Particionamiento ====
70/15/15 (random=True)
Train: 700, Val: 150, Test: 150

# Con --verbose-decimals:
Tiempo: 0.0247835623456789s

# Con --plot:
Gráficas guardadas en: plots/
```

---

### 9.0.1 Generación de Gráficas (Flag --plot)

**Descripción:**

El flag `--plot` activa la generación de 4 gráficas de visualización que ayudan a analizar el rendimiento del modelo. Las gráficas se guardan en la carpeta `plots/` en formato PNG con alta resolución (300 DPI). Esta funcionalidad es útil para análisis exploratorio y presentación de resultados en documentación o presentaciones académicas.

**Gráficas generadas:**

| Gráfica | Descripción | Archivo generado |
|---------|-------------|------------------|
| **Matriz de Confusión** | Heatmap con valores numéricos de TP, TN, FP, FN | `{dataset}_confusion_matrix.png` |
| **Distribución de Clases** | Barras comparativas de Train/Val/Test por clase | `{dataset}_class_distribution.png` |
| **Métricas por Clase** | Barras agrupadas de Precision, Recall, Specificity, F1 | `{dataset}_metrics_by_class.png` |
| **Scatter Plot de Datos** | Visualización 2D/3D de datos (solo si features ≤ 3) | `{dataset}_data_scatter.png` |

**Path:**
- `scripts/train_dataset_rbf_1.py` - Función `plot_results()` (líneas 211-319)

**IMG (Código - Función plot_results):**

```python
# Path: scripts/train_dataset_rbf_1.py
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
```

**IMG (Código - Uso en main):**

```python
# Path: scripts/train_dataset_rbf_1.py
def main():
    # ... entrenamiento y evaluación
    
    # Generar gráficas si se solicita
    if args.plot:
        plot_results(net, X_train, y_train, X_val, y_val, X_test, y_test,
                    data_dict['dataset'], args.verbose_decimals)
```

**Ejemplo de uso:**

```bash
# Generar gráficas
python scripts/train_dataset_rbf_1.py --plot

# Generar gráficas con todos los decimales
python scripts/train_dataset_rbf_1.py --plot --verbose-decimals

# Generar gráficas solo en modo test
python scripts/train_dataset_rbf_1.py --mode test --plot
```

**Output (Ejemplo de salida):**

```
Gráficas guardadas en: plots/
```

**Archivos generados (ejemplo):**

```
plots/
├── clasificacion_2_clases_grande_confusion_matrix.png
├── clasificacion_2_clases_grande_class_distribution.png
├── clasificacion_2_clases_grande_metrics_by_class.png
└── clasificacion_2_clases_grande_data_scatter.png  (solo si features ≤ 3)
```

**Dependencias:**
- `matplotlib.pyplot` - Gráficos base
- `seaborn` - Heatmaps y estilos mejorados

---

### 9.1 Sobre el Dataset

**Descripción:**

El conjunto de datos utilizado es numérico, compuesto por múltiples patrones para clasificación. Cada patrón está formado por 4 variables de entrada (features) y 1 variable de salida discreta (clase). Los datos se almacenan en formato JSON con una estructura de clave-valor, donde cada registro contiene un vector de entrada con sus características numéricas y su etiqueta de clase correspondiente. El dataset está balanceado, con 500 muestras por clase para un total de 1000 patrones. El tipo de problema es clasificación binaria, donde las clases son 0 y 1.

**Path:**
- `jsons/dataset_rbf_1.json` - Dataset en formato JSON
- `scripts/train_dataset_rbf_1.py` - Script de entrenamiento (funciones `load_json_data` y `clean_data`)

**IMG (Código - Script de entrenamiento):**

```python
# Path: scripts/train_dataset_rbf_1.py
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
```

**IMG (Código - Uso en main):**

```python
# Path: scripts/train_dataset_rbf_1.py
def main():
    # ...
    json_path = os.path.join(script_dir, '..', 'jsons', 'dataset_rbf_1.json')

    print("Cargando datos...")
    data_dict = load_json_data(json_path)
    print(f"Dataset: {data_dict['dataset']}, muestras: {len(data_dict['data'])}")

    print("Limpiando datos...")
    X, y = clean_data(data_dict)
    # ...
```

**Output (Ejemplo de salida):**

```
Cargando datos...
Dataset: clasificacion_2_clases_grande, muestras: 1000
Limpiando datos...
```

---

### 9.2 Definición de Parámetros de Entrada

**Descripción:**

Se especifican las dimensiones del problema definiendo el número total de patrones, el número de variables de entrada y el número de salidas del modelo. Esta definición es fundamental para configurar correctamente la arquitectura de la red RBF. Se calculan estadísticas descriptivas para entender la distribución de los datos, incluyendo la media, desviación estándar, valores mínimos y máximos por característica. También se analiza el balanceo de clases para verificar si el dataset está equilibrado, lo cual es importante para el entrenamiento del modelo de clasificación.

**Path:**
- `scripts/train_dataset_rbf_1.py` - Función `show_input_parameters()`

**IMG (Código - Script de entrenamiento):**

```python
# Path: scripts/train_dataset_rbf_1.py
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
```

**IMG (Código - Uso en main):**

```python
# Path: scripts/train_dataset_rbf_1.py
def main():
    # ...
    X, y = clean_data(data_dict)
    show_input_parameters(X, y, data_dict, args.verbose_decimals)
    # ...
```

**Output (Ejemplo de salida):**

```
==== Parámetros de Entrada ====
Dataset: clasificacion_2_clases_grande
Features: ['x1', 'x2', 'x3', 'x4']
Muestras: 1000
Características: 4
Clases: [0 1]
Número de clases: 2
Balanceo: {0: 500, 1: 500}

==== Estadísticas Descriptivas ====
Media X: [3.4911 3.4711 3.5463 3.468 ]
Std X: [2.5992 2.5957 2.6082 2.603 ]
Min X: [-1.561 -1.28  -1.139 -1.844]
Max X: [8.37  8.6   8.648 8.282]
Media y: 0.5000
Std y: 0.5000
```

---

### 9.3 Partición del Dataset

**Descripción:**

El conjunto de datos se divide en tres subconjuntos para entrenamiento, validación y prueba (simulación). La partición sigue una proporción de 70% para entrenamiento, 15% para validación y 15% para prueba. Esta división permite entrenar el modelo con la mayoría de los datos, validar el rendimiento durante el entrenamiento para ajustar hiperparámetros, y finalmente evaluar el modelo en datos nunca vistos para estimar su desempeño real. La partición se realiza de forma aleatoria para asegurar que los datos estén mezclados y no haya sesgos en la distribución de clases entre los subconjuntos. Se puede controlar la reproducibilidad mediante una semilla aleatoria.

**Path:**
- `scripts/train_dataset_rbf_1.py` - Función `split_data()`

**IMG (Código - Script de entrenamiento):**

```python
# Path: scripts/train_dataset_rbf_1.py
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
```

**IMG (Código - Uso en main):**

```python
# Path: scripts/train_dataset_rbf_1.py
def main():
    # ...
    # Randomizar o no según flag
    random_state = None if args.random else 42
    print(f"\n==== Particionamiento ====")
    print(f"70/15/15 (random={args.random})")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    # ...
```

**Output (Ejemplo de salida):**

```
==== Particionamiento ====
70/15/15 (random=False)
Train: 700, Val: 150, Test: 150
```

---

### 9.4 Parámetros de Activación/Entrenamiento de la Red RBF

**Descripción:**

Se configura la arquitectura y parámetros clave del modelo RBF. El número de neuronas (centros) se calcula dinámicamente como el 10% de las muestras de entrenamiento con un máximo de 50 centros para mantener el modelo eficiente. La función de activación utilizada es la gaussiana, que calcula la activación como φ(r) = exp(-(r/σ)²), donde r es la distancia euclidiana y σ es el parámetro de ancho. Los centros radiales se inicializan aleatoriamente seleccionando muestras de los datos de entrenamiento, asegurando que cada centro tenga la misma dimensión que las entradas (4 características). La cantidad de centros coincide con el número de neuronas. La estructura de red es unicapa, compuesta por una capa oculta de neuronas RBF y una capa de salida lineal. Se incluye regularización Tikhonov con parámetro λ para mejorar la estabilidad numérica en el cálculo de la pseudoinversa, y se incluye un término de bias en la capa de salida.

**Parámetros de personalización disponibles:**

| Parámetro | Tipo | Default | Descripción | Valores válidos |
|-----------|------|---------|-------------|----------------|
| `n_centers` | int | 20 | Número de centros RBF (neuronas ocultas) | Cualquier entero positivo |
| `sigma` | float | 1.0 | Parámetro de ancho de la función gaussiana | Cualquier float positivo |
| `activation_rbf` | str | 'gaussian' | Función de activación RBF | 'gaussian', 'multiquadratic', 'inverse_multiquadratic', 'thin_plate' (ln), 'thin_plate_log10' |
| `regularization` | float | 0.01 | Parámetro de regularización Tikhonov (λ) | Cualquier float ≥ 0 |
| `use_bias` | bool | True | Incluir término de bias en capa de salida | True, False |
| `random_state` | int | 42 | Semilla aleatoria para reproducibilidad | Cualquier entero o None |
| `initializer` | str | 'kmeans' | Estrategia de inicialización de centros | 'kmeans', 'random' |

**Path:**
- `scripts/train_dataset_rbf_1.py` - Configuración en `main()` (líneas 355-363)
- `api/neural_network.py` - Clase `NeuralNetwork` (método `__init__`, líneas 33-52)
- `api/config.py` - Clase `NeuralNetworkConfig` (líneas 10-68)
- `src/models/rbf/config.py` - Clase `RBFConfig`

**IMG (Código - Script de entrenamiento):**

```python
# Path: scripts/train_dataset_rbf_1.py
def main():
    # ...
    print(f"\n==== Entrenamiento ====")
    net = NeuralNetwork(
        model_type=ModelType.RBF,
        n_centers=min(50, X_train.shape[0] // 10),  # 10% de muestras o máx 50
        sigma=1.0,                                    # Ancho de función gaussiana
        activation_rbf='gaussian',                    # Función de activación
        regularization=0.01,                          # Regularización Tikhonov
        random_state=42                               # Reproducibilidad
    )
    result = net.train(X_train, y_train, verbose=True)
    print(f"Tiempo: {format_float(result.training_time, args.verbose_decimals)}s")
    # ...
```

**IMG (Código - API NeuralNetwork):**

```python
# Path: api/neural_network.py
class NeuralNetwork:
    def __init__(
        self,
        model_type: ModelType,
        config: Optional[NeuralNetworkConfig] = None,
        hidden_layers: Optional[List[int]] = None,
        # Parámetros específicos de RBF
        n_centers: Optional[int] = None,
        sigma: float = 1.0,
        activation_rbf: str = 'gaussian',
        # Parámetros específicos de backpropagation
        activation_backprop: str = 'sigmoid',
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: int = 32,
        # Parámetros comunes
        use_bias: bool = True,
        regularization: float = 0.01,
        random_state: int = 42,
        initializer: str = 'kmeans'
    ):
        """
        Inicializar la red neuronal.
        
        Args:
            model_type: Tipo de modelo (ModelType.RBF o ModelType.BACKPROP)
            config: Objeto de configuración opcional (NeuralNetworkConfig)
            hidden_layers: Lista de tamaños de capas ocultas (para backpropagation)
            n_centers: Número de centros RBF (para RBF)
            sigma: Parámetro de ancho de RBF
            activation_rbf: Función de activación RBF ('gaussian', 'multiquadratic', 'inverse_multiquadratic', 'thin_plate', 'thin_plate_log10')
            activation_backprop: Función de activación backpropagation ('sigmoid', 'tanh', 'relu')
            learning_rate: Tasa de aprendizaje para backpropagation
            epochs: Número de épocas de entrenamiento (para backpropagation)
            batch_size: Tamaño de lote para mini-batch training
            use_bias: Si usar términos de bias
            regularization: Parámetro de regularización
            random_state: Semilla aleatoria para reproducibilidad
            initializer: Estrategia de inicialización de centros ('kmeans' o 'random')
        """
        # Usar configuración proporcionada o crear desde parámetros individuales
        if config is None:
            config = NeuralNetworkConfig(
                hidden_layers=hidden_layers,
                n_centers=n_centers,
                sigma=sigma,
                activation_rbf=activation_rbf,
                activation_backprop=activation_backprop,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                use_bias=use_bias,
                regularization=regularization,
                random_state=random_state,
                initializer=initializer
            )
        
        # Validar configuración
        config.validate()
        
        self.model_type = model_type
        self.config = config
        
        # Modelo y entrenador internos
        self.model = None
        self.trainer = None
        self.evaluator = Evaluator()
```

**IMG (Código - Configuración API):**

```python
# Path: api/config.py
@dataclass
class NeuralNetworkConfig:
    """
    Clase de configuración para la API de redes neuronales.
    """
    hidden_layers: List[int] = None
    """Lista de tamaños de capas ocultas (para backpropagation)"""
    
    n_centers: Optional[int] = None
    """Número de centros RBF (para RBF)"""
    
    sigma: float = 1.0
    """Parámetro de ancho de RBF"""
    
    activation_rbf: str = 'gaussian'
    """Función de activación RBF: 'gaussian', 'multiquadratic', 'inverse_multiquadratic', 'thin_plate' (ln), 'thin_plate_log10'"""
    
    activation_backprop: str = 'sigmoid'
    """Función de activación backpropagation: 'sigmoid', 'logsig', 'tanh', 'tansig', 'relu', 'linear', 'purelin', 'leaky_relu'"""
    
    learning_rate: float = 0.01
    """Tasa de aprendizaje para backpropagation"""
    
    epochs: int = 1000
    """Número de épocas de entrenamiento (para backpropagation)"""
    
    batch_size: int = 32
    """Tamaño de lote para mini-batch training"""
    
    use_bias: bool = True
    """Si usar términos de bias"""
    
    regularization: float = 0.01
    """Parámetro de regularización"""
    
    random_state: int = 42
    """Semilla aleatoria para reproducibilidad"""
    
    initializer: str = 'kmeans'
    """Estrategia de inicialización de centros: 'kmeans' o 'random'"""
    
    def __post_init__(self):
        """Validar y establecer valores por defecto."""
        if self.hidden_layers is None:
            self.hidden_layers = [10]
        
        if self.n_centers is None:
            self.n_centers = 20
```

**IMG (Código - Configuración interna RBF):**

```python
# Path: src/models/rbf/config.py
@dataclass
class RBFConfig:
    """
    Clase de configuración para parámetros de red RBF.
    """
    n_centers: int = 10
    """Número de centros de función de base radial (neuronas en capa oculta)"""
    
    sigma: float = 1.0
    """Parámetro de ancho para la función de activación (dispersión de cada RBF)"""
    
    activation: ActivationFunction = None
    """Función de activación a usar (por defecto Gaussiana si no se especifica)"""
    
    regularization: float = 0.0
    """Parámetro de regularización para la pseudoinversa (agrega matriz identidad * λ)"""
    
    use_bias: bool = True
    """Si incluir término de bias en la capa de salida"""
    
    random_state: int = None
    """Semilla aleatoria para reproducibilidad"""
    
    def __post_init__(self):
        """Establecer función de activación por defecto a Gaussiana."""
        if self.activation is None:
            self.activation = GaussianActivation()
    
    def validate(self) -> None:
        """Validar los parámetros de configuración."""
        if self.n_centers <= 0:
            raise InvalidConfigError(f"n_centers debe ser positivo")
        if self.sigma <= 0:
            raise InvalidConfigError(f"sigma debe ser positivo")
        if self.regularization < 0:
            raise InvalidConfigError(f"regularization debe ser no negativo")
```

**Ejemplos de personalización:**

```python
# Ejemplo 1: Configuración estándar (como en el script)
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=50,
    sigma=1.0,
    activation_rbf='gaussian',
    regularization=0.01,
    random_state=42
)

# Ejemplo 2: Más centros, sigma más pequeño (funciones más localizadas)
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=100,
    sigma=0.5,
    activation_rbf='gaussian',
    regularization=0.001,
    random_state=42
)

# Ejemplo 3: Sin regularización, sin bias
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=30,
    sigma=2.0,
    activation_rbf='gaussian',
    regularization=0.0,
    use_bias=False,
    random_state=42
)

# Ejemplo 4: Usando función de activación multiquadrática
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=40,
    sigma=1.0,
    activation_rbf='multiquadratic',
    regularization=0.01,
    random_state=42
)

# Ejemplo 5: Usando objeto de configuración
from api.config import NeuralNetworkConfig

config = NeuralNetworkConfig(
    n_centers=60,
    sigma=0.8,
    activation_rbf='gaussian',
    regularization=0.005,
    random_state=123,
    initializer='kmeans'
)

net = NeuralNetwork(model_type=ModelType.RBF, config=config)
```

**Output (Ejemplo de salida):**

```
==== Entrenamiento ====
Entrenamiento completado en 0.0248 segundos
Error final: 0.021922
Épocas: 1
Tiempo: 0.0248s
```

---

### 9.5 Inicialización de Centros Radiales

**Descripción:**

Los centros radiales se asignan mediante estrategias de inicialización que determinan cómo posicionar los centros RBF en el espacio de entrada. Cada centro tiene la misma dimensión que las entradas (4 características). Existen dos estrategias principales: selección aleatoria de patrones del conjunto de entrenamiento (RandomInitializer) y clustering k-means (KMeansInitializer). La estrategia por defecto en el script es la selección aleatoria directa dentro del método `fit()` de `RBFNetwork`, pero se puede personalizar usando el parámetro `initializer` en la configuración o proporcionando centros pre-calculados directamente al método `fit()`. Esto asegura que los centros estén dentro del rango de los datos y sean representativos de la distribución. La dimensión de cada centro es igual al número de entradas, y la cantidad de centros coincide con el número de neuronas en la capa oculta.

**Estrategias de inicialización disponibles:**

| Estrategia | Descripción | Ventajas | Desventajas |
|-----------|-------------|----------|-------------|
| `random` (default en script) | Selecciona aleatoriamente muestras de los datos | Simple, rápido, centros dentro del rango de datos | Puede no ser representativo de la distribución |
| `kmeans` | Usa clustering k-means para encontrar centroides | Centros bien distribuidos, representativos | Más lento, requiere iteraciones |
| `custom` | Proporcionar centros pre-calculados | Control total, puede usar cualquier algoritmo | Requiere cálculo previo manual |

**Parámetros de personalización:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `initializer` | str | 'kmeans' | Estrategia: 'kmeans' o 'random' (en API) |
| `centers` | np.ndarray | None | Centros pre-calculados (forma: n_centers × n_features) |
| `max_iterations` | int | 100 | Máximo de iteraciones para k-means |
| `tolerance` | float | 1e-4 | Tolerancia de convergencia para k-means |

**Path:**
- `src/models/rbf/network.py` - Método `fit()` (líneas 110-126, inicialización de centros)
- `src/training/initializer.py` - Clases `RandomInitializer` y `KMeansInitializer` (líneas 10-130)
- `api/config.py` - Parámetro `initializer` en `NeuralNetworkConfig` (línea 58)

**IMG (Código - Red RBF interna - Inicialización en fit):**

```python
# Path: src/models/rbf/network.py
class RBFNetwork(BaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray, centers: np.ndarray = None) -> None:
        """
        Entrenar la red RBF con datos de entrada X y salidas objetivo y.
        """
        # Validar formas de entrada
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Almacenar dimensiones
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        
        # Opción 1: Usar centros proporcionados manualmente
        if centers is not None:
            self.centers = centers
            self.config.n_centers = centers.shape[0]
        elif self.centers is None:
            # Opción 2: Si no se proporcionan centros, muestrear aleatoriamente de datos de entrenamiento
            n_samples = X.shape[0]
            n_centers = min(self.config.n_centers, n_samples)
            indices = np.random.choice(n_samples, n_centers, replace=False)
            self.centers = X[indices]
        
        # Crear la capa RBF con los centros
        self.rbflayer = RBFLayer(
            centers=self.centers,
            activation=self.config.activation,
            sigma=self.config.sigma
        )
        # ... continuación del entrenamiento
```

**IMG (Código - RandomInitializer):**

```python
# Path: src/training/initializer.py
class RandomInitializer(BaseCenterInitializer):
    """
    Estrategia de inicialización aleatoria.
    
    Esta estrategia selecciona aleatoriamente n_centers muestras de los datos de entrada
    para servir como centros RBF. Es simple y a menudo funciona bien en la práctica.
    """
    
    def initialize(self, X: np.ndarray, n_centers: int) -> np.ndarray:
        """
        Inicializar centros muestreando aleatoriamente de los datos de entrada.
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            n_centers: Número de centros a inicializar
            
        Returns:
            Matriz de centros de forma (n_centers, n_features)
        """
        n_samples = X.shape[0]
        n_centers = min(n_centers, n_samples)
        
        indices = np.random.choice(n_samples, n_centers, replace=False)
        centers = X[indices]
        
        return centers
```

**IMG (Código - KMeansInitializer):**

```python
# Path: src/training/initializer.py
class KMeansInitializer(BaseCenterInitializer):
    """
    Estrategia de inicialización K-means.
    
    Esta estrategia usa clustering k-means para encontrar centros representativos.
    Los centros son los centroides de los clusters, que tienden a estar bien distribuidos
    a través de la distribución de datos.
    
    Objetivo de k-means: minimizar J = sum_i ||x_i - mu_{c_i}||^2
    donde mu_{c_i} es el centroide del cluster asignado al punto x_i.
    
    Algoritmo:
    1. Inicializar centroides aleatoriamente
    2. Asignar cada punto al centroide más cercano
    3. Recalcular centroides como media de puntos asignados
    4. Repetir hasta convergencia
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-4):
        """
        Inicializar la estrategia k-means.
        
        Args:
            max_iterations: Número máximo de iteraciones k-means
            tolerance: Tolerancia de convergencia para movimiento de centroides
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def initialize(self, X: np.ndarray, n_centers: int) -> np.ndarray:
        """
        Inicializar centros usando clustering k-means.
        
        El algoritmo minimiza: J = sum_i ||x_i - mu_{c_i}||^2
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            n_centers: Número de centros a inicializar
            
        Returns:
            Matriz de centros de forma (n_centers, n_features)
        """
        n_samples = X.shape[0]
        n_centers = min(n_centers, n_samples)
        
        # Inicializar centroides aleatoriamente
        indices = np.random.choice(n_samples, n_centers, replace=False)
        centroids = X[indices].copy()
        
        # Ejecutar iteraciones k-means
        for iteration in range(self.max_iterations):
            # Asignar cada muestra al centroide más cercano
            distances = self._compute_distances(X, centroids)
            assignments = np.argmin(distances, axis=1)
            
            # Calcular nuevos centroides: mu_k = (1/|C_k|) * sum_{x_i in C_k} x_i
            new_centroids = np.zeros_like(centroids)
            for k in range(n_centers):
                mask = assignments == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # Verificar convergencia: ||mu_new - mu_old|| < tolerance
            centroid_shift = np.linalg.norm(new_centroids - centroids)
            if centroid_shift < self.tolerance:
                break
            
            centroids = new_centroids
        
        return centroids
```

**Ejemplos de personalización de centros:**

```python
# Ejemplo 1: Inicialización por defecto (aleatoria en fit)
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=50, sigma=1.0)
net.train(X_train, y_train)  # Centros se inicializan aleatoriamente dentro de fit()

# Ejemplo 2: Proporcionar centros pre-calculados manualmente
import numpy as np

# Calcular centros usando cualquier método personalizado
custom_centers = X_train[:50]  # Usar las primeras 50 muestras como centros
# O usar clustering de sklearn:
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=50, random_state=42)
# custom_centers = kmeans.fit(X_train).cluster_centers_

net = NeuralNetwork(model_type=ModelType.RBF, n_centers=50, sigma=1.0)
# Pasa los centros directamente al método train (o fit interno)
net.model.fit(X_train, y_train, centers=custom_centers)

# Ejemplo 3: Usar KMeansInitializer directamente
from src.training.initializer import KMeansInitializer

initializer = KMeansInitializer(max_iterations=100, tolerance=1e-4)
centers = initializer.initialize(X_train, n_centers=50)

net = NeuralNetwork(model_type=ModelType.RBF, n_centers=50, sigma=1.0)
net.model.fit(X_train, y_train, centers=centers)

# Ejemplo 4: Usar configuración con initializer='kmeans'
from api.config import NeuralNetworkConfig

config = NeuralNetworkConfig(
    n_centers=50,
    sigma=1.0,
    initializer='kmeans'  # Usará KMeansInitializer internamente
)

net = NeuralNetwork(model_type=ModelType.RBF, config=config)
net.train(X_train, y_train)

# Ejemplo 5: Centros en posiciones específicas (grid)
import numpy as np

# Crear un grid de centros en el espacio de características
x1 = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 10)
x2 = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 10)
x3 = np.linspace(X_train[:, 2].min(), X_train[:, 2].max(), 10)
x4 = np.linspace(X_train[:, 3].min(), X_train[:, 3].max(), 10)

grid_centers = np.array(np.meshgrid(x1, x2, x3, x4)).T.reshape(-1, 4)
grid_centers = grid_centers[:50]  # Tomar primeros 50

net = NeuralNetwork(model_type=ModelType.RBF, n_centers=50, sigma=1.0)
net.model.fit(X_train, y_train, centers=grid_centers)
```

**Output (Ejemplo de salida):**

```
Entrenamiento completado en 0.0248 segundos
Error final: 0.021922
Épocas: 1
Tiempo: 0.0248s
```

---

### 9.6 Entrenamiento del Modelo

**Descripción:**

El entrenamiento de la red RBF calcula la respuesta de la red mediante varios pasos. Primero, se mide la distancia euclidiana entre cada patrón de entrada y cada centro radial. Luego, se aplica la función de activación (gaussiana) sobre dichas distancias para obtener la salida de las neuronas ocultas. Se construye la matriz de diseño Φ (phi) donde cada elemento φ(i,j) es la activación de la neurona j para el patrón i. Finalmente, se resuelve el sistema ΦW = Y mediante pseudoinversa para obtener los pesos de salida. Este proceso es de forma cerrada (no iterativo), lo que hace que las redes RBF sean muy eficientes para entrenar comparadas con redes neuronales tradicionales.

**Path:**
- `api/neural_network.py` - Método `train()`
- `src/models/rbf/network.py` - Método `fit()`
- `src/models/rbf/layer.py` - Método `forward()`
- `src/core/distance.py` - Función `euclidean_distance_matrix()`
- `src/core/activation.py` - Clase `GaussianActivation`

**IMG (Código - API NeuralNetwork):**

```python
# Path: api/neural_network.py
class NeuralNetwork:
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False
    ) -> TrainingResult:
        """
        Entrenar la red neuronal.

        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            y: Matriz de salida objetivo de forma (n_samples, n_outputs)
            verbose: Si imprimir progreso de entrenamiento

        Returns:
            TrainingResult con resultados de entrenamiento y registros
        """
        # Validar entrada
        X, y = InputValidator.validate_input_pair(X, y)

        # Configurar modelo según tipo
        self._setup_model(X, y)

        # Limpiar registros anteriores
        self.training_log = {
            'error_history': [],
            'gradient_norms': [],
            'epoch_times': [],
            'performance_metrics': []
        }

        # Entrenar y rastrear métricas
        import time
        start_time = time.time()

        result = self.trainer.train(self.model, X, y)

        training_time = time.time() - start_time

        # Registrar resultados de entrenamiento
        self.training_log['error_history'] = result.error_history
        self.training_log['training_time'] = training_time

        if verbose:
            print(f"Entrenamiento completado en {training_time:.4f} segundos")
            print(f"Error final: {result.final_error:.6f}")
            print(f"Épocas: {result.epochs}")

        return TrainingResult(
            training_time=training_time,
            final_error=result.final_error,
            epochs=result.epochs,
            error_history=result.error_history,
            converged=result.converged,
            metadata=result.metadata
        )
```

**IMG (Código - Red RBF interna):**

```python
# Path: src/models/rbf/network.py
class RBFNetwork(BaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray, centers: np.ndarray = None) -> None:
        """
        Entrenar la red RBF con datos de entrada X y salidas objetivo y.
        
        El entrenamiento resuelve: W = pinv(Phi) @ y
        donde Phi es la matriz de diseño calculada como Phi = phi(d(X, C), sigma)
        """
        # ... inicialización de centros (ver sección 9.5)
        
        # Crear la capa RBF
        self.rbflayer = RBFLayer(
            centers=self.centers,
            activation=self.config.activation,
            sigma=self.config.sigma
        )
        
        # Calcular matriz de diseño (activaciones de capa oculta)
        Phi = self.rbflayer.forward(X)
        
        # Agregar columna de bias si está configurado
        if self.config.use_bias:
            Phi_bias = np.column_stack([Phi, np.ones(Phi.shape[0])])
        else:
            Phi_bias = Phi
        
        # Resolver para pesos de salida usando pseudoinversa
        self.weights = solve_pseudoinverse(
            Phi_bias,
            y,
            regularization=self.config.regularization
        )
        
        # Extraer bias si se usa
        if self.config.use_bias:
            self.bias = self.weights[-1, :]
            self.weights = self.weights[:-1, :]
        else:
            self.bias = np.zeros(self.n_outputs_)
        
        self.is_fitted = True
```

**IMG (Código - Capa RBF):**

```python
# Path: src/models/rbf/layer.py
class RBFLayer(BaseLayer):
    """
    Capa oculta de Función de Base Radial (RBF).
    """
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calcular el pase hacia adelante de la capa RBF.
        
        La salida se calcula como: Y = phi(d(X, C), sigma)
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            
        Returns:
            Matriz de activación de forma (n_samples, n_centers)
        """
        # Calcular distancias euclidianas entre todas las muestras y todos los centros
        distances = euclidean_distance_matrix(X, self.centers)
        
        # Aplicar función de activación para obtener la salida de la capa
        activations = self.activation.compute(distances, self.sigma)
        
        return activations
```

**IMG (Código - Distancia euclidiana):**

```python
# Path: src/core/distance.py
def euclidean_distance_matrix(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Calcular la matriz de distancia euclidiana entre todos los puntos y centros.
    
    Fórmula: d(i, j) = sqrt(sum((xi - cj)^2))
    
    Args:
        X: Matriz de entrada de forma (n_samples, n_features)
        centers: Matriz de centros de forma (n_centers, n_features)
        
    Returns:
        Matriz de distancias de forma (n_samples, n_centers)
    """
    X_expanded = X[:, np.newaxis, :]
    centers_expanded = centers[np.newaxis, :, :]
    squared_diff = (X_expanded - centers_expanded) ** 2
    return np.sqrt(np.sum(squared_diff, axis=2))
```

**IMG (Código - Activación gaussiana):**

```python
# Path: src/core/activation.py
class GaussianActivation(ActivationFunction):
    """
    Función de activación Gaussiana (RBF).
    
    Fórmula: phi(r, sigma) = exp(-(r/sigma)^2)
    """

    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return np.exp(-(distances / sigma) ** 2)
```

**Output (Ejemplo de salida):**

```
==== Entrenamiento ====
Entrenamiento completado en 0.0248 segundos
Error final: 0.021922
Épocas: 1
Tiempo: 0.0248s
```

---

### 9.7 Construcción de la Matriz de Interpolación

**Descripción:**

Se genera la matriz Φ (phi) de interpolación, que representa las activaciones de todas las neuronas RBF para todos los patrones de entrenamiento. Esta matriz es fundamental para el entrenamiento ya que captura la transformación no lineal del espacio de entrada al espacio de características RBF. La matriz Φ tiene filas iguales al número de patrones de entrenamiento y columnas iguales al número de neuronas RBF/centros. Cada elemento φ(i,j) representa la activación de la neurona j para el patrón i. Si se configura el uso de bias, se agrega una columna de unos a la matriz para incluir el término de bias en el cálculo de los pesos.

**Path:**
- `src/models/rbf/network.py` - Método `fit()` (construcción de Φ)
- `src/models/rbf/layer.py` - Método `forward()` (cálculo de activaciones)

**IMG (Código - Red RBF interna):**

```python
# Path: src/models/rbf/network.py
class RBFNetwork(BaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray, centers: np.ndarray = None) -> None:
        """
        Entrenar la red RBF.
        """
        # ... inicialización de centros y capa RBF
        
        # Calcular matriz de diseño (activaciones de capa oculta)
        Phi = self.rbflayer.forward(X)
        
        # Agregar columna de bias si está configurado
        if self.config.use_bias:
            Phi_bias = np.column_stack([Phi, np.ones(Phi.shape[0])])
        else:
            Phi_bias = Phi
        
        # Phi_bias tiene forma (n_samples, n_centers + 1)
        # ... continuación con cálculo de pesos
```

**IMG (Código - Capa RBF):**

```python
# Path: src/models/rbf/layer.py
class RBFLayer(BaseLayer):
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calcular el pase hacia adelante de la capa RBF.
        
        La salida se calcula como: Y = phi(d(X, C), sigma)
        
        Returns:
            Matriz de activación de forma (n_samples, n_centers)
        """
        # Calcular distancias euclidianas entre todas las muestras y todos los centros
        distances = euclidean_distance_matrix(X, self.centers)
        
        # Aplicar función de activación para obtener la salida de la capa
        activations = self.activation.compute(distances, self.sigma)
        
        return activations
```

**Output (Ejemplo de salida):**

```
Entrenamiento completado en 0.0248 segundos
```

---

### 9.8 Cálculo de la Matriz de Pesos

**Descripción:**

Se determinan los pesos de salida resolviendo el sistema lineal ΦW = Y. La solución se obtiene mediante pseudoinversa de Moore-Penrose, que proporciona la solución de mínimos cuadrados. Se puede aplicar regularización Tikhonov para mejorar la estabilidad numérica. Sin regularización, la solución es W = Φ⁺ Y (pseudoinversa directa). Con regularización, la solución es W = (ΦᵀΦ + λI)⁻¹ Φᵀ Y (regularización Tikhonov), donde λ es el parámetro de regularización e I es la matriz identidad. Este método de solución de forma cerrada es una de las principales ventajas de las redes RBF sobre las redes neuronales tradicionales que requieren entrenamiento iterativo.

**Path:**
- `src/models/rbf/network.py` - Método `fit()` (llamada a solve_pseudoinverse)
- `src/models/rbf/solver.py` - Función `solve_pseudoinverse()`

**IMG (Código - Red RBF interna):**

```python
# Path: src/models/rbf/network.py
class RBFNetwork(BaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray, centers: np.ndarray = None) -> None:
        """
        Entrenar la red RBF.
        """
        # ... cálculo de matriz de diseño Phi_bias
        
        # Resolver para pesos de salida usando pseudoinversa
        self.weights = solve_pseudoinverse(
            Phi_bias,
            y,
            regularization=self.config.regularization
        )
        
        # Extraer bias si se usa
        if self.config.use_bias:
            self.bias = self.weights[-1, :]
            self.weights = self.weights[:-1, :]
        else:
            self.bias = np.zeros(self.n_outputs_)
        
        self.is_fitted = True
```

**IMG (Código - Solucionador de pseudoinversa):**

```python
# Path: src/models/rbf/solver.py
def solve_pseudoinverse(Phi: np.ndarray, y: np.ndarray, regularization: float = 0.0) -> np.ndarray:
    """
    Resolver para pesos de salida usando pseudoinversa con regularización opcional.
    
    La capa de salida de red RBF computa: y = Phi @ W
    donde Phi es la matriz de diseño (activaciones de la capa oculta)
    y W son los pesos de salida que necesitamos encontrar.
    
    La solución es: W = pinv(Phi) @ y
    Con regularización: W = pinv(Phi.T @ Phi + lambda * I) @ Phi.T @ y
    
    Args:
        Phi: Matriz de diseño de forma (n_samples, n_centers)
        y: Salidas objetivo de forma (n_samples, n_outputs)
        regularization: Parámetro de regularización (lambda)
    
    Returns:
        Matriz de pesos de forma (n_centers, n_outputs)
    """
    n_samples, n_centers = Phi.shape
    
    if regularization > 0:
        # Pseudoinversa regularizada: (Phi.T @ Phi + lambda * I)^-1 @ Phi.T @ y
        Phi_T_Phi = Phi.T @ Phi
        identity = np.eye(n_centers)
        regularized_matrix = Phi_T_Phi + regularization * identity
        pseudoinverse = np.linalg.inv(regularized_matrix) @ Phi.T
    else:
        # Pseudoinversa estándar
        pseudoinverse = np.linalg.pinv(Phi)
    
    # Calcular pesos
    W = pseudoinverse @ y
    
    return W
```

**Output (Ejemplo de salida):**

```
Entrenamiento completado en 0.0248 segundos
Error final: 0.021922
```

---

### 9.9 Evaluación del Modelo (Simulación)

**Descripción:**

Se aplica el modelo entrenado sobre datos no vistos (validación y prueba). El proceso de simulación/evaluación consiste en repetir el cálculo de distancias entre los nuevos patrones y los centros entrenados, aplicar la función de activación para obtener las activaciones de la capa oculta, y generar predicciones usando los pesos obtenidos durante el entrenamiento. La predicción se calcula como y_pred = Φ @ W + b, donde Φ es la matriz de activaciones para los nuevos datos, W son los pesos de salida y b es el bias. Nota: "simular" y "evaluar" corresponden al mismo proceso en este contexto (inferencia sobre nuevos datos).

**Path:**
- `api/neural_network.py` - Método `predict()`
- `src/models/rbf/network.py` - Método `predict()`
- `scripts/train_dataset_rbf_1.py` - Función `evaluate_model()`

**IMG (Código - API NeuralNetwork):**

```python
# Path: api/neural_network.py
class NeuralNetwork:
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Simular/predecir con la red entrenada.
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            
        Returns:
            Predicciones de forma (n_samples, n_outputs)
        """
        self._ensure_fitted()
        X, _ = InputValidator.validate_input_pair(X)
        
        return self.model.predict(X)
```

**IMG (Código - Red RBF interna):**

```python
# Path: src/models/rbf/network.py
class RBFNetwork(BaseModel):
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Hacer predicciones para datos de entrada X.
        
        La predicción se calcula como: y_pred = Phi @ W + b
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
        
        Returns:
            Matriz de predicciones de forma (n_samples, n_outputs)
        """
        if not self.is_fitted:
            raise NotFittedError("El modelo debe ser ajustado antes de hacer predicciones")
        
        X = np.asarray(X)
        
        if X.ndim != 2:
            raise InvalidInputError(f"X debe ser array 2D, se obtuvo forma {X.shape}")
        
        if X.shape[1] != self.n_features_:
            raise InvalidInputError(
                f"Se esperaban {self.n_features_} características, se obtuvieron {X.shape[1]}"
            )
        
        # Calcular activaciones de capa oculta
        hidden_output = self.rbflayer.forward(X)
        
        # Calcular salida: hidden_output * weights + bias
        predictions = hidden_output @ self.weights + self.bias
        
        return predictions
```

**IMG (Código - Script de entrenamiento):**

```python
# Path: scripts/train_dataset_rbf_1.py
def evaluate_model(net, X, y, label, verbose_decimals=False):
    """
    Evaluar modelo y mostrar métricas.
    """
    print(f"\n==== {label} ====")
    metrics = net.evaluate(X, y)
    print(f"MSE: {format_float(metrics.mse, verbose_decimals)}, R²: {format_float(metrics.r2, verbose_decimals)}")
    pred = net.predict(X)
    result = net.confusion_matrix(y, pred)
    print(f"\nMatriz de Confusión:\n{result.matrix}")
    print(f"\nAccuracy global: {format_float(result.accuracy, verbose_decimals)}")
    # ... más métricas
```

**Output (Ejemplo de salida):**

```
==== Validation ====
MSE: 0.0258, R²: 0.8896

Matriz de Confusión:
[[54  2]
 [ 0 94]]

Accuracy global: 0.9867
```

---

### 9.10 Métricas de Evaluación / Matriz de Confusión

**Descripción:**

Se implementan métricas para medir el desempeño del modelo en tareas de clasificación. Se calcula la matriz de confusión que muestra los verdaderos positivos (TP), verdaderos negativos (TN), falsos positivos (FP) y falsos negativos (FN) por clase. A partir de esta matriz se derivan métricas como: Precisión (TP / (TP + FP)) que mide la exactitud de predicciones positivas; Recall o Sensibilidad (TP / (TP + FN)) que mide la capacidad de detectar positivos; Specificity o Especificidad (TN / (TN + FP)) que mide la capacidad de detectar negativos; y F1-score (2 * (Precision * Recall) / (Precision + Recall)) que es la media armónica de precisión y recall. También se calculan promedios macro (promedio simple por clase) y ponderado (promedio ponderado por número de muestras por clase). Para problemas de regresión se calculan métricas como MSE, RMSE y R².

**Path:**
- `api/neural_network.py` - Métodos `evaluate()` y `confusion_matrix()`
- `src/evaluation/confusion_matrix.py` - Clase `ConfusionMatrixCalculator`
- `scripts/train_dataset_rbf_1.py` - Función `evaluate_model()`

**IMG (Código - API NeuralNetwork):**

```python
# Path: api/neural_network.py
class NeuralNetwork:
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        detailed: bool = False
    ) -> EvaluationResult:
        """
        Evaluar el modelo en datos de prueba.

        Args:
            X: Matriz de entrada de prueba
            y: Valores objetivo de prueba
            detailed: Si devolver métricas detalladas

        Returns:
            EvaluationResult con métricas de evaluación
        """
        self._ensure_fitted()
        X, y = InputValidator.validate_input_pair(X, y)

        report = self.evaluator.evaluate(self.model, X, y)

        return EvaluationResult(
            mse=report.mse,
            mae=report.mae,
            rmse=report.rmse,
            r2=report.r2,
            accuracy=report.accuracy,
            predictions=report.predictions if detailed else None,
            metadata=report.metadata if detailed else {}
        )
    
    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None
    ) -> Union[ConfusionMatrixResult, Dict[int, ConfusionMatrixResult]]:
        """
        Calcular matriz de confusión y métricas derivadas.
        
        Args:
            y_true: Valores verdaderos
            y_pred: Valores predichos opcionales
            X: Datos de entrada opcionales para generar predicciones
            
        Returns:
            ConfusionMatrixResult con matriz y métricas
        """
        # Si no se proporciona y_pred, hacer predicciones
        if y_pred is None:
            if X is None:
                raise ValueError("Debes proporcionar y_pred o X para calcular la matriz de confusión.")
            self._ensure_fitted()
            y_pred = self.predict(X)
        
        # Usar el evaluador para calcular la matriz de confusión
        return self.evaluator.confusion_matrix(y_true, y_pred)
```

**IMG (Código - Script de entrenamiento):**

```python
# Path: scripts/train_dataset_rbf_1.py
def evaluate_model(net, X, y, label, verbose_decimals=False):
    """
    Evaluar modelo y mostrar métricas.
    """
    print(f"\n==== {label} ====")
    
    # Métricas de regresión
    metrics = net.evaluate(X, y)
    print(f"MSE: {format_float(metrics.mse, verbose_decimals)}, R²: {format_float(metrics.r2, verbose_decimals)}")
    
    # Predicción y matriz de confusión
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
```

**Output (Ejemplo de salida):**

```
==== Test ====
MSE: 0.0304, R²: 0.8762

Matriz de Confusión:
[[83  2]
 [ 0 65]]

Accuracy global: 0.9867

Métricas por clase:
  Precision: {'0': 1.0, '1': 0.9701}
  Recall (Sensibilidad): {'0': 0.9765, '1': 1.0}
  Specificity (Especificidad): {'0': 1.0, '1': 0.9765}
  F1-score: {'0': 0.9881, '1': 0.9848}

Promedios:
  Macro: {'precision': 0.9851, 'recall': 0.9882, 'specificity': 0.9882, 'f1-score': 0.9865}
  Ponderado: {'precision': 0.9871, 'recall': 0.9867, 'specificity': 0.9898, 'f1-score': 0.9867}
```

---

## Resumen del Flujo Completo

```
1. Dataset JSON → load_json_data() → clean_data() → X, y arrays
2. show_input_parameters() → Parámetros y estadísticas
3. split_data(70/15/15) → Train/Val/Test
4. NeuralNetwork(model_type=RBF) → Configuración (n_centers, sigma, activation)
5. net.train() → RBFNetwork.fit() → Inicializar centros → Calcular distancias → Activaciones → Φ → Pesos
6. net.predict() → RBFNetwork.predict() → Inferencia sobre nuevos datos
7. net.evaluate() → Métricas de regresión (MSE, R²)
8. net.confusion_matrix() → Matriz de confusión y métricas de clasificación
```

---

## Archivos Principales Referenciados

| Archivo | Descripción |
|---------|-------------|
| `jsons/dataset_rbf_*.json` | Datasets en formato JSON |
| `scripts/train_dataset_rbf_*.py` | Scripts de entrenamiento |
| `api/neural_network.py` | API pública (NeuralNetwork) |
| `src/models/rbf/config.py` | Configuración de parámetros RBF |
| `src/models/rbf/network.py` | Red RBF completa (RBFNetwork) |
| `src/models/rbf/layer.py` | Capa oculta RBF (RBFLayer) |
| `src/models/rbf/solver.py` | Resolución de pesos (solve_pseudoinverse) |
| `src/training/initializer.py` | Inicialización de centros |
| `src/core/distance.py` | Cálculo de distancias euclidianas |
| `src/core/activation.py` | Funciones de activación |
| `src/evaluation/confusion_matrix.py` | Matriz de confusión y métricas |

---

# Apéndice: Recomendaciones y Usos Avanzados

Este apéndice contiene secciones adicionales que complementan la documentación principal. Aquí se cubren temas avanzados como funciones de activación alternativas, uso directo de la API, persistencia de modelos, inspección de pesos, y comparación de hiperparámetros.

---

## A. Funciones de Activación RBF Alternativas

### A.1 Descripción

Además de la función gaussiana, el sistema soporta múltiples funciones de activación RBF. Cada una tiene características diferentes que pueden ser más adecuadas dependiendo de la distribución de los datos y el problema específico.

### A.2 Funciones Disponibles

| Función | Fórmula | Características | Cuándo usar |
|---------|---------|-----------------|-------------|
| **Gaussiana** | φ(r) = exp(-(r/σ)²) | Suave, localizada, decrece rápidamente | Datos con clusters compactos (default) |
| **Multiccuadrática** | φ(r) = √(1 + (r/σ)²) | Crece con la distancia, no localizada | Datos dispersos, tendencias globales |
| **Multiccuadrática Inversa** | φ(r) = 1/√(1 + (r/σ)²) | Decrece suavemente, más plana que gaussiana | Datos con ruido, suavizado fuerte |
| **Thin Plate Spline (ln)** | φ(r) = r² ln(r) | Interpolación exacta en puntos de entrenamiento (forma original) | Interpolación, no aproximación |
| **Thin Plate Spline (log10)** | φ(r) = r² log₁₀(r) | Variante con logaritmo base 10 | Interpolación alternativa |

### A.3 Código de Implementación

**Path:** `src/core/activation.py`

```python
class GaussianActivation(ActivationFunction):
    """Función de activación Gaussiana."""
    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return np.exp(-(distances / sigma) ** 2)

class MultiquadraticActivation(ActivationFunction):
    """Función de activación Multiccuadrática."""
    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return np.sqrt(1 + (distances / sigma) ** 2)

class InverseMultiquadraticActivation(ActivationFunction):
    """Función de activación Multiccuadrática Inversa."""
    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return 1 / np.sqrt(1 + (distances / sigma) ** 2)

class ThinPlateSplineActivation(ActivationFunction):
    """
    Función de activación Thin Plate Spline (RBF) - Variante con logaritmo natural.
    
    Fórmula: phi(r) = r^2 * ln(r), con phi(0) = 0
    Usa logaritmo natural (np.log), que es la forma original de Thin Plate Spline.
    """
    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        scaled_distances = distances / sigma
        result = np.zeros_like(scaled_distances)
        mask = scaled_distances > 0
        result[mask] = scaled_distances[mask] ** 2 * np.log(scaled_distances[mask])
        return result

class ThinPlateSplineLog10Activation(ActivationFunction):
    """
    Función de activación Thin Plate Spline (RBF) - Variante con logaritmo base 10.
    
    Fórmula: phi(r) = r^2 * log10(r), con phi(0) = 0
    Usa logaritmo base 10 (np.log10), variante alternativa de Thin Plate Spline.
    """
    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        scaled_distances = distances / sigma
        result = np.zeros_like(scaled_distances)
        mask = scaled_distances > 0
        result[mask] = scaled_distances[mask] ** 2 * np.log10(scaled_distances[mask])
        return result
```

### A.4 Ejemplos de Uso

```python
# Usar función multicuadrática
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=50,
    sigma=1.0,
    activation_rbf='multiquadratic',  # Cambiar función
    regularization=0.01
)

# Usar función inversa multicuadrática
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=50,
    sigma=1.0,
    activation_rbf='inverse_multiquadratic',
    regularization=0.01
)

# Comparar diferentes funciones
functions = ['gaussian', 'multiquadratic', 'inverse_multiquadratic']
for func in functions:
    net = NeuralNetwork(
        model_type=ModelType.RBF,
        n_centers=50,
        sigma=1.0,
        activation_rbf=func,
        regularization=0.01
    )
    net.train(X_train, y_train)
    metrics = net.evaluate(X_val, y_val)
    print(f"{func}: MSE = {metrics.mse:.4f}")
```

---

## B. Uso Directo de la API (Sin Script)

### B.1 Descripción

Aunque los scripts de entrenamiento son convenientes, también puedes usar la API `NeuralNetwork` directamente en tu código Python para mayor flexibilidad. Esto es útil cuando necesitas integrar el modelo en una aplicación más grande o automatizar experimentos.

### B.2 Ejemplo Básico

```python
import numpy as np
from api.neural_network import NeuralNetwork
from api.core.model_type import ModelType

# Crear datos de ejemplo (o cargar desde cualquier fuente)
X_train = np.random.randn(100, 4)  # 100 muestras, 4 features
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int).reshape(-1, 1)  # Clasificación binaria

X_val = np.random.randn(20, 4)
y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int).reshape(-1, 1)

# Crear y entrenar modelo
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=20,
    sigma=1.0,
    activation_rbf='gaussian',
    regularization=0.01,
    random_state=42
)

# Entrenar
result = net.train(X_train, y_train, verbose=True)
print(f"Entrenamiento completado en {result.training_time:.4f}s")

# Evaluar
metrics = net.evaluate(X_val, y_val)
print(f"MSE: {metrics.mse:.4f}, R²: {metrics.r2:.4f}")

# Predecir
predictions = net.predict(X_val)
```

### B.3 Ejemplo con Pipeline Completo

```python
import numpy as np
from api.neural_network import NeuralNetwork
from api.core.model_type import ModelType
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generar datos sintéticos
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_classes=2,
    n_redundant=0,
    n_informative=4,
    random_state=42
)
y = y.reshape(-1, 1)

# Preprocesamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Partición manual
train_size = int(0.7 * len(X_scaled))
val_size = int(0.15 * len(X_scaled))

X_train = X_scaled[:train_size]
y_train = y[:train_size]
X_val = X_scaled[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X_scaled[train_size + val_size:]
y_test = y[train_size + val_size:]

# Crear modelo con configuración personalizada
net = NeuralNetwork(
    model_type=ModelType.RBF,
    n_centers=50,
    sigma=0.8,
    activation_rbf='gaussian',
    regularization=0.005,
    use_bias=True,
    random_state=42
)

# Entrenar
result = net.train(X_train, y_train, verbose=True)

# Evaluar en todos los conjuntos
for name, X, y in [('Train', X_train, y_train), 
                    ('Validation', X_val, y_val), 
                    ('Test', X_test, y_test)]:
    metrics = net.evaluate(X, y)
    print(f"{name}: MSE={metrics.mse:.4f}, Acc={metrics.accuracy:.4f}")
```

---

## C. Guardar y Cargar Modelos

### C.1 Descripción

Los modelos entrenados pueden guardarse en disco para uso posterior sin necesidad de reentrenar. Esto es útil para despliegue en producción o para compartir modelos entre diferentes aplicaciones.

### C.2 Métodos de Persistencia

**Path:** `api/neural_network.py`

```python
class NeuralNetwork:
    def save(self, filepath: str) -> None:
        """
        Guardar el modelo entrenado en disco.
        
        Args:
            filepath: Ruta donde guardar el modelo (ej: 'model.pkl')
        """
        self._ensure_fitted()
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'config': self.config,
                'model': self.model,
                'training_log': self.training_log
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'NeuralNetwork':
        """
        Cargar un modelo guardado desde disco.
        
        Args:
            filepath: Ruta del modelo guardado (ej: 'model.pkl')
            
        Returns:
            Instancia de NeuralNetwork con el modelo cargado
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Crear nueva instancia sin entrenar
        instance = cls(
            model_type=data['model_type'],
            config=data['config']
        )
        
        # Restaurar estado entrenado
        instance.model = data['model']
        instance.training_log = data['training_log']
        
        return instance
```

### C.3 Ejemplos de Uso

```python
# Guardar modelo entrenado
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=50, sigma=1.0)
net.train(X_train, y_train)
net.save('models/rbf_model.pkl')

# Cargar modelo en otra aplicación
from api.neural_network import NeuralNetwork

loaded_net = NeuralNetwork.load('models/rbf_model.pkl')

# Usar directamente sin reentrenar
predictions = loaded_net.predict(X_new_data)
metrics = loaded_net.evaluate(X_test, y_test)

# Ver historial de entrenamiento
print(loaded_net.training_log)
```

### C.4 Estructura del Archivo Guardado

```
model.pkl (archivo pickle)
├── model_type: ModelType.RBF
├── config: NeuralNetworkConfig
│   ├── n_centers
│   ├── sigma
│   ├── activation_rbf
│   ├── regularization
│   └── ...
├── model: RBFNetwork (estado entrenado)
│   ├── centers: np.ndarray
│   ├── weights: np.ndarray
│   ├── bias: np.ndarray
│   └── is_fitted: True
└── training_log: dict
    ├── error_history: list
    ├── training_time: float
    └── epochs: int
```

---

## D. Obtener Pesos y Centros del Modelo

### D.1 Descripción

Después del entrenamiento, puedes inspeccionar los pesos y centros aprendidos por el modelo. Esto es útil para análisis, visualización o para entender qué características son más importantes.

### D.2 Métodos de Inspección

**Path:** `api/neural_network.py`

```python
class NeuralNetwork:
    def get_weights(self) -> Dict[str, Any]:
        """
        Obtener los pesos del modelo.
        
        Returns:
            Diccionario con pesos, bias y centros
        """
        self._ensure_fitted()
        
        if self.model_type == ModelType.RBF:
            return {
                'weights': self.model.weights,      # Pesos de salida (n_centers, n_outputs)
                'bias': self.model.bias,            # Bias (n_outputs,)
                'centers': self.model.centers       # Centros RBF (n_centers, n_features)
            }
    
    def get_layer_info(self) -> List[Dict[str, Any]]:
        """
        Obtener información detallada de todas las capas.
        
        Returns:
            Lista de diccionarios con información de cada capa
        """
        self._ensure_fitted()
        
        if self.model_type == ModelType.RBF:
            return [{
                'layer_type': 'rbf',
                'n_centers': self.model.config.n_centers,
                'activation': str(self.model.config.activation),
                'sigma': self.model.config.sigma,
                'weights_shape': self.model.weights.shape,
                'centers_shape': self.model.centers.shape
            }]
```

### D.3 Ejemplos de Uso

```python
# Entrenar modelo
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=50, sigma=1.0)
net.train(X_train, y_train)

# Obtener pesos y centros
weights_info = net.get_weights()

print("Pesos de salida:")
print(weights_info['weights'])  # Forma: (50, 1)

print("\nBias:")
print(weights_info['bias'])  # Forma: (1,)

print("\nCentros RBF:")
print(weights_info['centers'])  # Forma: (50, 4)

# Analizar distribución de centros
centers = weights_info['centers']
print(f"\nRango de centros por dimensión:")
for i in range(centers.shape[1]):
    print(f"  Feature {i}: [{centers[:, i].min():.2f}, {centers[:, i].max():.2f}]")

# Visualizar pesos
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(len(weights_info['weights'])), weights_info['weights'].flatten())
plt.title('Pesos de Salida')
plt.xlabel('Centro')
plt.ylabel('Peso')

plt.subplot(1, 2, 2)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', alpha=0.3)
plt.title('Centros RBF vs Datos')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(['Centros', 'Datos'])
plt.tight_layout()
plt.savefig('centros_y_pesos.png')
```

---

## E. Comparación de Hiperparámetros

### E.1 Descripción

Para encontrar la mejor configuración de hiperparámetros, puedes ejecutar búsqueda en cuadrícula (grid search) o búsqueda aleatoria probando diferentes combinaciones de `n_centers`, `sigma`, `activation_rbf` y `regularization`.

### E.2 Implementación de Grid Search

```python
import numpy as np
from api.neural_network import NeuralNetwork
from api.core.model_type import ModelType
import pandas as pd

# Definir rangos de hiperparámetros
param_grid = {
    'n_centers': [20, 50, 100],
    'sigma': [0.5, 1.0, 2.0],
    'regularization': [0.0, 0.01, 0.1],
    'activation_rbf': ['gaussian', 'multiquadratic']
}

# Resultados
def evaluate_config(n_centers, sigma, regularization, activation_rbf):
    """Evaluar una configuración específica."""
    net = NeuralNetwork(
        model_type=ModelType.RBF,
        n_centers=n_centers,
        sigma=sigma,
        regularization=regularization,
        activation_rbf=activation_rbf,
        random_state=42
    )
    
    # Entrenar
    net.train(X_train, y_train)
    
    # Evaluar
    train_metrics = net.evaluate(X_train, y_train)
    val_metrics = net.evaluate(X_val, y_val)
    
    return {
        'n_centers': n_centers,
        'sigma': sigma,
        'regularization': regularization,
        'activation_rbf': activation_rbf,
        'train_mse': train_metrics.mse,
        'val_mse': val_metrics.mse,
        'val_accuracy': val_metrics.accuracy
    }

# Ejecutar todas las combinaciones
results = []
for n_centers in param_grid['n_centers']:
    for sigma in param_grid['sigma']:
        for regularization in param_grid['regularization']:
            for activation_rbf in param_grid['activation_rbf']:
                print(f"Probando: n_centers={n_centers}, sigma={sigma}, "
                      f"reg={regularization}, act={activation_rbf}")
                result = evaluate_config(n_centers, sigma, regularization, activation_rbf)
                results.append(result)

# Crear DataFrame y ordenar por mejor validación
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('val_mse')

print("\nTop 5 configuraciones:")
print(results_df.head())

# Mejor configuración
best = results_df.iloc[0]
print(f"\nMejor configuración:")
print(f"  n_centers: {best['n_centers']}")
print(f"  sigma: {best['sigma']}")
print(f"  regularization: {best['regularization']}")
print(f"  activation_rbf: {best['activation_rbf']}")
print(f"  Val MSE: {best['val_mse']:.6f}")
```

### E.3 Visualización de Resultados

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap de sigma vs n_centers para cada activación
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, activation in enumerate(['gaussian', 'multiquadratic']):
    subset = results_df[results_df['activation_rbf'] == activation]
    
    # Crear matriz pivot
    pivot_data = subset.pivot_table(
        values='val_mse',
        index='sigma',
        columns='n_centers',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.4f', ax=axes[idx], cmap='viridis_r')
    axes[idx].set_title(f'Validation MSE - {activation}')

plt.tight_layout()
plt.savefig('hyperparameter_search.png')
```

### E.4 Búsqueda Aleatoria (Random Search)

```python
import numpy as np
from scipy.stats import uniform, randint

# Definir distribuciones aleatorias
random_params = {
    'n_centers': randint(10, 200),
    'sigma': uniform(0.1, 2.0),
    'regularization': uniform(0.0, 0.1)
}

# Ejecutar 20 configuraciones aleatorias
results_random = []
for i in range(20):
    n_centers = random_params['n_centers'].rvs()
    sigma = random_params['sigma'].rvs()
    regularization = random_params['regularization'].rvs()
    
    result = evaluate_config(n_centers, sigma, regularization, 'gaussian')
    results_random.append(result)

# Analizar resultados
results_random_df = pd.DataFrame(results_random)
results_random_df = results_random_df.sort_values('val_mse')
print("Mejor configuración encontrada:")
print(results_random_df.iloc[0])
```

---

## F. Resumen de Recomendaciones

### F.1 Guía Rápida de Selección

| Escenario | Recomendación | Ejemplo |
|-----------|---------------|---------|
| **Default/Primer intento** | Usar valores por defecto | `n_centers=50, sigma=1.0, activation='gaussian'` |
| **Overfitting** | Reducir centros, aumentar regularización | `n_centers=30, regularization=0.1` |
| **Underfitting** | Aumentar centros, reducir sigma | `n_centers=100, sigma=0.5` |
| **Datos ruidosos** | Usar función inversa multicuadrática | `activation='inverse_multiquadratic'` |
| **Datos dispersos** | Usar función multicuadrática | `activation='multiquadratic'` |
| **Reproducibilidad** | Fijar random_state | `random_state=42` |
| **Producción** | Guardar modelo entrenado | `net.save('model.pkl')` |

### F.2 Checklist de Buenas Prácticas

- [ ] Verificar que `n_centers` ≤ número de muestras de entrenamiento
- [ ] Elegir `sigma` proporcional al rango de los datos
- [ ] Usar `regularization > 0` para problemas mal condicionados
- [ ] Validar con conjunto de validación separado
- [ ] Guardar modelo entrenado para uso posterior
- [ ] Documentar configuración usada para reproducibilidad
- [ ] Visualizar centros para verificar distribución apropiada
- [ ] Comparar múltiples configuraciones cuando sea posible

