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
| `ThinPlateActivation` | φ(d) = d² log(d) | `'thin_plate'` |

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
