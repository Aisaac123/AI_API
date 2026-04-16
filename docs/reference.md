# Referencia de API de Redes Neuronales

Esta es una referencia detallada de todos los métodos y parámetros de la API de redes neuronales.

**Nota:** La API de este proyecto está inspirada en el estilo de MATLAB para facilitar su uso a usuarios familiarizados con ese entorno, pero esta guía se enfoca en la implementación Python sin hacer comparaciones directas.

## Tabla de Contenidos

1. [NeuralNetwork](#neuralnetwork)
2. [NeuralNetworkConfig](#neuralnetworkconfig)
3. [ModelType](#modeltype)

---

## NeuralNetwork

Clase principal de la API para crear y entrenar redes neuronales.

### Constructor

```python
NeuralNetwork(
    model_type: ModelType,
    config: Optional[NeuralNetworkConfig] = None,
    hidden_layers: Optional[List[int]] = None,
    n_centers: Optional[int] = None,
    sigma: float = 1.0,
    activation_rbf: str = 'gaussian',
    activation_backprop: str = 'sigmoid',
    learning_rate: float = 0.01,
    epochs: int = 1000,
    batch_size: int = 32,
    use_bias: bool = True,
    regularization: float = 0.01,
    random_state: int = 42,
    initializer: str = 'kmeans'
)
```

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `model_type` | `ModelType` | **Requerido** | Tipo de modelo: `ModelType.RBF` o `ModelType.BACKPROP` |
| `config` | `NeuralNetworkConfig` | `None` | Objeto de configuración (alternativa a parámetros individuales) |
| `hidden_layers` | `List[int]` | `None` | Lista de tamaños de capas ocultas (ej. `[10, 5]`) |
| `n_centers` | `int` | `None` | Número de centros RBF (ej. `20`) |
| `sigma` | `float` | `1.0` | Parámetro de ancho de RBF |
| `activation_rbf` | `str` | `'gaussian'` | Función de activación RBF: `'gaussian'`, `'multiquadratic'`, `'inverse_multiquadratic'`, `'thin_plate'` |
| `activation_backprop` | `str` | `'sigmoid'` | Función de activación por defecto para backprop: `'sigmoid'`, `'logsig'`, `'tanh'`, `'tansig'`, `'relu'`, `'linear'`, `'purelin'`, `'leaky_relu'` |
| `learning_rate` | `float` | `0.01` | Tasa de aprendizaje para backpropagation |
| `epochs` | `int` | `1000` | Número de épocas de entrenamiento |
| `batch_size` | `int` | `32` | Tamaño de lote para mini-batch training |
| `use_bias` | `bool` | `True` | Si usar términos de bias |
| `regularization` | `float` | `0.01` | Parámetro de regularización |
| `random_state` | `int` | `42` | Semilla aleatoria para reproducibilidad |
| `initializer` | `str` | `'kmeans'` | Estrategia de inicialización: `'kmeans'` o `'random'` |

**Retorna:** Instancia de `NeuralNetwork`

**Ejemplo:**
```python
from api import NeuralNetwork, ModelType

# RBF
net_rbf = NeuralNetwork(model_type=ModelType.RBF, n_centers=20, sigma=0.5)

# Backpropagation
net_bp = NeuralNetwork(
    model_type=ModelType.BACKPROP,
    hidden_layers=[10, 5],
    learning_rate=0.01,
    epochs=1000
)
```

---

### train()

Entrena la red neuronal con los datos proporcionados.

```python
train(
    X: np.ndarray,
    y: np.ndarray,
    verbose: bool = False
) -> Dict[str, Any]
```

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | **Requerido** | Matriz de entrada de forma `(n_samples, n_features)` |
| `y` | `np.ndarray` | **Requerido** | Matriz de salida de forma `(n_samples, n_outputs)` |
| `verbose` | `bool` | `False` | Si imprimir progreso de entrenamiento |

**Retorna:** `Dict[str, Any]` con las siguientes claves:
- `training_time`: `float` - Tiempo de entrenamiento en segundos
- `final_error`: `float` - Error final de entrenamiento
- `epochs`: `int` - Número de épocas ejecutadas
- `error_history`: `List[float]` - Historial de errores por época
- `converged`: `bool` - Si el modelo convergió
- `metadata`: `Dict[str, Any]` - Metadatos adicionales

**Ejemplo:**
```python
import numpy as np
from api import NeuralNetwork, ModelType

X = np.random.randn(100, 2)
y = np.random.randn(100, 1)

net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)
result = net.train(X, y, verbose=True)

print(f"Tiempo de entrenamiento: {result['training_time']:.4f}s")
print(f"Error final: {result['final_error']:.6f}")
```

---

### predict()

Realiza predicciones con el modelo entrenado.

```python
predict(X: np.ndarray) -> np.ndarray
```

**Parámetros:**

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `X` | `np.ndarray` | Matriz de entrada de forma `(n_samples, n_features)` |

**Retorna:** `np.ndarray` de forma `(n_samples, n_outputs)` con las predicciones

**Excepciones:**
- `RuntimeError`: Si el modelo no ha sido entrenado

**Ejemplo:**
```python
X_test = np.random.randn(20, 2)
predictions = net.predict(X_test)
print(f"Predicciones: {predictions}")
```

---

### evaluate()

Evalúa el modelo en datos de prueba.

```python
evaluate(
    X: np.ndarray,
    y: np.ndarray,
    detailed: bool = False
) -> Dict[str, Any]
```

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | **Requerido** | Matriz de entrada de prueba |
| `y` | `np.ndarray` | **Requerido** | Valores objetivo de prueba |
| `detailed` | `bool` | `False` | Si incluir predicciones y metadatos en el resultado |

**Retorna:** `Dict[str, Any]` con las siguientes claves:
- `mse`: `float` - Error cuadrático medio
- `mae`: `float` - Error absoluto medio
- `rmse`: `float` - Raíz del error cuadrático medio
- `r2`: `float` - Coeficiente R²
- `accuracy`: `float` - Precisión (para clasificación)
- `predictions`: `np.ndarray` (opcional) - Predicciones (si `detailed=True`)
- `metadata`: `Dict[str, Any]` (opcional) - Metadatos adicionales (si `detailed=True`)

**Ejemplo:**
```python
X_test = np.random.randn(20, 2)
y_test = np.random.randn(20, 1)

metrics = net.evaluate(X_test, y_test)
print(f"MSE: {metrics['mse']:.6f}")
print(f"R²: {metrics['r2']:.6f}")
```

---

### get_weights()

Obtiene todos los pesos del modelo.

```python
get_weights() -> Dict[str, Any]
```

**Retorna:** `Dict[str, Any]` con los pesos del modelo:
- Para RBF: `'weights'`, `'bias'`, `'centers'`
- Para Backprop: Diccionario con `'layer_i'` para cada capa, cada uno con `'weights'` y `'bias'`

**Ejemplo:**
```python
weights = net.get_weights()
print(f"Pesos: {weights}")
```

---

### get_layer_weights()

Obtiene pesos y bias de una capa específica (solo para backpropagation).

```python
get_layer_weights(layer_index: int) -> Dict[str, Any]
```

**Parámetros:**

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `layer_index` | `int` | Índice de la capa (`0` para primera capa oculta, `-1` para capa de salida) |

**Retorna:** `Dict[str, Any]` con:
- `layer_index`: `int` - Índice de la capa
- `layer_type`: `str` - Tipo de capa (`'hidden'` o `'output'`)
- `input_size`: `int` - Tamaño de entrada
- `output_size`: `int` - Tamaño de salida
- `weights`: `np.ndarray` - Matriz de pesos (copia)
- `bias`: `np.ndarray` - Vector de bias (copia)
- `activation`: `str` - Función de activación
- `use_bias`: `bool` - Si usa bias

**Excepciones:**
- `RuntimeError`: Si el modelo no está entrenado
- `IndexError`: Si el índice de capa es inválido
- `ValueError`: Si el modelo es RBF

**Ejemplo:**
```python
# Obtener pesos de la primera capa oculta
layer_0 = net.get_layer_weights(0)
print(f"Pesos capa 0: {layer_0['weights'].shape}")

# Obtener pesos de la capa de salida
layer_output = net.get_layer_weights(-1)
print(f"Pesos salida: {layer_output['weights'].shape}")
```

---

### get_layer_info()

Obtiene información detallada de todas las capas del modelo.

```python
get_layer_info() -> List[Dict[str, Any]]
```

**Retorna:** `List[Dict[str, Any]]` con información de cada capa:
- `layer_index`: `int` - Índice de la capa
- `layer_type`: `str` - Tipo de capa
- `input_size`: `int` - Tamaño de entrada
- `output_size`: `int` - Tamaño de salida
- `activation`: `str` - Función de activación
- `use_bias`: `bool` - Si usa bias
- `weights_shape`: `Tuple[int, int]` - Forma de los pesos
- `bias_shape`: `Tuple[int]` (opcional) - Forma del bias

**Ejemplo:**
```python
layer_info = net.get_layer_info()
for layer in layer_info:
    print(f"Capa {layer['layer_index']}: {layer['layer_type']}")
    print(f"  Activación: {layer['activation']}")
    print(f"  Forma: {layer['input_size']} -> {layer['output_size']}")
```

---

### summary()

Obtiene un resumen del modelo.

```python
summary() -> Dict[str, Any]
```

**Retorna:** `Dict[str, Any]` con información del modelo:
- `model_type`: `str` - Tipo de modelo
- `is_fitted`: `bool` - Si el modelo está entrenado
- `config`: `Dict[str, Any]` - Configuración del modelo
- Información adicional específica del tipo de modelo

**Ejemplo:**
```python
summary = net.summary()
print(f"Tipo: {summary['model_type']}")
print(f"Entrenado: {summary['is_fitted']}")
```

---

## NeuralNetworkConfig

Clase de configuración estructurada para la API.

### Constructor

```python
NeuralNetworkConfig(
    hidden_layers: List[int] = None,
    n_centers: Optional[int] = None,
    sigma: float = 1.0,
    activation_rbf: str = 'gaussian',
    activation_backprop: str = 'sigmoid',
    layer_activations: Optional[List[str]] = None,
    output_activation: str = 'linear',
    learning_rate: float = 0.01,
    epochs: int = 1000,
    batch_size: int = 32,
    use_bias: bool = True,
    regularization: float = 0.01,
    random_state: int = 42,
    initializer: str = 'kmeans'
)
```

**Parámetros adicionales (no incluidos en NeuralNetwork directo):**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `layer_activations` | `List[str]` | `None` | Lista de funciones de activación por capa (ej. `['tansig', 'logsig', 'purelin']`). Si es `None`, usa `activation_backprop` para todas las capas |
| `output_activation` | `str` | `'linear'` | Función de activación de la capa de salida |

**Métodos:**

- `validate()`: Valida los parámetros de configuración
- `to_dict()`: Convierte la configuración a diccionario

**Ejemplo:**
```python
from api import NeuralNetworkConfig

config = NeuralNetworkConfig(
    hidden_layers=[10, 5],
    layer_activations=['tansig', 'logsig'],
    output_activation='purelin',
    learning_rate=0.02
)

net = NeuralNetwork(model_type=ModelType.BACKPROP, config=config)
```

---

## ModelType

Enumeración para seleccionar el tipo de modelo.

### Valores

- `ModelType.RBF`: Red de Funciones de Base Radial
- `ModelType.BACKPROP`: Red de Retropropagación

**Ejemplo:**
```python
from api import ModelType

net_rbf = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)
net_bp = NeuralNetwork(model_type=ModelType.BACKPROP, hidden_layers=[10])
```

---

## Funciones de Activación Disponibles

### Para Backpropagation

| Nombre | Descripción |
|--------|-------------|
| `'sigmoid'` | Sigmoide: 1 / (1 + exp(-z)) |
| `'tanh'` | Tangente hiperbólica: tanh(z) |
| `'relu'` | Rectified Linear Unit: max(0, z) |
| `'linear'` | Lineal: z |
| `'leaky_relu'` | Leaky ReLU: max(0.01*z, z) |

### Para RBF

| Nombre | Descripción |
|--------|-------------|
| `'gaussian'` | Gaussiana: exp(-(r/sigma)^2) |
| `'multiquadratic'` | Multicuadrática: sqrt(1 + (r/sigma)^2) |
| `'inverse_multiquadratic'` | Multicuadrática Inversa: 1 / sqrt(1 + (r/sigma)^2) |
| `'thin_plate'` | Thin Plate Spline: r^2 * ln(r) |

---

## Ejemplos Completos

### RBF Básico

```python
from api import NeuralNetwork, ModelType
import numpy as np

X = np.random.randn(100, 2)
y = np.random.randn(100, 1)

net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20, sigma=0.5)
net.train(X, y, verbose=True)
predictions = net.predict(X)
metrics = net.evaluate(X, y)
```

### Backpropagation con Funciones por Capa

```python
from api import NeuralNetwork, ModelType, NeuralNetworkConfig
import numpy as np

X = np.random.randn(100, 2)
y = np.random.randn(100, 1)

config = NeuralNetworkConfig(
    hidden_layers=[10, 5],
    layer_activations=['tansig', 'logsig'],
    output_activation='purelin',
    learning_rate=0.01,
    epochs=100
)

net = NeuralNetwork(model_type=ModelType.BACKPROP, config=config)
net.train(X, y, verbose=True)

# Ver información de las capas
layer_info = net.get_layer_info()
for layer in layer_info:
    print(f"Capa {layer['layer_index']}: {layer['activation']}")

# Ver pesos de una capa específica
layer_weights = net.get_layer_weights(0)
print(f"Pesos primera capa: {layer_weights['weights'].shape}")
```

### Inspección de Pesos y Bias

```python
# Obtener todos los pesos
all_weights = net.get_weights()

# Obtener pesos de una capa específica
layer_0 = net.get_layer_weights(0)
layer_output = net.get_layer_weights(-1)

# Obtener información de todas las capas
layer_info = net.get_layer_info()
```

---

## Notas Importantes

1. **Formas de Arrays**: `X` debe ser 2D de forma `(n_samples, n_features)`, `y` puede ser 1D o 2D
2. **Entrenamiento Previo**: Los métodos `predict()`, `evaluate()`, `get_weights()`, `get_layer_weights()`, `get_layer_info()` requieren que el modelo haya sido entrenado primero con `train()`
3. **Índices de Capa**: Se pueden usar índices negativos para acceder a capas desde el final (`-1` para última capa)
