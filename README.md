# Implementación de Redes RBF y Backpropagation

Este proyecto implementa redes neuronales de Funciones de Base Radial (RBF) y redes neuronales de retropropagación en Python con una arquitectura modular limpia y una API compacta.

## Inicio Rápido (API Compacta)

La forma recomendada de usar este proyecto es a través de la API compacta:

```python
from api import NeuralNetwork, ModelType
import numpy as np

# Crear y entrenar red RBF
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20, sigma=0.5)
net.train(X_train, y_train, verbose=True)
predictions = net.predict(X_test)
metrics = net.evaluate(X_test, y_test)

# Crear y entrenar red de retropropagación
net2 = NeuralNetwork(model_type=ModelType.BACKPROP, hidden_layers=[20, 10], epochs=1000)
net2.train(X_train, y_train, verbose=True)
predictions2 = net2.predict(X_test)
```

## Instalación

1. Activar el entorno virtual:
```bash
.venv/Scripts/activate
```

2. Instalar dependencias:
```bash
py -m pip install numpy
```

## Documentación

- **[Guía Técnica](docs/guide.md)** - Explicación detallada de arquitectura, modelos matemáticos, y flujo de la aplicación
- **[Operaciones Matemáticas](docs/mathematical_operations.md)** - Documentación completa de todas las operaciones matemáticas utilizadas en el código
- **[Referencia de API](api/reference.md)** - Documentación detallada de la API con parámetros, tipos, y ejemplos

## Uso

### Nuevas Características

#### Funciones de Activación Estilo MATLAB

El proyecto ahora soporta funciones de activación estilo MATLAB para redes de retropropagación:

| Nombre Estándar | Nombre MATLAB | Descripción |
|-----------------|---------------|-------------|
| `'sigmoid'` | `'logsig'` | Sigmoide: 1 / (1 + exp(-z)) |
| `'tanh'` | `'tansig'` | Tangente hiperbólica: tanh(z) |
| `'relu'` | - | Rectified Linear Unit: max(0, z) |
| `'linear'` | `'purelin'` | Lineal: z |
| `'leaky_relu'` | - | Leaky ReLU: max(0.01*z, z) |

**Ejemplo:**
```python
from api import NeuralNetwork, ModelType, NeuralNetworkConfig

# Usar nombres estilo MATLAB
config = NeuralNetworkConfig(
    hidden_layers=[10, 5],
    activation_backprop='tansig',  # o 'tanh'
    output_activation='purelin'     # o 'linear'
)

net = NeuralNetwork(model_type=ModelType.BACKPROP, config=config)
```

#### Funciones de Activación por Capa

Ahora puedes especificar diferentes funciones de activación para cada capa, similar a MATLAB:

```python
config = NeuralNetworkConfig(
    hidden_layers=[10, 5],
    layer_activations=['tansig', 'logsig'],  # Función por capa
    output_activation='purelin'                # Función de salida
)

net = NeuralNetwork(model_type=ModelType.BACKPROP, config=config)
net.train(X, y)

# Ver información de las capas
layer_info = net.get_layer_info()
for layer in layer_info:
    print(f"Capa {layer['layer_index']}: {layer['activation']}")
```

#### Inspección de Pesos y Bias por Capa

Nuevos métodos para inspeccionar pesos y bias de capas específicas:

```python
# Obtener pesos de una capa específica
layer_0 = net.get_layer_weights(0)  # Primera capa oculta
layer_output = net.get_layer_weights(-1)  # Capa de salida

print(f"Pesos: {layer_0['weights'].shape}")
print(f"Bias: {layer_0['bias']}")
print(f"Activación: {layer_0['activation']}")

# Obtener información de todas las capas
layer_info = net.get_layer_info()
```

#### Documentación de API

Para una referencia detallada de todos los métodos y parámetros, consulta:
- `api/reference.md` - Documentación completa de la API con ejemplos

### Demo Rápida (API Compacta)

Ejecutar el script de demo principal:
```bash
py train.py
```

### Ejemplos de API (API Compacta)

Ejecutar ejemplos específicos de la API:
```bash
# RBF básico
py api/examples/basic_rbf.py

# Retropropagación básica
py api/examples/basic_backprop.py

# Uso avanzado con ajuste de parámetros
py api/examples/advanced_usage.py

# Ejemplos de clasificación
py api/examples/classification.py
```

### Tests

Ejecutar tests unitarios:
```bash
# Test de red RBF
py tests/test_rbf_network.py

# Test de red de retropropagación
py tests/test_backprop_network.py

# Test de funciones de activación
py tests/test_activation.py

# Test de métricas de evaluación
py tests/test_metrics.py

# Test del REPL
py tests/test_repl.py
```

### REPL Interactivo

El proyecto incluye un REPL (Read-Eval-Print Loop) interactivo con todo el contexto de la aplicación cargado. Esto es útil para experimentar con la API y prototipar rápidamente.

**Iniciar el REPL:**

Desde el directorio del proyecto:
```bash
py neural.py
```

**Características del REPL:**
- Todo el contexto de la aplicación cargado automáticamente
- Imports de la API principal: `NeuralNetwork`, `ModelType`, `NeuralNetworkConfig`
- Imports de modelos internos: `RBFNetwork`, `BackpropNetwork`, etc.
- Funciones de activación: `GaussianActivation`, `MultiquadraticActivation`, etc.
- Entrenadores y evaluadores: `RBFTrainer`, `BackpropTrainer`, `Evaluator`
- Métricas: `mse`, `mae`, `rmse`, `r2_score`, `accuracy`
- Integración con IPython para una experiencia mejorada (si está instalado)

**Ejemplo de uso en el REPL:**
```python
# Crear datos de prueba
X = np.random.randn(100, 2)
y = np.random.randn(100, 1)

# Crear y entrenar red RBF
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)
net.train(X, y, verbose=True)

# Predecir
predictions = net.predict(X)

# Evaluar
metrics = net.evaluate(X, y)
print(metrics)

# Usar configuración estructurada
config = NeuralNetworkConfig(n_centers=30, sigma=0.5, activation_rbf='gaussian')
net2 = NeuralNetwork(model_type=ModelType.RBF, config=config)
net2.train(X, y)
```

**Para salir del REPL:** usa `exit()` o `Ctrl+D`

## Referencia de la API Compacta

### Clase NeuralNetwork

La clase principal para crear y entrenar redes neuronales con interfaz estilo MATLAB.

#### Constructor

```python
NeuralNetwork(model_type, hidden_layers=None, n_centers=20, sigma=0.5, 
              activation_rbf='gaussian', activation_backprop='sigmoid', 
              learning_rate=0.01, epochs=1000, batch_size=32, use_bias=True, 
              regularization=0.01, random_state=42, initializer='kmeans')
```

**Parámetros del Constructor:**

- **model_type** (ModelType): Tipo de modelo a crear
  - `ModelType.RBF`: Red de Función de Base Radial
  - `ModelType.BACKPROP`: Red de retropropagación

- **hidden_layers** (List[int], opcional): Lista de tamaños de capas ocultas para retropropagación
  - Por defecto: `[10]`
  - Ejemplo: `[20, 10]` crea dos capas ocultas con 20 y 10 neuronas respectivamente

- **n_centers** (int): Número de centros RBF
  - Por defecto: `20`
  - Solo aplicable para redes RBF
  - Afecta la complejidad del modelo

- **sigma** (float): Parámetro de ancho para funciones de activación RBF
  - Por defecto: `0.5`
  - Controla la influencia de cada centro
  - Valores más pequeños = influencia más localizada

- **activation_rbf** (str): Tipo de función de activación RBF
  - Opciones: `'gaussian'`, `'multiquadratic'`, `'inverse_multiquadratic'`, `'thin_plate'` (ln), `'thin_plate_log10'`
  - Por defecto: `'gaussian'`

- **activation_backprop** (str): Tipo de función de activación para retropropagación
  - Opciones: `'sigmoid'`, `'tanh'`, `'relu'`
  - Por defecto: `'sigmoid'`

- **learning_rate** (float): Tasa de aprendizaje para descenso de gradiente
  - Por defecto: `0.01`
  - Solo aplicable para redes de retropropagación
  - Valores típicos: 0.001, 0.01, 0.1

- **epochs** (int): Número máximo de épocas de entrenamiento
  - Por defecto: `1000`
  - Solo aplicable para redes de retropropagación

- **batch_size** (int): Tamaño de lote para entrenamiento mini-batch
  - Por defecto: `32`
  - Usar `-1` para lote completo (batch gradient descent)
  - Solo aplicable para redes de retropropagación

- **use_bias** (bool): Si agregar términos de bias
  - Por defecto: `True`

- **regularization** (float): Parámetro de regularización
  - Por defecto: `0.01`
  - Ayuda a prevenir sobreajuste
  - Para RBF: regularización de pseudoinversa
  - Para backprop: puede extenderse a L2 regularization

- **random_state** (int): Semilla aleatoria para reproducibilidad
  - Por defecto: `42`

- **initializer** (str): Estrategia de inicialización de centros RBF
  - Opciones: `'kmeans'`, `'random'`
  - Por defecto: `'kmeans'`
  - Solo aplicable para redes RBF

#### Métodos

##### train(X, y, verbose=False, log_gradients=False, log_performance=False)

Entrena la red neuronal con los datos proporcionados.

**Parámetros:**
- **X** (np.ndarray): Matriz de entrada de forma `(n_samples, n_features)`
- **y** (np.ndarray): Matriz de salida objetivo de forma `(n_samples, n_outputs)`
- **verbose** (bool): Si imprimir progreso de entrenamiento
  - Por defecto: `False`
- **log_gradients** (bool): Si registrar gradientes durante entrenamiento
  - Por defecto: `False`
- **log_performance** (bool): Si registrar métricas de rendimiento por época
  - Por defecto: `False`

**Retorna:** `TrainingResult` con historial de entrenamiento y métricas

**Ejemplo:**
```python
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)
result = net.train(X_train, y_train, verbose=True)
print(f"Tiempo de entrenamiento: {result.training_time:.2f}s")
print(f"Error final: {result.final_error:.6f}")
```

##### predict(X)

Hace predicciones para nuevos datos de entrada.

**Parámetros:**
- **X** (np.ndarray): Matriz de entrada de forma `(n_samples, n_features)`

**Retorna:** `np.ndarray` de forma `(n_samples, n_outputs)` con predicciones

**Ejemplo:**
```python
predictions = net.predict(X_test)
```

##### evaluate(X, y, detailed=False)

Evalúa el rendimiento del modelo en datos de prueba.

**Parámetros:**
- **X** (np.ndarray): Matriz de entrada de prueba de forma `(n_samples, n_features)`
- **y** (np.ndarray): Matriz de salida objetivo de forma `(n_samples, n_outputs)`
- **detailed** (bool): Si retornar reporte detallado con predicciones
  - Por defecto: `False`

**Retorna:** Diccionario con métricas de evaluación o `EvaluationReport` detallado

**Métricas calculadas:**
- **MSE**: Error Cuadrático Medio
- **MAE**: Error Absoluto Medio
- **RMSE**: Raíz del Error Cuadrático Medio
- **R2**: Coeficiente de determinación R-cuadrado
- **Accuracy**: Precisión (para clasificación)

**Ejemplo:**
```python
metrics = net.evaluate(X_test, y_test)
print(f"MSE: {metrics['mse']:.6f}")
print(f"R2: {metrics['r2']:.6f}")
```

##### get_weights()

Obtiene los pesos del modelo entrenado.

**Retorna:** Diccionario con pesos del modelo

**Para RBF:**
- `'centers'`: Matriz de centros
- `'weights'`: Matriz de pesos de salida
- `'bias'`: Vector de bias

**Para Backpropagation:**
- Lista de pesos de cada capa

**Ejemplo:**
```python
weights = net.get_weights()
print(f"Forma de pesos: {weights['weights'].shape}")
```

##### summary()

Obtiene un resumen del modelo con información de configuración y estado.

**Retorna:** Diccionario con información del modelo

**Información incluida:**
- Tipo de modelo
- Si está entrenado
- Configuración completa
- Número de características y salidas
- Para RBF: número de centros, activación, formas de pesos
- Para Backpropagation: capas ocultas, épocas, tasa de aprendizaje

**Ejemplo:**
```python
info = net.summary()
print(f"Tipo de modelo: {info['model_type']}")
print(f"Entrenado: {info['is_fitted']}")
```

## Estructura del Proyecto

```
PythonProject1/
├── src/                     # Implementación interna del modelo
│   ├── core/                # Interfaces y utilidades principales
│   │   ├── activation.py   # Funciones de activación
│   │   ├── distance.py     # Funciones de distancia euclidiana
│   │   ├── exceptions.py   # Excepciones personalizadas
│   │   ├── interfaces.py   # Clases base abstractas
│   │   └── results.py      # Dataclasses para resultados
│   ├── models/              # Implementaciones de modelos
│   │   ├── rbf/            # Implementación de red RBF
│   │   │   ├── config.py   # Configuración RBF
│   │   │   ├── layer.py    # Capa RBF
│   │   │   ├── network.py  # Red RBF
│   │   │   └── solver.py   # Solucionador pseudoinverso
│   │   └── backprop/       # Implementación de retropropagación
│   │       ├── config.py   # Configuración retropropagación
│   │       ├── layer.py    # Capa densa
│   │       └── network.py  # Red de retropropagación
│   ├── training/            # Utilidades de entrenamiento
│   │   ├── initializer.py  # Estrategias de inicialización de centros
│   │   ├── rbf_trainer.py  # Entrenador RBF
│   │   └── backprop_trainer.py # Entrenador de retropropagación
│   └── evaluation/          # Utilidades de evaluación
│       ├── metrics.py      # Métricas de evaluación
│       └── evaluator.py    # Clase evaluadora
├── api/                     # API compacta estilo MATLAB (RECOMENDADO)
│   ├── model_type.py       # Enum para tipos de modelo
│   ├── neural_network.py   # Clase compacta principal
│   ├── config.py           # Clase de configuración
│   ├── examples/           # Ejemplos de la API
│   │   ├── basic_rbf.py
│   │   ├── basic_backprop.py
│   │   ├── advanced_usage.py
│   │   └── classification.py
│   └── __init__.py
├── repl/                    # REPL interactivo con contexto completo
│   ├── neural_repl.py      # Script del REPL con contexto cargado
│   ├── repl.py             # Script de comando para iniciar REPL
│   └── start_repl.bat      # Comando rápido para Windows
├── tests/                   # Tests unitarios
├── train.py                 # Punto de entrada principal (usa API compacta)
└── requirements.txt         # Dependencias
```

## Implementación de la API Compacta

La API compacta proporciona una interfaz unificada para ambos tipos de redes neuronales:

### NeuralNetwork

La clase `NeuralNetwork` es el punto de entrada principal. Internamente utiliza la arquitectura modular en `src/` pero expone una interfaz simple.

**Funcionalidades:**

1. **Creación de Modelos**: Selecciona el tipo de modelo mediante el enum `ModelType`
2. **Entrenamiento**: Método `train()` con opciones de logging y verbose
3. **Predicción**: Método `predict()` para simulación
4. **Evaluación**: Método `evaluate()` para métricas de rendimiento
5. **Inspección**: Métodos `get_weights()` y `summary()` para inspección del modelo

### Parámetros Configurables

La API permite configurar extensivamente ambos tipos de redes:

**Parámetros RBF:**
- `n_centers`: Número de centros de funciones de base radial
- `sigma`: Parámetro de ancho que controla la influencia de cada centro
- `activation_rbf`: Tipo de función de activación (gaussian, multiquadratic, inverse_multiquadratic, thin_plate)
- `regularization`: Parámetro de regularización para evitar sobreajuste
- `initializer`: Estrategia de inicialización de centros (kmeans o random)

**Parámetros Backpropagation:**
- `hidden_layers`: Lista de tamaños de capas ocultas
- `learning_rate`: Tasa de aprendizaje para descenso de gradiente
- `epochs`: Número de épocas de entrenamiento
- `batch_size`: Tamaño de lote para entrenamiento mini-batch
- `activation_backprop`: Función de activación (sigmoid, tanh, relu)

### Funciones de Activación

El proyecto implementa múltiples funciones de activación para redes RBF:

- **Gaussiana**: exp(-r^2 / (2*sigma^2)) - La más común, suave y diferenciable
- **Multicuadrática**: sqrt(1 + (r/sigma)^2) - Adecuada para interpolación
- **Multicuadrática Inversa**: 1 / sqrt(1 + (r/sigma)^2) - Similar a kernels RBF estándar
- **Thin Plate Spline**: r^2 * ln(r) - Para problemas de suavizado (manejo especial en r=0)

Para backpropagation:
- **Sigmoid**: 1 / (1 + exp(-x)) - Función clásica, salida en (0,1)
- **Tanh**: tanh(x) - Salida en (-1,1), más centrada
- **ReLU**: max(0, x) - Función de unidad lineal rectificada, rápida y popular

### Métricas de Evaluación

El sistema de evaluación proporciona múltiples métricas:

- **MSE (Error Cuadrático Medio)**: Promedio de errores al cuadrado
- **MAE (Error Absoluto Medio)**: Promedio de errores absolutos
- **RMSE (Raíz del Error Cuadrático Medio)**: Raíz cuadrada del MSE
- **R2 (R-cuadrado)**: Coeficiente de determinación, mide la calidad del ajuste
- **Accuracy**: Precisión para problemas de clasificación

### Arquitectura Interna

El proyecto sigue principios de código limpio con:

- Clases base abstractas para extensibilidad
- Patrón Strategy para funciones de activación
- Dataclasses para configuración y resultados
- Clases de entrenador separadas para diferentes modelos
- Sistema de evaluación modular
- API compacta estilo MATLAB para facilidad de uso

### Flujo de Trabajo Típico

1. **Preparación de Datos**: Organizar datos en matrices X (n_samples, n_features) y y (n_samples, n_outputs)
2. **Creación del Modelo**: Instanciar `NeuralNetwork` con parámetros deseados
3. **Entrenamiento**: Llamar al método `train()` con datos de entrenamiento
4. **Predicción**: Usar el método `predict()` con nuevos datos
5. **Evaluación**: Evaluar el rendimiento con `evaluate()` en datos de prueba
6. **Inspección**: Opcionalmente inspeccionar pesos y resumen del modelo
