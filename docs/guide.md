# Guía Técnica de Redes Neuronales RBF y Retropropagación

Esta guía proporciona una explicación detallada del flujo de la aplicación, los modelos matemáticos implementados, y cómo cada componente del código resuelve los problemas matemáticos.

**Nota:** La API de este proyecto está inspirada en el estilo de MATLAB para facilitar su uso a usuarios familiarizados con ese entorno, pero esta guía se enfoca en la implementación Python sin hacer comparaciones directas.

## Tabla de Contenidos

1. [Arquitectura del Proyecto](#arquitectura-del-proyecto)
2. [Modelo Matemático RBF](#modelo-matemático-rbf)
3. [Modelo Matemático de Retropropagación](#modelo-matemático-de-retropropagación)
4. [Flujo de la Aplicación](#flujo-de-la-aplicación)
5. [Implementación en Código](#implementación-en-código)

## Arquitectura del Proyecto

### Estructura Modular

El proyecto sigue una arquitectura modular con separación clara de responsabilidades:

```
├── src/              # Implementación interna (núcleo)
│   ├── core/         # Interfaces y utilidades base
│   ├── models/       # Implementaciones de modelos
│   ├── training/     # Entrenadores y estrategias
│   └── evaluation/  # Métricas y evaluación
├── api/              # API pública (Abstracción del modelo para facil utilización)
├── repl/             # REPL interactivo
└── tests/            # Tests unitarios
```

### Principios de Diseño

1. **Separación de Intereses**: Cada módulo tiene una responsabilidad única
2. **Interfaces Abstractas**: Clases base para extensibilidad
3. **Patrón Strategy**: Funciones de activación intercambiables
4. **Dataclasses**: Configuración y resultados estructurados
5. **API Compacta**: Interfaz simple para uso fácil

## Modelo Matemático RBF

### Concepto Fundamental

Las Redes de Funciones de Base Radial (RBF) son como redes neuronales de tres capas que funcionan de manera diferente a las redes tradicionales. Imagina que tienes tres capas:

- **Capa de entrada**: Simplemente recibe los datos que quieres procesar
- **Capa oculta RBF**: Esta es la parte especial. Cada neurona en esta capa actúa como un "centro" que responde más fuerte a los datos que están cerca de ella. Es como tener varios sensores que cada uno se especializa en una región del espacio
- **Capa de salida**: Combina las respuestas de todas las neuronas ocultas de manera lineal para producir la predicción final

### Fórmulas Matemáticas

**1. Distancia Euclidiana**

Esta fórmula calcula qué tan lejos está un punto de datos de un centro RBF. Es como medir con una regla la distancia en línea recta entre dos puntos en el espacio.

```
d(x, c) = sqrt(sum((x_i - c_i)^2))
```

**Cómo funciona paso a paso:**
1. Calcula la diferencia entre cada característica del punto y el centro
2. Eleva cada diferencia al cuadrado
3. Suma todos los cuadrados
4. Toma la raíz cuadrada de la suma

Donde:
- x: punto de entrada (tus datos)
- c: centro de la función de base radial (una neurona oculta)
- i: índice de cada característica (dimensión)

**2. Función de Activación RBF**

Una vez que calculamos la distancia, necesitamos convertirla en una "activación" - un valor que indica qué tan fuerte responde esa neurona. Las funciones de base radial hacen esto de diferentes maneras:

**Gaussiana:** Es la más común. Convierte la distancia en un valor entre 0 y 1 usando una forma de campana. Distancias pequeñas dan valores cercanos a 1 (fuerte activación), distancias grandes dan valores cercanos a 0 (activación débil).

```
phi(d, sigma) = exp(-(d/sigma)^2)
```

**Multicuadrática:** La activación crece con la distancia. Útil cuando quieres que puntos lejanos tengan más influencia.

```
phi(d, sigma) = sqrt(1 + (d/sigma)^2)
```

**Multicuadrática Inversa:** Lo opuesto a la multicuadrática. La activación disminuye con la distancia, similar a la gaussiana pero con una forma diferente.

```
phi(d, sigma) = 1 / sqrt(1 + (d/sigma)^2)
```

**Thin Plate Spline:** Usada para interpolación suave. Combina el cuadrado de la distancia con un logaritmo.

```
phi(d) = d^2 * ln(d)  (con phi(0) = 0)
```

**¿Qué es sigma?** Es el parámetro de ancho que controla qué tan "ancha" es la función. Un sigma grande hace que la neurona responda a puntos más lejanos, un sigma pequeño hace que solo responda a puntos muy cercanos.

**3. Matriz de Diseño**

La matriz de diseño Phi es como una tabla gigante que contiene todas las activaciones de todas las neuronas para todos los datos de entrenamiento. Cada fila representa un dato de entrenamiento, cada columna representa una neurona oculta.

Para un conjunto de entrenamiento con n muestras y k centros:

```
Phi = [phi(d(x1, c1), sigma)  phi(d(x1, c2), sigma)  ...  phi(d(x1, ck), sigma)]
      [phi(d(x2, c1), sigma)  phi(d(x2, c2), sigma)  ...  phi(d(x2, ck), sigma)]
      [    ...                    ...                    ...      ...          ]
      [phi(d(xn, c1), sigma)  phi(d(xn, c2), sigma)  ...  phi(d(xn, ck), sigma)]
```

Forma: (n_samples, n_centers)

**Cómo interpretar esto:**
- La fila i contiene las activaciones del dato xi con todos los centros
- La columna j contiene las activaciones de todos los datos con el centro cj
- Phi[i,j] = qué tan fuerte responde el centro j al dato i

**4. Solución de Pesos de Salida**

La capa de salida es simplemente una combinación lineal de las activaciones: y = Phi @ W + b. Esto significa que tomamos las activaciones de la capa oculta, las multiplicamos por unos pesos W, y sumamos un bias b para obtener la predicción final.

Los pesos óptimos se encuentran usando pseudoinversa, que es como una "inversa generalizada" que funciona incluso cuando la matriz no es cuadrada:

```
W = pinv(Phi) @ y
```

**Con regularización** (para estabilidad numérica):
```
W = pinv(Phi.T @ Phi + lambda * I) @ Phi.T @ y
```

**Qué hace cada parte:**
- W: matriz de pesos de salida (lo que queremos encontrar)
- y: vector de salidas objetivo (los valores que queremos predecir)
- pinv: pseudoinversa de Moore-Penrose (la "inversa generalizada")
- lambda: parámetro de regularización (un pequeño número que estabiliza el cálculo)
- I: matriz identidad (matriz diagonal con 1s)

Para más detalles sobre cómo funciona la pseudoinversa paso a paso, consulta [Operaciones Matemáticas](mathematical_operations.md).

### Ventajas del Modelo RBF

1. **Solución de Forma Cerrada**: A diferencia de las redes neuronales tradicionales que necesitan iterar muchas veces para aprender, las redes RBF encuentran la solución óptima en un solo paso matemático. Es como resolver una ecuación directamente en lugar de intentar aproximaciones sucesivas
2. **Convergencia Garantizada**: La pseudoinversa siempre encuentra la mejor solución posible (en el sentido de mínimos cuadrados). No hay riesgo de quedar atrapado en mínimos locales como en otros métodos
3. **Entrenamiento Rápido**: Solo requiere un paso de álgebra lineal. Para problemas pequeños y medianos, esto es mucho más rápido que iterar cientos o miles de épocas
4. **Interpretabilidad**: Los centros RBF tienen un significado físico claro - cada centro representa un "prototipo" o "ejemplo típico" de los datos. Puedes ver qué regiones del espacio representa cada neurona

### Por Qué Pseudoinversa en Lugar de Métodos Clásicos

Las redes neuronales tradicionales (como las de retropropagación) usan métodos iterativos como descenso de gradiente para entrenar. Sin embargo, las redes RBF usan **pseudoinversa de Moore-Penrose** por las siguientes razones:

**Diferencias Fundamentales:**

| Aspecto | Métodos Clásicos (Descenso de Gradiente) | Pseudoinversa (RBF) |
|---------|------------------------------------------|---------------------|
| **Enfoque** | Iterativo | Forma cerrada |
| **Convergencia** | No garantizada (puede converger a mínimos locales) | Garantizada (solución óptima global) |
| **Tiempo de Entrenamiento** | O(n × épocas) - lento para muchas épocas | O(n³) - un solo paso |
| **Determinismo** | Puede variar por inicialización aleatoria | Determinista (siempre mismo resultado) |
| **Hiperparámetros** | Requiere tuning de learning rate, épocas, batch size | Solo requiere parámetro de regularización |

**Por Qué Funciona en RBF:**

La capa oculta RBF es fija después de inicializar los centros. Esto significa que la matriz de diseño Φ es constante durante el entrenamiento. El problema de encontrar los pesos de salida W se reduce a resolver un sistema lineal:

```
ΦW = y
```

Dado que Φ es generalmente rectangular (más muestras que centros), no tiene inversa directa. La pseudoinversa Φ⁺ proporciona la **mejor solución en mínimos cuadrados**:

```
W = Φ⁺y = argmin_W ||ΦW - y||²
```

**Implementación con Regularización:**

Para evitar problemas de multicolinealidad y estabilidad numérica, se usa regularización Tikhonov:

```
W = (Φ^T Φ + λI)^(-1) @ Φ^T @ y
```

Donde λI actúa como un "ridge" que estabiliza la inversión matricial. Ver implementación en `src/models/rbf/solver.py`.

**Referencia Matemática Detallada:**

Para las fórmulas matemáticas completas, derivaciones, y explicación de cada operación (pinv, @, inv, eye, etc.), consulta:

📖 **[Operaciones Matemáticas](mathematical_operations.md)** - Documentación completa de todas las operaciones matemáticas utilizadas en el código, incluyendo:
- Fórmula matemática de pseudoinversa y su derivación
- Implementación manual de pinv con regularización
- Explicación de por qué se usa cada operación
- Mapeo código-matemáticas de cada operación

## Modelo Matemático de Retropropagación

### Concepto Fundamental

Las redes de retropropagación (backpropagation) son las redes neuronales más comunes y funcionan de manera iterativa. A diferencia de las redes RBF que encuentran la solución en un solo paso, estas redes aprenden gradualmente ajustando sus pesos muchas veces.

Imagina el proceso así:
1. La red hace una predicción con sus pesos actuales
2. Calcula cuánto se equivocó (el error)
3. Calcula cómo debe cambiar cada peso para reducir ese error
4. Ajusta los pesos un poco en esa dirección
5. Repite este proceso muchas veces hasta que el error sea pequeño

Es como aprender a tirar una canasta: intentas, ves dónde cayó la pelota, ajustas tu técnica, y repites hasta que mejoras.

### Fórmulas Matemáticas

**1. Pase Hacia Adelante (Forward Pass)**

El pase hacia adelante es cuando la red procesa los datos de entrada para producir una predicción. Para cada capa l, hacemos dos pasos:

```
z_l = a_{l-1} @ W_l + b_l
a_l = phi(z_l)
```

**Qué hace cada paso:**
1. **z_l = a_{l-1} @ W_l + b_l**: Tomamos la salida de la capa anterior, la multiplicamos por los pesos de la capa actual, y sumamos el bias. Esto es como una "suma ponderada" de las entradas
2. **a_l = phi(z_l)**: Aplicamos una función de activación no lineal a la suma ponderada. Esto introduce no linealidad y permite que la red aprenda patrones complejos

Donde:
- a_{l-1}: activación de la capa anterior (lo que vino de la capa previa)
- W_l: matriz de pesos de la capa l (qué tan fuerte es cada conexión)
- b_l: vector de bias de la capa l (un ajuste para cada neurona)
- phi: función de activación (sigmoid, tanh, relu, etc.)
- z_l: pre-activación (la suma ponderada antes de la activación)
- a_l: activación final de la capa l (después de aplicar phi)

**Funciones de Activación:**

**Sigmoid:** Convierte cualquier número a un valor entre 0 y 1. Útil cuando quieres probabilidades.

```
phi(z) = 1 / (1 + exp(-z))
phi'(z) = phi(z) * (1 - phi(z))
```

**Tanh:** Convierte cualquier número a un valor entre -1 y 1. Centra los datos alrededor de cero, lo cual ayuda en el entrenamiento.

```
phi(z) = tanh(z)
phi'(z) = 1 - tanh(z)^2
```

**ReLU:** Si el valor es positivo, lo deja igual; si es negativo, lo convierte a cero. Muy popular en redes profundas.

```
phi(z) = max(0, z)
phi'(z) = 1 si z > 0, 0 si z <= 0
```

**2. Función de Pérdida**

La función de pérdida mide qué tan mal está haciendo la red. Usamos error cuadrático medio:

```
L(theta) = (1/2) * sum((y_true - y_pred)^2)
```

**Cómo funciona:**
1. Calcula la diferencia entre el valor real y la predicción
2. Eleva al cuadrado (para que errores positivos y negativos no se cancelen)
3. Suma todos los errores al cuadrado
4. Divide por 2 (el 1/2 es solo conveniencia matemática para la derivada)

**3. Pase Hacia Atrás (Backpropagation)**

El pase hacia atrás es la magia de las redes neuronales. Calculamos cómo cada peso contribuyó al error, usando la regla de la cadena del cálculo.

```
delta_L = (y_pred - y_true) * phi'(z_L)
delta_l = delta_{l+1} @ W_{l+1}^T * phi'(z_l)
```

**Qué está pasando:**
1. **delta_L**: En la capa de salida, calculamos el error directo (predicción menos valor real) y lo multiplicamos por la derivada de la activación
2. **delta_l**: Para las capas anteriores, propagamos el error hacia atrás: tomamos el error de la capa siguiente, lo multiplicamos por los pesos transpuestos, y luego por la derivada de la activación

Es como rastrear el error hacia atrás para ver quién es el culpable.

Donde:
- delta_L: gradiente en la capa de salida (qué tanto error hay en la salida)
- delta_l: gradiente en la capa l (qué tanto error se propaga a esta capa)
- phi': derivada de la función de activación (qué tan rápido cambia la activación)

**4. Actualización de Pesos**

Una vez que tenemos los gradientes, actualizamos los pesos usando descenso de gradiente:

```
W_l(t+1) = W_l(t) - alpha * gradient_W_l
b_l(t+1) = b_l(t) - alpha * gradient_b_l
```

**Intuición:** Imagina que estás en una montaña y quieres bajar. El gradiente te dice en qué dirección es más pronunciada la pendiente hacia abajo. La tasa de aprendizaje (alpha) controla qué tan grande es el paso que das en esa dirección.

Donde:
- alpha: tasa de aprendizaje (qué tan grande es el paso que damos)
- gradient_W_l = a_{l-1}^T @ delta_l (gradiente de los pesos)
- gradient_b_l = sum(delta_l) (gradiente del bias)

**5. Inicialización Xavier/Glorot**

Si inicializamos los pesos con valores muy grandes o muy pequeños, la red puede tener problemas. La inicialización Xavier/Glorot ayuda a que los gradientes fluyan bien a través de las capas:

```
W ~ Uniform(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))
```

**Por qué funciona:** Mantiene la varianza de las activaciones y gradientes aproximadamente constante a través de las capas, previniendo problemas de gradientes desvanecidos o explosivos.

### Ventajas del Modelo de Retropropagación

1. **Universal Approximator**: Teóricamente, una red con suficientes neuronas ocultas puede aproximar cualquier función continua con la precisión que desees. Es como tener una herramienta muy flexible que puede adaptarse a cualquier patrón
2. **Escalabilidad**: Funciona muy bien con grandes conjuntos de datos. A medida que tienes más datos, el modelo puede seguir aprendiendo y mejorando (a diferencia de RBF que puede volverse más lento con muchos datos)
3. **Flexibilidad**: Puedes tener múltiples capas ocultas con diferentes funciones de activación en cada capa. Esto permite que la red aprenda representaciones jerárquicas de los datos
4. **Entrenamiento Iterativo**: Mejora gradualmente con más épocas. Cada iteración refina un poco más la solución, lo cual puede llevar a resultados muy precisos

## Flujo de la Aplicación

### 1. Inicialización del Modelo

Esta es la fase donde preparamos el modelo antes de entrenarlo. Es como preparar los ingredientes antes de cocinar.

**RBF:**
```
1. Determinar número de centros (k) - cuántas neuronas ocultas queremos
2. Inicializar centros - colocarlos aleatoriamente o usar k-means para encontrar posiciones buenas
3. Crear capa RBF con centros y función de activación - configurar la capa oculta
4. Inicializar pesos de salida - se calcularán automáticamente después del entrenamiento
```

**Backpropagation:**
```
1. Determinar arquitectura (capas ocultas) - cuántas capas y cuántas neuronas por capa
2. Inicializar pesos con Xavier/Glorot - valores iniciales que ayudan al entrenamiento
3. Inicializar bias a cero - empiezan sin ajuste
4. Crear capas densas con función de activación - construir la red capa por capa
```

### 2. Proceso de Entrenamiento

Aquí es donde el modelo aprende de los datos.

**RBF (Solución de Forma Cerrada):**
```
1. Calcular matriz de diseño Phi - para cada dato, calcular su distancia a cada centro y aplicar la función de activación
2. Resolver W = pinv(Phi) @ y usando pseudoinversa - encontrar los pesos óptimos en un solo paso matemático
3. Calcular bias si se usa - ajustar la predicción para que esté mejor centrada
4. Modelo listo para predicción - ¡entrenamiento completado!
```

**Backpropagation (Iterativo):**
```
Para cada época hasta convergencia (repetir muchas veces):
    1. Mezclar datos aleatoriamente - para evitar que el modelo memorice el orden
    2. Para cada mini-batch (subconjunto de datos):
        a. Pase hacia adelante - la red hace predicciones con los datos actuales
        b. Calcular error en la salida - qué tan equivocadas fueron las predicciones
        c. Pase hacia atrás - propagar el error hacia atrás para calcular gradientes
        d. Actualizar pesos y bias - ajustar los pesos un poco en la dirección correcta
    3. Registrar error de la época - monitorear cómo mejora el modelo
```

### 3. Predicción

Una vez entrenado, el modelo puede hacer predicciones sobre nuevos datos.

**Ambos modelos:**
```
1. Validar que el modelo está entrenado - asegurarse de que ya aprendió
2. Calcular pase hacia adelante con datos de entrada - procesar los nuevos datos
3. Retornar predicciones - devolver los resultados
```

### 4. Evaluación

Para saber qué tan bueno es el modelo, lo evaluamos con datos de prueba.

```
1. Obtener predicciones en datos de prueba - usar datos que el modelo no vio durante entrenamiento
2. Comparar con valores verdaderos - medir qué tan cerca están las predicciones
3. Calcular métricas:
   - MSE: Error cuadrático medio - promedio de errores al cuadrado
   - MAE: Error absoluto medio - promedio de errores absolutos
   - RMSE: Raíz del error cuadrático medio - en las mismas unidades que los datos
   - R2: Coeficiente de determinación - qué proporción de varianza explica el modelo
   - Accuracy: Precisión - porcentaje de predicciones correctas (para clasificación)
```

## Implementación en Código: Mapeo Detallado Código-Matemáticas

Esta sección conecta el código con las matemáticas. Para cada componente del código, veremos qué problema matemático resuelve, cómo lo hace paso a paso, y cómo se mapea a las fórmulas matemáticas. Es como tener un diccionario que traduce entre el código Python y las fórmulas matemáticas.

### Flujo Completo de la API

La API (`api/neural_network.py`) es como un director de orquesta que coordina todos los componentes internos. Cuando tú llamas a `train()` o `predict()`, la API se encarga de:
1. Validar que los datos estén correctos
2. Crear el modelo apropiado (RBF o Backprop)
3. Inicializar el entrenador correcto
4. Ejecutar el entrenamiento
5. Retornar los resultados

Aquí está el flujo completo:

```
Usuario → NeuralNetwork.train(X, y)
    ↓
1. _validate_input(X, y)
    - Verifica que X sea array 2D
    - Verifica que y sea array 1D o 2D
    - Verifica que n_samples de X == n_samples de y
    ↓
2. _setup_rbf(X, y) o _setup_backprop(X, y)
    - Crea configuración del modelo
    - Inicializa modelo (RBFNetwork o BackpropNetwork)
    - Inicializa entrenador (RBFTrainer o BackpropTrainer)
    ↓
3. trainer.train(model, X, y)
    - Ejecuta algoritmo de entrenamiento específico
    - Retorna TrainingResult con métricas
    ↓
4. Usuario usa predict() y evaluate()
```

### 1. Distancia Euclidiana (`src/core/distance.py`)

**Problema Matemático**: Calcular la distancia entre puntos en un espacio multidimensional. Es como medir con una regla la distancia en línea recta entre dos puntos, pero en lugar de 2D o 3D, puede ser en cualquier número de dimensiones.

**Fórmula Matemática**:
```
d(x, c) = sqrt(sum((x_i - c_i)^2))
```

**Implementación en Código**:

```python
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcular distancia euclidiana entre dos puntos.
    
    Resuelve: d(x, y) = sqrt(sum((x_i - y_i)^2))
    
    Código:
        diff = x - y           # (x_i - y_i)
        squared = diff ** 2    # (x_i - y_i)^2
        summed = np.sum(squared)  # sum((x_i - y_i)^2)
        distance = np.sqrt(summed)  # sqrt(...)
    """
    diff = x - y
    squared = diff ** 2
    summed = np.sum(squared)
    return np.sqrt(summed)

def euclidean_distance_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calcular matriz de distancias euclidianas entre dos conjuntos de puntos.
    
    Resuelve: D[i,j] = d(X[i], Y[j]) para todo i, j
    
    Código:
        Usa broadcasting de numpy para calcular todas las distancias
        simultáneamente sin bucles explícitos.
    """
    # X: (n_samples_X, n_features)
    # Y: (n_samples_Y, n_features)
    # Resultado: (n_samples_X, n_samples_Y)
    
    # Expandir dimensiones para broadcasting
    X_expanded = X[:, np.newaxis, :]  # (n_samples_X, 1, n_features)
    Y_expanded = Y[np.newaxis, :, :]  # (1, n_samples_Y, n_features)
    
    # Calcular diferencias
    diff = X_expanded - Y_expanded  # (n_samples_X, n_samples_Y, n_features)
    
    # Calcular distancias
    squared = diff ** 2
    summed = np.sum(squared, axis=2)  # (n_samples_X, n_samples_Y)
    distances = np.sqrt(summed)
    
    return distances
```

**Mapeo Código-Matemáticas**:
- `x - y` → `(x_i - y_i)` : resta elemento por elemento
- `diff ** 2` → `(x_i - y_i)^2` : cuadrado de cada elemento
- `np.sum(squared)` → `sum((x_i - y_i)^2)` : suma de cuadrados
- `np.sqrt(summed)` → `sqrt(sum(...))` : raíz cuadrada

### 2. Funciones de Activación RBF (`src/core/activation.py`)

**Problema Matemático**: Transformar distancias en activaciones usando funciones de base radial. Una vez que calculamos qué tan lejos está un punto de cada centro, necesitamos convertir esa distancia en un valor de "activación" que indique qué tan fuerte responde esa neurona. Las funciones de base radial hacen esta conversión de diferentes maneras.

**Fórmulas Matemáticas**:

**Gaussiana**: `phi(d, sigma) = exp(-(d/sigma)^2)`

**Implementación en Código**:
```python
class GaussianActivation(ActivationFunction):
    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Calcular activación gaussiana.
        
        Resuelve: phi(d, sigma) = exp(-(d/sigma)^2)
        
        Código:
            normalized = distances / sigma  # d/sigma
            squared = normalized ** 2       # (d/sigma)^2
            negated = -squared              # -(d/sigma)^2
            activation = np.exp(negated)    # exp(-(d/sigma)^2)
        """
        normalized = distances / sigma
        squared = normalized ** 2
        negated = -squared
        return np.exp(negated)
```

**Mapeo Código-Matemáticas**:
- `distances / sigma` → `d/sigma` : normalización por ancho
- `normalized ** 2` → `(d/sigma)^2` : cuadrado
- `-squared` → `-(d/sigma)^2` : negación
- `np.exp(negated)` → `exp(-(d/sigma)^2)` : función exponencial

**Multicuadrática**: `phi(d, sigma) = sqrt(1 + (d/sigma)^2)`

**Implementación en Código**:
```python
class MultiquadraticActivation(ActivationFunction):
    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Calcular activación multicuadrática.
        
        Resuelve: phi(d, sigma) = sqrt(1 + (d/sigma)^2)
        
        Código:
            normalized = distances / sigma
            squared = normalized ** 2
            added = 1 + squared
            activation = np.sqrt(added)
        """
        normalized = distances / sigma
        squared = normalized ** 2
        return np.sqrt(1 + squared)
```

### 3. Capa RBF (`src/models/rbf/layer.py`)

**Problema Matemático**: Calcular la matriz de diseño Phi que contiene todas las activaciones RBF. Esta capa toma todos los datos de entrenamiento, calcula su distancia a cada centro, aplica la función de activación, y devuelve una matriz gigante donde cada celda representa qué tan fuerte responde un centro a un dato específico.

**Fórmula Matemática**:
```
Phi = [phi(d(x1, c1), sigma)  phi(d(x1, c2), sigma)  ...  phi(d(x1, ck), sigma)]
      [phi(d(x2, c1), sigma)  phi(d(x2, c2), sigma)  ...  phi(d(x2, ck), sigma)]
      [    ...                    ...                    ...      ...          ]
      [phi(d(xn, c1), sigma)  phi(d(xn, c2), sigma)  ...  phi(d(xn, ck), sigma)]
```

**Implementación en Código**:
```python
class RBFLayer(BaseLayer):
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calcular pase hacia adelante de la capa RBF.
        
        Resuelve: Construir matriz de diseño Phi
        
        Flujo del código:
            1. Calcular distancias euclidianas entre X y centros
               → d(x_i, c_j) para todo i, j
            2. Aplicar función de activación a las distancias
               → phi(d(x_i, c_j), sigma)
            3. Retornar matriz de diseño Phi
        """
        # Paso 1: Calcular distancias
        distances = euclidean_distance_matrix(X, self.centers)
        # distances[i,j] = d(X[i], centers[j])
        
        # Paso 2: Aplicar función de activación
        activations = self.activation.compute(distances, self.sigma)
        # activations[i,j] = phi(d(X[i], centers[j]), sigma)
        
        # Paso 3: Retornar matriz de diseño
        return activations  # Phi
```

**Mapeo Código-Matemáticas**:
- `euclidean_distance_matrix(X, self.centers)` → Calcula `d(x_i, c_j)` para todo i, j
- `self.activation.compute(distances, self.sigma)` → Calcula `phi(d(x_i, c_j), sigma)` para todo i, j
- Resultado `activations` → Matriz de diseño `Phi` de forma (n_samples, n_centers)

### 4. Solucionador de Pseudoinversa (`src/models/rbf/solver.py`)

**Problema Matemático**: Encontrar pesos óptimos W que minimizan el error cuadrático. Una vez que tenemos la matriz de diseño Phi (todas las activaciones), necesitamos encontrar los pesos W que hagan que Phi @ W sea lo más cercano posible a los valores objetivo y. La pseudoinversa nos da la solución óptima en un solo paso matemático.

**Fórmula Matemática**:
```
W = pinv(Phi) @ y
```
Con regularización:
```
W = pinv(Phi.T @ Phi + lambda * I) @ Phi.T @ y
```

**Implementación en Código**:
```python
def solve_pseudoinverse(Phi: np.ndarray, y: np.ndarray, 
                       regularization: float = 0.0) -> np.ndarray:
    """
    Resolver pesos óptimos usando pseudoinversa.
    
    Resuelve: W = pinv(Phi) @ y
    
    Con regularización: W = pinv(Phi.T @ Phi + lambda * I) @ Phi.T @ y
    
    Flujo del código:
        1. Si hay regularización:
           a. Calcular Phi.T @ Phi
           b. Agregar lambda * I a la diagonal
           c. Calcular inversa del resultado
           d. Multiplicar por Phi.T y luego por y
        2. Si no hay regularización:
           a. Usar np.linalg.pinv(Phi) directamente
           b. Multiplicar por y
    """
    n_samples, n_centers = Phi.shape
    
    if regularization > 0:
        # Con regularización
        # Paso 1a: Phi.T @ Phi
        Phi_T_Phi = Phi.T @ Phi  # (n_centers, n_centers)
        
        # Paso 1b: Agregar lambda * I a la diagonal
        identity = np.eye(n_centers)
        regularized_matrix = Phi_T_Phi + regularization * identity
        # = Phi.T @ Phi + lambda * I
        
        # Paso 1c: Calcular inversa
        inv_regularized = np.linalg.inv(regularized_matrix)
        # = pinv(Phi.T @ Phi + lambda * I)
        
        # Paso 1d: Multiplicar por Phi.T y luego por y
        pseudoinverse = inv_regularized @ Phi.T  # pinv(Phi.T @ Phi + lambda * I) @ Phi.T
        W = pseudoinverse @ y  # pinv(Phi.T @ Phi + lambda * I) @ Phi.T @ y
    else:
        # Sin regularización
        # Paso 2a: Usar pseudoinversa estándar
        pseudoinverse = np.linalg.pinv(Phi)  # pinv(Phi)
        
        # Paso 2b: Multiplicar por y
        W = pseudoinverse @ y  # pinv(Phi) @ y
    
    return W
```

**Mapeo Código-Matemáticas**:
- `Phi.T @ Phi` → `Phi^T @ Phi` : producto matricial
- `Phi_T_Phi + regularization * identity` → `Phi^T @ Phi + lambda * I` : regularización
- `np.linalg.inv(regularized_matrix)` → `(Phi^T @ Phi + lambda * I)^(-1)` : inversión
- `inv_regularized @ Phi.T @ y` → `(Phi^T @ Phi + lambda * I)^(-1) @ Phi^T @ y` : solución completa
- `np.linalg.pinv(Phi) @ y` → `pinv(Phi) @ y` : solución sin regularización

### 5. Red RBF (`src/models/rbf/network.py`)

**Problema Matemático**: Orquestar el entrenamiento y predicción de la red RBF completa. Esta clase es como el cerebro de la red RBF - coordina la inicialización de centros, el cálculo de la matriz de diseño, la solución de pesos mediante pseudoinversa, y las predicciones futuras.

**Fórmula Matemática para Predicción**:
```
y_pred = Phi @ W + b
```

**Implementación en Código**:
```python
class RBFNetwork(BaseModel):
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predecir salidas para nuevas entradas.
        
        Resuelve: y_pred = Phi @ W + b
        
        Flujo del código:
            1. Calcular matriz de diseño Phi para X
            2. Multiplicar Phi por pesos W
            3. Agregar bias b si se usa
            4. Retornar predicciones
        """
        # Paso 1: Calcular matriz de diseño
        Phi = self.layer.forward(X)
        # Phi[i,j] = phi(d(X[i], centers[j]), sigma)
        
        # Paso 2: Multiplicar por pesos
        y_pred = Phi @ self.weights
        # y_pred = Phi @ W
        
        # Paso 3: Agregar bias si se usa
        if self.use_bias:
            y_pred = y_pred + self.bias
            # y_pred = Phi @ W + b
        
        # Paso 4: Retornar predicciones
        return y_pred
```

**Mapeo Código-Matemáticas**:
- `self.layer.forward(X)` → `Phi` : matriz de diseño
- `Phi @ self.weights` → `Phi @ W` : producto matricial
- `y_pred + self.bias` → `Phi @ W + b` : suma de bias

### 6. Entrenador RBF (`src/training/rbf_trainer.py`)

**Problema Matemático**: Coordinar inicialización de centros y cálculo de pesos. Este entrenador es como el gerente de entrenamiento - se encarga de inicializar los centros (usando aleatorización o k-means), asignarlos al modelo, calcular la matriz de diseño, resolver los pesos mediante pseudoinversa, y calcular el bias si es necesario.

**Fórmula Matemática**:
```
1. Inicializar centros C (aleatorio o k-means)
2. Calcular Phi = phi(d(X, C), sigma)
3. Calcular W = pinv(Phi) @ y
4. Calcular bias b si se usa
```

**Implementación en Código**:
```python
class RBFTrainer(BaseTrainer):
    def train(self, model: RBFNetwork, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """
        Entrenar red RBF usando solución de forma cerrada.
        
        Resuelve:
            1. C = initializer.initialize(X, n_centers)
            2. Phi = phi(d(X, C), sigma)
            3. W = pinv(Phi) @ y
            4. b = mean(y) si se usa bias
        
        Flujo del código:
            1. Inicializar centros usando estrategia seleccionada
            2. Asignar centros al modelo
            3. Calcular matriz de diseño Phi
            4. Resolver pesos usando pseudoinversa
            5. Calcular bias si se usa
            6. Calcular error final
        """
        # Paso 1: Inicializar centros
        centers = self.initializer.initialize(X, model.config.n_centers)
        # C = initializer.initialize(X, n_centers)
        
        # Paso 2: Asignar centros al modelo
        model.layer.centers = centers
        # model.centers = C
        
        # Paso 3: Calcular matriz de diseño
        Phi = model.layer.forward(X)
        # Phi = phi(d(X, C), sigma)
        
        # Paso 4: Resolver pesos
        model.weights = solve_pseudoinverse(
            Phi, y, 
            regularization=model.config.regularization
        )
        # W = pinv(Phi) @ y
        
        # Paso 5: Calcular bias si se usa
        if model.use_bias:
            # Bias = media de las diferencias
            predictions_no_bias = Phi @ model.weights
            model.bias = np.mean(y - predictions_no_bias)
            # b = mean(y - Phi @ W)
        
        # Paso 6: Calcular error final
        predictions = model.predict(X)
        mse = np.mean((y - predictions) ** 2)
        # MSE = (1/n) * sum((y - y_pred)^2)
        
        return TrainingResult(
            final_error=mse,
            epochs=1,
            error_history=[mse],
            converged=True,
            metadata={}
        )
```

**Mapeo Código-Matemáticas**:
- `self.initializer.initialize(X, model.config.n_centers)` → `C = initializer(X, k)` : inicialización de centros
- `model.layer.forward(X)` → `Phi = phi(d(X, C), sigma)` : matriz de diseño
- `solve_pseudoinverse(Phi, y, ...)` → `W = pinv(Phi) @ y` : solución de pesos
- `np.mean(y - predictions_no_bias)` → `b = mean(y - Phi @ W)` : cálculo de bias
- `np.mean((y - predictions) ** 2)` → `MSE = (1/n) * sum((y - y_pred)^2)` : error final

### 7. Capa Densa de Retropropagación (`src/models/backprop/layer.py`)

**Problema Matemático**: Implementar neurona densa con pase hacia adelante y hacia atrás. Esta capa es el bloque fundamental de las redes de retropropagación - cada neurona toma las entradas, las multiplica por pesos, suma un bias, aplica una función de activación no lineal, y también puede propagar gradientes hacia atrás durante el entrenamiento.

**Fórmulas Matemáticas**:

**Pase hacia adelante**:
```
z = a_in @ W + b
a_out = phi(z)
```

**Pase hacia atrás**:
```
delta_out = gradient_in * phi'(z)
gradient_W = a_in^T @ delta_out
gradient_b = sum(delta_out)
gradient_in = delta_out @ W^T
```

**Implementación en Código**:
```python
class DenseLayer(BaseLayer):
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Pase hacia adelante.
        
        Resuelve: z = a_in @ W + b
                   a_out = phi(z)
        
        Flujo del código:
            1. Guardar entrada para backpropagation
            2. Calcular z = X @ W + b
            3. Aplicar función de activación
            4. Retornar activación
        """
        # Paso 1: Guardar entrada
        self.last_input = X
        # a_in = X
        
        # Paso 2: Calcular pre-activación
        z = X @ self.weights + self.bias
        # z = a_in @ W + b
        
        # Paso 3: Guardar z para backpropagation
        self.last_z = z
        
        # Paso 4: Aplicar función de activación
        output = self._apply_activation(z)
        # a_out = phi(z)
        
        # Paso 5: Retornar activación
        return output
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Pase hacia atrás.
        
        Resuelve: delta_out = gradient_in * phi'(z)
                   gradient_W = a_in^T @ delta_out
                   gradient_b = sum(delta_out)
                   gradient_in = delta_out @ W^T
        
        Flujo del código:
            1. Calcular derivada de activación
            2. Calcular delta en esta capa
            3. Calcular gradientes de W y b
            4. Calcular gradiente de entrada
            5. Retornar gradiente de entrada
        """
        # Paso 1: Calcular derivada de activación
        activation_grad = self._activation_derivative(self.last_z)
        # phi'(z)
        
        # Paso 2: Calcular delta en esta capa
        delta = output_gradient * activation_grad
        # delta_out = gradient_in * phi'(z)
        
        # Paso 3: Calcular gradientes
        self.weights_gradient = self.last_input.T @ delta
        # gradient_W = a_in^T @ delta_out
        
        self.bias_gradient = np.sum(delta, axis=0)
        # gradient_b = sum(delta_out)
        
        # Paso 4: Calcular gradiente de entrada
        input_gradient = delta @ self.weights.T
        # gradient_in = delta_out @ W^T
        
        # Paso 5: Retornar gradiente de entrada
        return input_gradient
```

**Mapeo Código-Matemáticas**:
- `X @ self.weights + self.bias` → `a_in @ W + b` : pre-activación
- `self._apply_activation(z)` → `phi(z)` : activación
- `self._activation_derivative(self.last_z)` → `phi'(z)` : derivada de activación
- `output_gradient * activation_grad` → `gradient_in * phi'(z)` : delta
- `self.last_input.T @ delta` → `a_in^T @ delta_out` : gradiente de pesos
- `np.sum(delta, axis=0)` → `sum(delta_out)` : gradiente de bias
- `delta @ self.weights.T` → `delta_out @ W^T` : gradiente de entrada

### 8. Red de Retropropagación (`src/models/backprop/network.py`)

**Problema Matemático**: Coordinar múltiples capas densas para pase hacia adelante y hacia atrás. Esta red es como una cadena de procesamiento - conecta múltiples capas densas en secuencia, donde cada capa toma la salida de la anterior como entrada. Durante el entrenamiento, coordina el flujo de datos hacia adelante y la propagación de gradientes hacia atrás a través de todas las capas.

**Fórmulas Matemáticas**:

**Pase hacia adelante** (para cada capa):
```
a_0 = X
a_l = phi(a_{l-1} @ W_l + b_l) para l = 1, 2, ..., L
y_pred = a_L
```

**Pase hacia atrás**:
```
delta_L = (y_pred - y) * phi'(z_L)
delta_l = delta_{l+1} @ W_{l+1}^T * phi'(z_l) para l = L-1, ..., 1
```

**Implementación en Código**:
```python
class BackpropNetwork(BaseModel):
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Pase hacia adelante a través de todas las capas.
        
        Resuelve: a_0 = X
                   a_l = phi(a_{l-1} @ W_l + b_l) para l = 1, 2, ..., L
                   y_pred = a_L
        
        Flujo del código:
            1. Inicializar activación con X
            2. Para cada capa:
               a. Calcular pase hacia adelante
               b. Actualizar activación
            3. Retornar activación final
        """
        # Paso 1: Inicializar activación
        activation = X
        # a_0 = X
        
        # Paso 2: Pase hacia adelante a través de todas las capas
        for layer in self.layers:
            activation = layer.forward(activation)
            # a_l = phi(a_{l-1} @ W_l + b_l)
        
        # Paso 3: Retornar activación final
        return activation
        # y_pred = a_L
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Pase hacia atrás para calcular gradientes.
        
        Resuelve: delta_L = (y_pred - y) * phi'(z_L)
                   delta_l = delta_{l+1} @ W_{l+1}^T * phi'(z_l)
        
        Flujo del código:
            1. Calcular pase hacia adelante
            2. Calcular error en la salida
            3. Calcular delta en la capa de salida
            4. Propagar delta hacia atrás a través de las capas
        """
        # Paso 1: Calcular pase hacia adelante
        y_pred = self.forward(X)
        
        # Paso 2: Calcular error en la salida
        error = y_pred - y
        # error = y_pred - y
        
        # Paso 3: Calcular delta en la capa de salida
        gradient = error
        # gradient = y_pred - y
        
        # Paso 4: Propagar hacia atrás a través de las capas
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
            # delta_l = delta_{l+1} @ W_{l+1}^T * phi'(z_l)
```

**Mapeo Código-Matemáticas**:
- `activation = X` → `a_0 = X` : activación inicial
- `layer.forward(activation)` → `a_l = phi(a_{l-1} @ W_l + b_l)` : activación de cada capa
- `y_pred - y` → `y_pred - y` : error de predicción
- `layer.backward(gradient)` → `delta_l = delta_{l+1} @ W_{l+1}^T * phi'(z_l)` : propagación de delta

### 9. Entrenador de Retropropagación (`src/training/backprop_trainer.py`)

**Problema Matemático**: Coordinar entrenamiento iterativo con descenso de gradiente. Este entrenador es como el coach del equipo - coordina el entrenamiento repetitivo de la red, mezclando los datos, procesándolos en mini-batches, calculando gradientes, actualizando pesos gradualmente, y monitoreando el error a través de las épocas.

**Fórmulas Matemáticas**:

**Actualización de pesos**:
```
W_l(t+1) = W_l(t) - alpha * gradient_W_l
b_l(t+1) = b_l(t) - alpha * gradient_b_l
```

**Implementación en Código**:
```python
class BackpropTrainer(BaseTrainer):
    def train(self, model: BackpropNetwork, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """
        Entrenar red usando descenso de gradiente.
        
        Resuelve: W_l(t+1) = W_l(t) - alpha * gradient_W_l
                   b_l(t+1) = b_l(t) - alpha * gradient_b_l
        
        Flujo del código:
            Para cada época hasta convergencia:
                1. Mezclar datos aleatoriamente
                2. Para cada mini-batch:
                   a. Calcular pase hacia adelante
                   b. Calcular pase hacia atrás
                   c. Actualizar pesos y bias
                3. Registrar error de la época
        """
        error_history = []
        
        # Para cada época
        for epoch in range(self.config.epochs):
            # Paso 1: Mezclar datos
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Paso 2: Procesar mini-batches
            for i in range(0, len(X), self.config.batch_size):
                X_batch = X_shuffled[i:i+self.config.batch_size]
                y_batch = y_shuffled[i:i+self.config.batch_size]
                
                # Paso 2a: Calcular gradientes
                model.backward(X_batch, y_batch)
                
                # Paso 2b: Actualizar pesos y bias
                for layer in model.layers:
                    layer.weights -= self.config.learning_rate * layer.weights_gradient
                    # W_l(t+1) = W_l(t) - alpha * gradient_W_l
                    
                    layer.bias -= self.config.learning_rate * layer.bias_gradient
                    # b_l(t+1) = b_l(t) - alpha * gradient_b_l
            
            # Paso 3: Calcular error de la época
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            error_history.append(mse)
        
        return TrainingResult(
            final_error=error_history[-1],
            epochs=self.config.epochs,
            error_history=error_history,
            converged=True,
            metadata={}
        )
```

**Mapeo Código-Matemáticas**:
- `layer.weights -= self.config.learning_rate * layer.weights_gradient` → `W_l(t+1) = W_l(t) - alpha * gradient_W_l` : actualización de pesos
- `layer.bias -= self.config.learning_rate * layer.bias_gradient` → `b_l(t+1) = b_l(t) - alpha * gradient_b_l` : actualización de bias
- `np.mean((y - y_pred) ** 2)` → `MSE = (1/n) * sum((y - y_pred)^2)` : error por época

### 10. API Compacta (`api/neural_network.py`)

**Problema Matemático**: Proveer interfaz unificada que abstrae la complejidad de ambos modelos. Esta API es como un panel de control simple - oculta toda la complejidad interna de las redes neuronales y te da una interfaz fácil de usar donde solo necesitas llamar a `train()`, `predict()`, y `evaluate()` sin preocuparte por los detalles matemáticos o de implementación.

**Flujo Completo de la API**:

```python
class NeuralNetwork:
    def train(self, X, y, verbose=False):
        """
        Entrenar el modelo.
        
        Flujo del código:
            1. Validar entrada
               - Verifica formas de X e y
               - Convierte a arrays numpy
            2. Configurar modelo según tipo
               - Si RBF: crear RBFNetwork y RBFTrainer
               - Si Backprop: crear BackpropNetwork y BackpropTrainer
            3. Delegar entrenamiento al entrenador específico
               - RBF: solución de forma cerrada
               - Backprop: descenso de gradiente iterativo
            4. Retornar resultados
        """
        # Paso 1: Validar entrada
        self._validate_input(X, y)
        # Verifica que X sea 2D, y sea 1D o 2D, n_samples coincidan
        
        # Paso 2: Configurar modelo
        if self.model_type == ModelType.RBF:
            self._setup_rbf(X, y)
            # Crea RBFNetwork con centros y función de activación
            # Crea RBFTrainer con inicializador
        else:
            self._setup_backprop(X, y)
            # Crea BackpropNetwork con capas y pesos Xavier
            # Crea BackpropTrainer con configuración
        
        # Paso 3: Delegar entrenamiento
        result = self.trainer.train(self.model, X, y)
        # RBF: W = pinv(Phi) @ y
        # Backprop: W(t+1) = W(t) - alpha * gradient
        
        # Paso 4: Retornar resultados
        return result
    
    def predict(self, X):
        """
        Predecir salidas.
        
        Flujo del código:
            1. Verificar que el modelo está entrenado
            2. Validar entrada
            3. Delegar predicción al modelo
            4. Retornar predicciones
        """
        # Paso 1: Verificar entrenamiento
        self._ensure_fitted()
        
        # Paso 2: Validar entrada
        self._validate_input(X)
        
        # Paso 3: Delegar predicción
        predictions = self.model.predict(X)
        # RBF: y_pred = Phi @ W + b
        # Backprop: y_pred = a_L (última activación)
        
        # Paso 4: Retornar predicciones
        return predictions
```

**Mapeo API-Matemáticas**:
- `self._validate_input(X, y)` → Validación de datos antes del procesamiento matemático
- `self._setup_rbf(X, y)` → Configuración de parámetros RBF (centros, sigma)
- `self.trainer.train(self.model, X, y)` → Ejecución del algoritmo matemático de entrenamiento
- `self.model.predict(X)` → Evaluación de la función matemática del modelo entrenado

## Nuevas Características

### Funciones de Activación

El proyecto soporta múltiples funciones de activación para redes de retropropagación, permitiendo flexibilidad en la configuración del modelo.

**Funciones Disponibles:**

| Nombre | Fórmula |
|--------|---------|
| `'sigmoid'` | `phi(z) = 1 / (1 + exp(-z))` |
| `'tanh'` | `phi(z) = tanh(z)` |
| `'relu'` | `phi(z) = max(0, z)` |
| `'linear'` | `phi(z) = z` |
| `'leaky_relu'` | `phi(z) = max(0.01*z, z)` |

**Implementación en Código:**

Las funciones de activación tienen derivadas para backpropagation:

```python
class SigmoidActivation(ActivationFunction):
    def compute(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        s = self.compute(x)
        return s * (1.0 - s)
```

**Mapeo Código-Matemáticas:**
- `1.0 / (1.0 + np.exp(-x))` → `phi(z) = 1 / (1 + exp(-z))`
- `s * (1.0 - s)` → `phi'(z) = phi(z) * (1 - phi(z))`

### Funciones de Activación por Capa

Ahora puedes especificar diferentes funciones de activación para cada capa de la red de retropropagación.

**Configuración:**

```python
config = NeuralNetworkConfig(
    hidden_layers=[10, 5],
    layer_activations=['tansig', 'logsig'],  # Función por capa
    output_activation='purelin'                # Función de salida
)
```

**Implementación en Código:**

En `BackpropNetwork._build_network()`:

```python
for i in range(len(layer_sizes) - 1):
    # Determinar función de activación para esta capa
    if i < len(layer_sizes) - 2:
        if self.config.layer_activations is not None:
            activation = self.config.layer_activations[i]
        else:
            activation = self.config.activation
    else:
        activation = self.config.output_activation
    
    layer = DenseLayer(
        input_size=layer_sizes[i],
        output_size=layer_sizes[i + 1],
        activation=activation,
        use_bias=self.config.use_bias
    )
```

**Correspondencia Matemática:**
- Cada capa `l` tiene su propia función de activación `phi_l`
- Pase hacia adelante: `a_l = phi_l(z_l)` donde `z_l = a_{l-1} @ W_l + b_l`
- Pase hacia atrás: `delta_l = delta_{l+1} @ W_{l+1}^T @ phi'_l(z_l)`

### Inspección de Pesos y Bias por Capa

Nuevos métodos para inspeccionar pesos y bias de capas específicas, permitiendo análisis detallado del modelo entrenado.

**Métodos API:**

1. `get_layer_weights(layer_index: int)` - Obtiene pesos y bias de una capa específica
2. `get_layer_info()` - Obtiene información de todas las capas

**Implementación en Código:**

```python
def get_layer_weights(self, layer_index: int) -> Dict[str, Any]:
    # Manejar índices negativos
    if layer_index < 0:
        layer_index = len(self.model.layers) + layer_index
    
    layer = self.model.layers[layer_index]
    
    return {
        'layer_index': layer_index,
        'layer_type': 'hidden' if layer_index < len(self.model.layers) - 1 else 'output',
        'input_size': layer.input_size,
        'output_size': layer.output_size,
        'weights': layer.weights.copy(),
        'bias': layer.bias.copy() if layer.bias is not None else None,
        'activation': str(layer.activation),
        'use_bias': layer.use_bias
    }
```

**Uso:**

```python
# Obtener pesos de la primera capa oculta
layer_0 = net.get_layer_weights(0)

# Obtener pesos de la capa de salida
layer_output = net.get_layer_weights(-1)

# Obtener información de todas las capas
layer_info = net.get_layer_info()
for layer in layer_info:
    print(f"Capa {layer['layer_index']}: {layer['activation']}")
    print(f"  Pesos: {layer['weights_shape']}")
```

### Documentación de API

Se ha creado un archivo de referencia completa de la API en `api/reference.md` que incluye:
- Documentación detallada de cada método
- Tipos de datos de parámetros y retornos
- Ejemplos de uso
- Tablas de referencia de funciones de activación
- Notas importantes sobre uso

## Comparación de Modelos

| Aspecto | RBF | Retropropagación |
|----------|-----|------------------|
| **Entrenamiento** | Forma cerrada (rápido) | Iterativo (lento) |
| **Convergencia** | Garantizada | No garantizada |
| **Interpretabilidad** | Alta (centros significativos) | Baja (pesos opacos) |
| **Escalabilidad** | Limitada (costo O(n^3)) | Alta (mini-batch) |
| **Flexibilidad** | Baja (arquitectura fija) | Alta (arquitectura variable) |
| **Uso típico** | Pequeños datasets, interpolación | Grandes datasets, clasificación |

## Ejemplo Práctico

```python
# Crear datos
X = np.random.randn(100, 2)
y = np.random.randn(100, 1)

# RBF
net_rbf = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)
net_rbf.train(X, y)
predictions_rbf = net_rbf.predict(X)

# Retropropagación
net_bp = NeuralNetwork(model_type=ModelType.BACKPROP, hidden_layers=[10, 5])
net_bp.train(X, y)
predictions_bp = net_bp.predict(X)

# Comparar
metrics_rbf = net_rbf.evaluate(X, y)
metrics_bp = net_bp.evaluate(X, y)
```

## Conclusión

Este proyecto implementa dos arquitecturas de redes neuronales fundamentales con código limpio, modular y bien documentado. La arquitectura modular permite fácil extensión y mantenimiento, mientras que la API compacta facilita el uso práctico.

Los modelos matemáticos implementados son estándar en el campo y se han traducido a código de manera clara y eficiente, con validación robusta y manejo de errores apropiado.
