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

La API (`api/neural_network.py`) es como un director de orquesta que coordina todos los componentes internos a través de un **sistema de registro dinámico**. Cuando tú llamas a `train()` o `predict()`, la API se encarga de:
1. Validar que los datos estén correctos
2. Consultar el registro de modelos para obtener la factory apropiada
3. Usar la factory para crear el modelo y entrenador configurados
4. Ejecutar el entrenamiento
5. Retornar resultados tipados con dataclasses

Aquí está el flujo completo actualizado:

```
Usuario → NeuralNetwork.train(X, y)
    ↓
1. _validate_input(X, y)
    - Verifica que X sea array 2D
    - Verifica que y sea array 1D o 2D
    - Verifica que n_samples de X == n_samples de y
    ↓
2. _setup_model(X, y) - Usa el sistema de registro dinámico
    a. model_type_str = self.model_type.value
       - Convierte el enum ModelType a string ('rbf' o 'backprop')
    b. factory = ModelRegistry.get_factory(model_type_str)
       - Consulta el registro centralizado para obtener la factory
       - El registro es un singleton que mapea strings a factories
    c. self.model = factory.create_network(X, y, self.config)
       - La factory crea la instancia del modelo configurado
       - Para RBF: RBFNetwork con centros, sigma, activación
       - Para Backprop: BackpropNetwork con capas, learning rate, épocas
    d. self.trainer = factory.create_trainer(self.config)
       - La factory crea el entrenador apropiado
       - Para RBF: RBFTrainer con inicializador de centros
       - Para Backprop: BackpropTrainer con verbose
    ↓
3. trainer.train(model, X, y)
    - Ejecuta algoritmo de entrenamiento específico
    - Retorna TrainingResult (dataclass tipado) con métricas
    ↓
4. Usuario usa predict() y evaluate()
    - predict() retorna np.ndarray con predicciones
    - evaluate() retorna EvaluationResult (dataclass tipado)
```

**¿Por qué el sistema de registro dinámico?**

El sistema de registro permite agregar nuevos modelos sin modificar el código central de la API. Antes, agregar un modelo requería modificar múltiples archivos. Ahora, solo necesitas:
1. Crear la implementación del modelo
2. Crear una factory que implemente ModelFactory
3. Registrar la factory con ModelRegistry.register()

Esto hace el código más modular, extensible y mantenible.

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

**Arquitectura de la Red RBF**:
```
Entrada X (n_samples, n_features)
    ↓
Capa RBF Oculta (n_centers neuronas)
    - Calcula distancias euclidianas a centros
    - Aplica función de activación (ej. gaussiana)
    - Salida: Phi (n_samples, n_centers)
    ↓
Capa de Salida Lineal
    - Multiplicación matricial: Phi @ W
    - Agrega bias si está configurado
    - Salida: y_pred (n_samples, n_outputs)
```

**Fórmula Matemática para Predicción**:
```
y_pred = Phi @ W + b
```
Donde:
- `Phi[i,j] = phi(d(X[i], C[j]), sigma)` es la matriz de diseño
- `W` son los pesos de salida (n_centers, n_outputs)
- `b` es el bias (n_outputs,)

**Flujo Completo de Entrenamiento** (`RBFNetwork.fit()` en `src/models/rbf/network.py`):

```
1. Validación de entrada
   - Verificar que X es 2D: (n_samples, n_features)
   - Verificar que y es 1D o 2D, convertir a 2D si es necesario
   - Verificar que n_samples de X == n_samples de y
   ↓
2. Almacenar dimensiones
   - self.n_features_ = X.shape[1]
   - self.n_outputs_ = y.shape[1]
   ↓
3. Inicialización de centros
   - Si no se proporcionan centros:
     a. n_centers = min(config.n_centers, n_samples)
     b. Seleccionar n_centers muestras aleatorias de X
     c. self.centers = X[indices]
   - Si se proporcionan centros:
     a. self.centers = centers
     b. config.n_centers = centers.shape[0]
   ↓
4. Crear capa RBF
   - self.rbflayer = RBFLayer(
       centers=self.centers,
       activation=self.config.activation,
       sigma=self.config.sigma
     )
   ↓
5. Calcular matriz de diseño Phi
   - Phi = self.rbflayer.forward(X)
   - Phi[i,j] = phi(d(X[i], centers[j]), sigma)
   ↓
6. Agregar columna de bias si está configurado
   - Si use_bias=True:
     Phi_bias = np.column_stack([Phi, np.ones(Phi.shape[0])])
   - Si use_bias=False:
     Phi_bias = Phi
   ↓
7. Resolver para pesos usando pseudoinversa
   - self.weights = solve_pseudoinverse(
       Phi_bias,
       y,
       regularization=self.config.regularization
     )
   - Con regularización: W = (Phi.T @ Phi + λI)^(-1) @ Phi.T @ y
   - Sin regularización: W = pinv(Phi) @ y
   ↓
8. Extraer bias si se usa
   - Si use_bias=True:
     self.bias = self.weights[-1, :]
     self.weights = self.weights[:-1, :]
   - Si use_bias=False:
     self.bias = np.zeros(self.n_outputs_)
   ↓
9. Marcar modelo como entrenado
   - self.is_fitted = True
```

**Implementación en Código** (`src/models/rbf/network.py`):
```python
def fit(self, X: np.ndarray, y: np.ndarray, centers: np.ndarray = None) -> None:
    """
    Entrenar la red RBF con datos de entrada X y salidas objetivo y.
    
    El entrenamiento resuelve: W = pinv(Phi) @ y
    donde Phi es la matriz de diseño calculada como Phi = phi(d(X, C), sigma)
    """
    # Paso 1: Validar formas de entrada
    X = np.asarray(X)
    y = np.asarray(y)
    
    if X.ndim != 2:
        raise InvalidInputError(f"X debe ser array 2D, se obtuvo forma {X.shape}")
    
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    if y.ndim != 2:
        raise InvalidInputError(f"y debe ser array 1D o 2D, se obtuvo forma {y.shape}")
    
    if X.shape[0] != y.shape[0]:
        raise InvalidInputError(
            f"Discrepancia en número de muestras: X tiene {X.shape[0]}, y tiene {y.shape[0]}"
        )
    
    # Paso 2: Almacenar dimensiones
    self.n_features_ = X.shape[1]
    self.n_outputs_ = y.shape[1]
    
    # Paso 3: Usar centros proporcionados o centros existentes
    if centers is not None:
        self.centers = centers
        self.config.n_centers = centers.shape[0]
    elif self.centers is None:
        # Paso 3a: Si no se proporcionan centros, muestrear aleatoriamente
        n_samples = X.shape[0]
        n_centers = min(self.config.n_centers, n_samples)
        indices = np.random.choice(n_samples, n_centers, replace=False)
        self.centers = X[indices]
    
    # Paso 4: Crear la capa RBF
    self.rbflayer = RBFLayer(
        centers=self.centers,
        activation=self.config.activation,
        sigma=self.config.sigma
    )
    
    # Paso 5: Calcular matriz de diseño (activaciones de capa oculta)
    Phi = self.rbflayer.forward(X)
    
    # Paso 6: Agregar columna de bias si está configurado
    if self.config.use_bias:
        Phi_bias = np.column_stack([Phi, np.ones(Phi.shape[0])])
    else:
        Phi_bias = Phi
    
    # Paso 7: Resolver para pesos de salida usando pseudoinversa
    self.weights = solve_pseudoinverse(
        Phi_bias,
        y,
        regularization=self.config.regularization
    )
    
    # Paso 8: Extraer bias si se usa
    if self.config.use_bias:
        self.bias = self.weights[-1, :]
        self.weights = self.weights[:-1, :]
    else:
        self.bias = np.zeros(self.n_outputs_)
    
    # Paso 9: Marcar como entrenado
    self.is_fitted = True
```

**Flujo de Predicción** (`RBFNetwork.predict()` en `src/models/rbf/network.py`):

```
1. Verificar que el modelo está entrenado
   - if not self.is_fitted: raise NotFittedError
   ↓
2. Validar forma de entrada
   - Verificar que X es 2D
   - Verificar que X.shape[1] == self.n_features_
   ↓
3. Calcular activaciones de capa oculta
   - hidden_output = self.rbflayer.forward(X)
   - hidden_output[i,j] = phi(d(X[i], centers[j]), sigma)
   ↓
4. Calcular salida final
   - predictions = hidden_output @ self.weights + self.bias
   - y_pred = Phi @ W + b
   ↓
5. Retornar predicciones
```

**Implementación en Código**:
```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Hacer predicciones para datos de entrada X.
    
    La predicción se calcula como: y_pred = Phi @ W + b
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
    
    # Paso 3: Calcular activaciones de capa oculta
    hidden_output = self.rbflayer.forward(X)
    
    # Paso 4: Calcular salida: hidden_output * weights + bias
    predictions = hidden_output @ self.weights + self.bias
    
    return predictions
```

**Mapeo Código-Matemáticas**:
- `self.rbflayer.forward(X)` → `Phi` : matriz de diseño (n_samples, n_centers)
- `hidden_output @ self.weights` → `Phi @ W` : producto matricial
- `predictions + self.bias` → `Phi @ W + b` : suma de bias
- `solve_pseudoinverse(Phi_bias, y, regularization)` → `W = pinv(Phi) @ y` : solución óptima

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

### 7. Sistema de Registro Dinámico (`api/core/registry.py`)

**Problema de Diseño**: Permitir agregar nuevos modelos al sistema sin modificar el código central de la API. Sin este sistema, cada nuevo modelo requería modificar múltiples archivos (neural_network.py, factories.py, config.py, etc.), lo cual crea acoplamiento y hace el código difícil de mantener.

**Solución**: Un registro centralizado (singleton) que mapea identificadores de strings a factories de modelos. Las factories implementan una interfaz común (`ModelFactory`) que sabe cómo crear modelos y entrenadores configurados.

**Implementación en Código**:
```python
class ModelRegistry:
    """
    Registro centralizado de modelos.
    
    Es un singleton que mantiene un diccionario de factories.
    Permite registrar y recuperar modelos dinámicamente.
    """
    _factories: Dict[str, ModelFactory] = {}
    
    @classmethod
    def register(cls, model_type: str, factory: ModelFactory) -> None:
        """
        Registrar un nuevo modelo.
        
        Flujo:
            1. Agrega la factory al diccionario interno
            2. La factory queda disponible para consultas futuras
        """
        cls._factories[model_type] = factory
    
    @classmethod
    def get_factory(cls, model_type: str) -> ModelFactory:
        """
        Obtener la factory de un modelo.
        
        Flujo:
            1. Busca el model_type en el diccionario
            2. Si existe, retorna la factory
            3. Si no existe, lanza ValueError con modelos disponibles
        """
        if model_type not in cls._factories:
            raise ValueError(
                f"Modelo '{model_type}' no registrado. "
                f"Modelos disponibles: {list(cls._factories.keys())}"
            )
        return cls._factories[model_type]
```

**Flujo de Registro**:
```python
# Registro inicial (se hace al importar el módulo)
ModelRegistry.register('rbf', RBFModelFactory())
ModelRegistry.register('backprop', BackpropModelFactory())

# Uso en la API
factory = ModelRegistry.get_factory('rbf')  # Obtiene RBFModelFactory
model = factory.create_network(X, y, config)  # Crea RBFNetwork configurado
trainer = factory.create_trainer(config)  # Crea RBFTrainer configurado
```

**Ventajas**:
- **Desacoplamiento**: La API no sabe cómo crear modelos específicos, solo usa factories
- **Extensibilidad**: Agregar un modelo nuevo requiere solo crear una factory y registrarla
- **Mantenibilidad**: El código central no crece con cada modelo nuevo
- **Dinámico**: Se pueden registrar modelos en tiempo de ejecución (útil para plugins)

### 8. Dataclasses para Resultados (`api/core/results.py`)

**Problema de Diseño**: Los métodos `train()` y `evaluate()` retornaban diccionarios genéricos (`Dict[str, Any]`), lo cual:
- No proporciona autocompletado en IDEs
- No tiene validación de tipos
- Es fácil cometer errores con nombres de claves
- No documenta claramente qué campos están disponibles

**Solución**: Usar dataclasses de Python para proporcionar tipado fuerte, autocompletado, y validación de estructura.

**Implementación en Código**:
```python
@dataclass
class TrainingResult:
    """
    Resultado del entrenamiento con tipado fuerte.
    
    Proporciona acceso estructurado y con autocompletado
    a los resultados de entrenamiento.
    """
    training_time: float        # Tiempo de entrenamiento en segundos
    final_error: float          # Error final de entrenamiento
    epochs: int                 # Número de épocas ejecutadas
    error_history: List[float]  # Historial de errores por época
    converged: bool             # Si el modelo convergió
    metadata: Dict[str, Any]    # Metadatos adicionales
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para compatibilidad con código existente."""
        return {
            'training_time': self.training_time,
            'final_error': self.final_error,
            'epochs': self.epochs,
            'error_history': self.error_history,
            'converged': self.converged,
            'metadata': self.metadata
        }

@dataclass
class EvaluationResult:
    """
    Resultado de la evaluación con tipado fuerte.
    """
    mse: float                    # Error cuadrático medio
    mae: float                    # Error absoluto medio
    rmse: float                   # Raíz del error cuadrático medio
    r2: float                     # Coeficiente R²
    accuracy: float               # Precisión (para clasificación)
    predictions: Optional[np.ndarray] = None  # Predicciones (opcional)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Uso en la API**:
```python
# Antes (diccionario genérico)
result = net.train(X, y)
print(result['training_time'])  # Sin autocompletado, fácil error de tipeo

# Ahora (dataclass tipado)
result: TrainingResult = net.train(X, y)
print(result.training_time)  # Con autocompletado, validación de tipos
# IDE sugiere: training_time, final_error, epochs, error_history, etc.
```

**Ventajas**:
- **Autocompletado**: Los IDEs sugieren los campos disponibles
- **Validación de tipos**: Python puede verificar tipos con mypy
- **Documentación integrada**: Cada campo tiene su tipo y propósito claro
- **Compatibilidad**: Método `to_dict()` para compatibilidad con código existente

### 9. Factories de Modelos (`api/factories_v2.py`)

**Problema de Diseño**: La API necesita crear instancias de modelos y entrenadores configurados correctamente. Sin factories, la lógica de creación estaría dispersa en múltiples lugares, haciendo difícil mantener consistencia y agregar nuevos modelos.

**Solución**: El patrón Factory - clases especializadas que saben cómo crear instancias configuradas de modelos y entrenadores. Cada factory implementa la interfaz `ModelFactory`.

**Interfaz Base**:
```python
class ModelFactory(ABC):
    """Interfaz abstracta para factories de modelos."""
    
    @abstractmethod
    def create_network(self, X: np.ndarray, y: np.ndarray, config: Any):
        """Crear una instancia del modelo configurado."""
        pass
    
    @abstractmethod
    def create_trainer(self, config: Any):
        """Crear una instancia del entrenador configurado."""
        pass
    
    @abstractmethod
    def get_config_class(self) -> type:
        """Obtener la clase de configuración del modelo."""
        pass
```

**Implementación RBF**:
```python
class RBFModelFactory(BaseModelFactory):
    """Factory para modelos RBF."""
    
    def create_network(self, X: np.ndarray, y: np.ndarray, 
                      config: NeuralNetworkConfig) -> RBFNetwork:
        """
        Crear red RBF configurada.
        
        Flujo:
            1. Determinar número de centros (limitado por n_samples)
            2. Crear función de activación RBF desde string
            3. Crear RBFConfig con todos los parámetros
            4. Retornar RBFNetwork instanciado
        """
        n_centers = min(config.n_centers, X.shape[0])
        activation = self._create_rbf_activation(config.activation_rbf)
        
        rbf_config = RBFConfig(
            n_centers=n_centers,
            sigma=config.sigma,
            activation=activation,
            regularization=config.regularization,
            use_bias=config.use_bias,
            random_state=config.random_state
        )
        
        return RBFNetwork(config=rbf_config)
    
    def create_trainer(self, config: NeuralNetworkConfig) -> RBFTrainer:
        """
        Crear entrenador RBF configurado.
        
        Flujo:
            1. Crear inicializador (k-means o random)
            2. Retornar RBFTrainer con el inicializador
        """
        if config.initializer == 'kmeans':
            init = KMeansInitializer(max_iterations=50)
        else:
            init = RandomInitializer()
        
        return RBFTrainer(initializer=init)
```

**Implementación Backpropagation**:
```python
class BackpropModelFactory(BaseModelFactory):
    """Factory para modelos de retropropagación."""
    
    def create_network(self, X: np.ndarray, y: np.ndarray, 
                      config: NeuralNetworkConfig) -> BackpropNetwork:
        """
        Crear red de retropropagación configurada.
        
        Flujo:
            1. Crear BackpropConfig con capas, learning rate, épocas
            2. Configurar funciones de activación por capa si se especificaron
            3. Retornar BackpropNetwork instanciado
        """
        backprop_config = BackpropConfig(
            hidden_layers=config.hidden_layers,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            batch_size=config.batch_size,
            activation=config.activation_backprop,
            layer_activations=config.layer_activations,
            output_activation=config.output_activation,
            use_bias=config.use_bias,
            random_state=config.random_state
        )
        
        return BackpropNetwork(config=backprop_config)
    
    def create_trainer(self, config: NeuralNetworkConfig) -> BackpropTrainer:
        """Crear entrenador de retropropagación configurado."""
        return BackpropTrainer(verbose=False)
```

**Ventajas**:
- **Encapsulamiento**: La lógica de creación está encapsulada en cada factory
- **Consistencia**: Todas las instancias se crean de manera consistente
- **Testabilidad**: Fácil testear la creación de modelos en aislamiento
- **Extensibilidad**: Agregar un modelo nuevo requiere solo una nueva factory

### 10. Persistencia de Modelos (`api/neural_network.py`)

**Problema de Diseño**: Los modelos entrenados necesitan ser guardados en disco para uso futuro. Sin persistencia, cada vez que se quiera usar un modelo entrenado, habría que reentrenarlo desde cero, lo cual es ineficiente.

**Solución**: Usar pickle de Python para serializar el estado completo del modelo (configuración, pesos, entrenador, logs de entrenamiento) y guardarlo en disco.

**Implementación en Código**:
```python
def save(self, filepath: str) -> None:
    """
    Guardar el modelo entrenado en disco.
    
    Flujo:
        1. Verificar que el modelo está entrenado
        2. Crear diccionario con todo el estado del modelo
        3. Serializar usando pickle
        4. Escribir al archivo especificado
    
    Estado guardado:
        - model_type: Tipo de modelo (RBF o Backprop)
        - config: Configuración completa del modelo
        - model: Instancia del modelo con pesos entrenados
        - training_log: Historial de entrenamiento
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
    
    Flujo:
        1. Leer el archivo pickle
        2. Extraer el estado guardado
        3. Crear nueva instancia de NeuralNetwork
        4. Restaurar el estado entrenado
        5. Retornar la instancia lista para usar
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

**Uso**:
```python
# Entrenar y guardar
net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)
net.train(X, y, verbose=True)
net.save('models/rbf_model.pkl')

# Cargar y usar
net_loaded = NeuralNetwork.load('models/rbf_model.pkl')
predictions = net_loaded.predict(X_test)  # Funciona inmediatamente
```

**Ventajas**:
- **Eficiencia**: No requiere reentrenamiento
- **Portabilidad**: Los modelos pueden compartirse entre sesiones
- **Estado completo**: Se guarda todo (configuración, pesos, logs)
- **Simple**: Usa pickle estándar de Python

### 11. Reproducibilidad (`api/neural_network.py`)

**Problema de Diseño**: Las redes neuronales usan aleatoriedad en múltiples lugares (inicialización de pesos, mezcla de datos, inicialización de centros). Sin control de semilla, cada ejecución produce resultados diferentes, lo cual dificulta depuración y comparación de resultados.

**Solución**: Un método estático `set_seed()` que controla `numpy.random` para asegurar resultados reproducibles.

**Implementación en Código**:
```python
@staticmethod
def set_seed(seed: int) -> None:
    """
    Establecer semilla aleatoria para reproducibilidad.
    
    Flujo:
        1. Llama a np.random.seed(seed)
        2. Esto controla todas las operaciones aleatorias de NumPy
        3. Afecta: inicialización de pesos, mezcla de datos, inicialización de centros
    
    Operaciones afectadas:
        - Inicialización Xavier/Glorot de pesos
        - Mezcla de datos en cada época (backprop)
        - Inicialización aleatoria de centros RBF
        - K-means (usa aleatoriedad en inicialización)
    """
    np.random.seed(seed)
```

**Uso**:
```python
# Para reproducibilidad
NeuralNetwork.set_seed(42)

net1 = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)
net1.train(X, y)

# En otra sesión o ejecución
NeuralNetwork.set_seed(42)

net2 = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)
net2.train(X, y)

# net1 y net2 tendrán exactamente los mismos pesos y resultados
```

**Ventajas**:
- **Reproducibilidad**: Mismos datos + misma semilla = mismos resultados
- **Depuración**: Facilita encontrar bugs consistentes
- **Comparación**: Permite comparar diferentes configuraciones de manera justa
- **Testing**: Esencial para tests unitarios deterministas

### 12. Inspección de Capas (`api/neural_network.py`)

**Problema de Diseño**: Para redes de retropropagación con múltiples capas, los usuarios necesitan inspeccionar pesos y configuración de capas individuales para entender qué está aprendiendo el modelo y depurar problemas.

**Solución**: Métodos específicos para inspeccionar capas individuales (`get_layer_weights()`) y obtener información de todas las capas (`get_layer_info()`).

**Implementación en Código**:
```python
def get_layer_weights(self, layer_index: int) -> LayerWeights:
    """
    Obtener pesos y bias de una capa específica (solo backprop).
    
    Flujo:
        1. Verificar que el modelo está entrenado
        2. Verificar que no es RBF (RBF no tiene capas múltiples)
        3. Manejar índices negativos (ej: -1 para última capa)
        4. Validar que el índice está en rango
        5. Retornar LayerWeights (dataclass) con pesos, bias, activación
    
    Retorna:
        LayerWeights con:
        - layer_index: Índice de la capa
        - layer_type: 'hidden' o 'output'
        - input_size: Tamaño de entrada
        - output_size: Tamaño de salida
        - weights: Matriz de pesos (copia)
        - bias: Vector de bias (copia)
        - activation: Función de activación
        - use_bias: Si usa bias
    """
    self._ensure_fitted()
    
    if self.model_type == ModelType.RBF:
        raise ValueError("get_layer_weights no está disponible para RBF")
    
    # Manejar índices negativos
    if layer_index < 0:
        layer_index = len(self.model.layers) + layer_index
    
    layer = self.model.layers[layer_index]
    
    return LayerWeights(
        layer_index=layer_index,
        layer_type='hidden' if layer_index < len(self.model.layers) - 1 else 'output',
        input_size=layer.input_size,
        output_size=layer.output_size,
        weights=layer.weights.copy(),
        bias=layer.bias.copy() if layer.bias is not None else None,
        activation=str(layer.activation),
        use_bias=layer.use_bias
    )

def get_layer_info(self) -> List[Dict[str, Any]]:
    """
    Obtener información detallada de todas las capas.
    
    Flujo:
        1. Verificar que el modelo está entrenado
        2. Para RBF: retornar información de la capa RBF única
        3. Para Backprop: iterar sobre todas las capas y recopilar info
        4. Retornar lista de diccionarios con información de cada capa
    
    Retorna:
        Lista con información de cada capa:
        - layer_index: Índice de la capa
        - layer_type: Tipo de capa
        - input_size: Tamaño de entrada
        - output_size: Tamaño de salida
        - activation: Función de activación
        - use_bias: Si usa bias
        - weights_shape: Forma de los pesos
        - bias_shape: Forma del bias
    """
    self._ensure_fitted()
    
    if self.model_type == ModelType.RBF:
        return [{
            'layer_type': 'rbf',
            'n_centers': self.model.config.n_centers,
            'activation': str(self.model.config.activation),
            'sigma': self.model.config.sigma,
            'weights_shape': self.model.weights.shape,
            'bias': self.model.bias,
            'centers_shape': self.model.centers.shape
        }]
    
    layer_info = []
    for i, layer in enumerate(self.model.layers):
        layer_info.append({
            'layer_index': i,
            'layer_type': 'hidden' if i < len(self.model.layers) - 1 else 'output',
            'input_size': layer.input_size,
            'output_size': layer.output_size,
            'activation': str(layer.activation),
            'use_bias': layer.use_bias,
            'weights_shape': layer.weights.shape,
            'bias_shape': layer.bias.shape if layer.bias is not None else None
        })
    
    return layer_info
```

**Uso**:
```python
# Ver información de todas las capas
layer_info = net.get_layer_info()
for layer in layer_info:
    print(f"Capa {layer['layer_index']}: {layer['layer_type']}")
    print(f"  Forma: {layer['input_size']} -> {layer['output_size']}")
    print(f"  Activación: {layer['activation']}")

# Ver pesos de una capa específica
layer_0 = net.get_layer_weights(0)  # Primera capa oculta
print(f"Pesos: {layer_0.weights.shape}")
print(f"Bias: {layer_0.bias}")

layer_output = net.get_layer_weights(-1)  # Capa de salida
print(f"Pesos salida: {layer_output.weights.shape}")
```

**Ventajas**:
- **Transparencia**: Permite entender qué está aprendiendo el modelo
- **Depuración**: Facilita identificar problemas en capas específicas
- **Análisis**: Permite analizar la distribución de pesos por capa
- **Flexibilidad**: Soporta índices negativos para acceso fácil

### 13. Capa Densa de Retropropagación (`src/models/backprop/layer.py`)

**Problema Matemático**: Implementar neurona densa con pase hacia adelante y hacia atrás. Esta capa es el bloque fundamental de las redes de retropropagación - cada neurona toma las entradas, las multiplica por pesos, suma un bias, aplica una función de activación no lineal, y también puede propagar gradientes hacia atrás durante el entrenamiento.

**Arquitectura de una Capa Densa**:
```
Entrada X (n_samples, input_size)
    ↓
Multiplicación por pesos: X @ W
    - W: (input_size, output_size)
    - Resultado: (n_samples, output_size)
    ↓
Agregar bias: + b
    - b: (output_size,)
    ↓
Pre-activación z = X @ W + b
    ↓
Aplicar función de activación: phi(z)
    - phi: sigmoid, tanh, relu, etc.
    ↓
Salida a_out = phi(z) (n_samples, output_size)
```

**Fórmulas Matemáticas**:

**Pase hacia adelante**:
```
z = a_in @ W + b
a_out = phi(z)
```

**Pase hacia atrás (retropropagación)**:
```
delta = gradient_in * phi'(z)
gradient_W = a_in^T @ delta
gradient_b = sum(delta)
gradient_in = delta @ W^T
```

**Actualización de pesos (descenso de gradiente)**:
```
W_{t+1} = W_t - alpha * gradient_W
b_{t+1} = b_t - alpha * gradient_b
```

**Inicialización de Pesos (Xavier/Glorot)**:
```
limit = sqrt(6 / (input_size + output_size))
W ~ Uniform(-limit, limit)
b = 0
```

**Flujo Completo de Inicialización** (`DenseLayer.__init__()` en `src/models/backprop/layer.py`):

```
1. Almacenar dimensiones
   - self.input_size = input_size
   - self.output_size = output_size
   ↓
2. Mapear nombre de activación a clase
   - self.activation = self._get_activation_function()
   - Mapeo: 'sigmoid' -> SigmoidActivation, 'tanh' -> TanhActivation, etc.
   ↓
3. Inicializar pesos con Xavier/Glorot
   - limit = sqrt(6 / (input_size + output_size))
   - self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
   ↓
4. Inicializar bias a ceros
   - Si use_bias=True: self.bias = np.zeros(output_size)
   - Si use_bias=False: self.bias = None
   ↓
5. Inicializar variables para backpropagation
   - self.last_input = None
   - self.last_output = None
   - self.last_z = None
```

**Implementación en Código** (`src/models/backprop/layer.py`):
```python
def __init__(self, input_size: int, output_size: int, activation: str = 'sigmoid', use_bias: bool = True):
    """
    Inicializar la capa densa.
    """
    self.input_size = input_size
    self.output_size = output_size
    self.activation_name = activation.lower()
    self.use_bias = use_bias
    
    # Paso 2: Mapear nombres estilo MATLAB a clases de activación
    self.activation = self._get_activation_function()
    
    # Paso 3: Inicializar pesos con inicialización Xavier/Glorot
    limit = np.sqrt(6 / (input_size + output_size))
    self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
    
    # Paso 4: Inicializar bias a ceros
    if self.use_bias:
        self.bias = np.zeros(output_size)
    else:
        self.bias = None
    
    # Paso 5: Almacenar para retropropagación
    self.last_input = None
    self.last_output = None
    self.last_z = None
```

**Flujo de Pase Hacia Adelante** (`DenseLayer.forward()` en `src/models/backprop/layer.py`):

```
1. Guardar entrada para backpropagation
   - self.last_input = X
   ↓
2. Calcular pre-activación z
   - z = X @ self.weights
   - Si use_bias=True: z += self.bias
   - self.last_z = z (guardar para backprop)
   ↓
3. Aplicar función de activación
   - output = self.activation.compute(z)
   - self.last_output = output (guardar para backprop)
   ↓
4. Retornar activación
```

**Implementación en Código**:
```python
def forward(self, X: np.ndarray) -> np.ndarray:
    """
    Calcular el pase hacia adelante de la capa.
    
    Fórmula: z = X @ W + b, y = phi(z)
    """
    self.last_input = X
    
    z = X @ self.weights
    if self.use_bias:
        z += self.bias
    self.last_z = z
    
    output = self.activation.compute(z)
    self.last_output = output
    
    return output
```

**Flujo de Pase Hacia Atrás** (`DenseLayer.backward()` en `src/models/backprop/layer.py`):

```
1. Calcular derivada de activación en z
   - activation_grad = self.activation.derivative(self.last_z)
   - phi'(z)
   ↓
2. Calcular delta (error local)
   - delta = output_gradient * activation_grad
   - delta = gradient_in * phi'(z)
   ↓
3. Calcular gradiente de pesos
   - self.weights_gradient = self.last_input.T @ delta
   - gradient_W = a_in^T @ delta
   ↓
4. Calcular gradiente de bias
   - Si use_bias=True: self.bias_gradient = np.sum(delta, axis=0)
   - gradient_b = sum(delta)
   ↓
5. Calcular gradiente de entrada (para capa anterior)
   - input_gradient = delta @ self.weights.T
   - gradient_in = delta @ W^T
   ↓
6. Retornar gradiente de entrada
```

**Implementación en Código**:
```python
def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    """
    Calcular el pase hacia atrás (computación de gradiente).
    
    Fórmulas:
    - delta = gradient_y @ phi'(z)
    - gradient_W = X.T @ delta
    - gradient_b = sum(delta)
    - gradient_X = delta @ W.T
    """
    activation_grad = self.activation.derivative(self.last_z)
    delta = output_gradient * activation_grad
    
    self.weights_gradient = self.last_input.T @ delta
    
    if self.use_bias:
        self.bias_gradient = np.sum(delta, axis=0)
    else:
        self.bias_gradient = None
    
    input_gradient = delta @ self.weights.T
    
    return input_gradient
```

**Flujo de Actualización de Pesos** (`DenseLayer.update_weights()` en `src/models/backprop/layer.py`):

```
1. Actualizar pesos con descenso de gradiente
   - self.weights -= learning_rate * self.weights_gradient
   - W_{t+1} = W_t - alpha * gradient_W
   ↓
2. Actualizar bias si se usa
   - Si use_bias=True: self.bias -= learning_rate * self.bias_gradient
   - b_{t+1} = b_t - alpha * gradient_b
```

**Implementación en Código**:
```python
def update_weights(self, learning_rate: float) -> None:
    """
    Actualizar pesos y bias usando gradientes calculados.
    
    Fórmula: theta_{t+1} = theta_t - alpha * gradient_L(theta_t)
    """
    self.weights -= learning_rate * self.weights_gradient
    
    if self.use_bias and self.bias_gradient is not None:
        self.bias -= learning_rate * self.bias_gradient
```

**Mapeo Código-Matemáticas**:
- `X @ self.weights` → `a_in @ W` : producto matricial
- `z += self.bias` → `z = a_in @ W + b` : suma de bias
- `self.activation.compute(z)` → `phi(z)` : función de activación
- `self.activation.derivative(self.last_z)` → `phi'(z)` : derivada de activación
- `output_gradient * activation_grad` → `gradient_in * phi'(z)` : regla de la cadena
- `self.last_input.T @ delta` → `a_in^T @ delta` : gradiente de pesos
- `np.sum(delta, axis=0)` → `sum(delta)` : gradiente de bias
- `delta @ self.weights.T` → `delta @ W^T` : gradiente de entrada
- `self.weights -= learning_rate * self.weights_gradient` → `W_{t+1} = W_t - alpha * gradient_W` : descenso de gradiente

### 8. Red de Retropropagación (`src/models/backprop/network.py`)

**Problema Matemático**: Coordinar múltiples capas densas para pase hacia adelante y hacia atrás. Esta red es como una cadena de procesamiento - conecta múltiples capas densas en secuencia, donde cada capa toma la salida de la anterior como entrada. Durante el entrenamiento, coordina el flujo de datos hacia adelante y la propagación de gradientes hacia atrás a través de todas las capas.

**Arquitectura de la Red de Retropropagación**:
```
Entrada X (n_samples, n_features)
    ↓
Capa 1 (hidden_layers[0] neuronas)
    - z_1 = X @ W_1 + b_1
    - a_1 = phi_1(z_1)
    ↓
Capa 2 (hidden_layers[1] neuronas)
    - z_2 = a_1 @ W_2 + b_2
    - a_2 = phi_2(z_2)
    ↓
...
    ↓
Capa L-1 (hidden_layers[-1] neuronas)
    - z_{L-1} = a_{L-2} @ W_{L-1} + b_{L-1}
    - a_{L-1} = phi_{L-1}(z_{L-1})
    ↓
Capa L (n_outputs neuronas - salida)
    - z_L = a_{L-1} @ W_L + b_L
    - a_L = phi_L(z_L)
    ↓
Salida y_pred = a_L (n_samples, n_outputs)
```

**Fórmulas Matemáticas**:

**Pase hacia adelante** (para cada capa l):
```
a_0 = X
z_l = a_{l-1} @ W_l + b_l
a_l = phi_l(z_l) para l = 1, 2, ..., L
y_pred = a_L
```

**Pase hacia atrás (retropropagación del error)**:
```
delta_L = (y_pred - y) * phi'_L(z_L)
delta_l = delta_{l+1} @ W_{l+1}^T * phi'_l(z_l) para l = L-1, ..., 1
```

**Gradientes**:
```
gradient_W_l = a_{l-1}^T @ delta_l
gradient_b_l = sum(delta_l)
```

**Actualización de pesos (descenso de gradiente)**:
```
W_l(t+1) = W_l(t) - alpha * gradient_W_l
b_l(t+1) = b_l(t) - alpha * gradient_b_l
```

**Flujo Completo de Construcción de la Red** (`BackpropNetwork._build_network()` en `src/models/backprop/network.py`):

```
1. Crear lista de tamaños de capas
   - layer_sizes = [input_size] + hidden_layers + [output_size]
   - Ejemplo: [2, 10, 5, 1] para 2 entradas, 10 neuronas capa 1, 5 neuronas capa 2, 1 salida
   ↓
2. Para cada par de capas consecutivas (i, i+1):
   a. Determinar función de activación
      - Si es capa oculta (i < len(layer_sizes) - 2):
        - Usar layer_activations[i] si está configurado
        - Sino usar activation por defecto
      - Si es capa de salida (i == len(layer_sizes) - 2):
        - Usar output_activation
   b. Crear DenseLayer
      - layer = DenseLayer(
          input_size=layer_sizes[i],
          output_size=layer_sizes[i+1],
          activation=activación_determinada,
          use_bias=config.use_bias
        )
   c. Agregar a self.layers
   ↓
3. Verificar que se crearon capas
   - if len(self.layers) == 0: raise RuntimeError
```

**Implementación en Código** (`src/models/backprop/network.py`):
```python
def _build_network(self, input_size: int, output_size: int) -> None:
    """
    Construir la arquitectura de red basada en configuración.
    """
    self.layers = []
    
    # Paso 1: Construir capas ocultas
    layer_sizes = [input_size] + self.config.hidden_layers + [output_size]
    
    # Paso 2: Para cada par de capas consecutivas
    for i in range(len(layer_sizes) - 1):
        # Paso 2a: Determinar función de activación
        if i < len(layer_sizes) - 2:
            # Capa oculta
            if self.config.layer_activations is not None:
                activation = self.config.layer_activations[i]
            else:
                activation = self.config.activation
        else:
            # Capa de salida
            activation = self.config.output_activation
        
        # Paso 2b: Crear capa densa
        layer = DenseLayer(
            input_size=layer_sizes[i],
            output_size=layer_sizes[i + 1],
            activation=activation,
            use_bias=self.config.use_bias
        )
        self.layers.append(layer)
```

**Flujo Completo de Entrenamiento** (`BackpropNetwork.fit()` en `src/models/backprop/network.py`):

```
1. Validar formas de entrada
   - Verificar que X es 2D
   - Convertir y a 2D si es 1D
   - Verificar que n_samples de X == n_samples de y
   ↓
2. Almacenar dimensiones
   - self.n_features_ = X.shape[1]
   - self.n_outputs_ = y.shape[1]
   ↓
3. Construir arquitectura de red
   - self._build_network(self.n_features_, self.n_outputs_)
   ↓
4. Verificar que las capas fueron creadas
   - if len(self.layers) == 0: raise RuntimeError
   ↓
5. Configurar tamaño de batch
   - batch_size = config.batch_size si != -1, sino n_samples
   ↓
6. Bucle de entrenamiento (para cada época)
   a. Mezclar datos aleatoriamente
      - indices = np.random.permutation(n_samples)
      - X_shuffled = X[indices]
      - y_shuffled = y[indices]
   b. Descenso de gradiente mini-batch
      Para cada batch:
        i. Pase hacia adelante: output = self._forward_pass(X_batch)
        ii. Calcular gradiente de error: error_gradient = output - y_batch
        iii. Pase hacia atrás: self._backward_pass(error_gradient)
        iv. Actualizar pesos: layer.update_weights(learning_rate)
   ↓
7. Marcar modelo como entrenado
   - self.is_fitted = True
```

**Implementación en Código**:
```python
def fit(self, X: np.ndarray, y: np.ndarray) -> None:
    """
    Entrenar la red de retropropagación.
    
    El entrenamiento minimiza: L(theta) = (1/2) * sum((y - y_pred)^2)
    usando descenso de gradiente: theta_{t+1} = theta_t - alpha * gradient_L(theta_t)
    """
    # Paso 1: Validar formas de entrada
    X = np.asarray(X)
    y = np.asarray(y)
    
    if X.ndim != 2:
        raise InvalidInputError(f"X debe ser array 2D, se obtuvo forma {X.shape}")
    
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    if y.ndim != 2:
        raise InvalidInputError(f"y debe ser array 1D o 2D, se obtuvo forma {y.shape}")
    
    if X.shape[0] != y.shape[0]:
        raise InvalidInputError(
            f"Discrepancia en número de muestras: X tiene {X.shape[0]}, y tiene {y.shape[0]}"
        )
    
    # Paso 2: Almacenar dimensiones
    self.n_features_ = X.shape[1]
    self.n_outputs_ = y.shape[1]
    
    # Paso 3: Construir arquitectura de red
    self._build_network(self.n_features_, self.n_outputs_)
    
    # Paso 4: Verificar que las capas fueron creadas
    if len(self.layers) == 0:
        raise RuntimeError("Las capas de red no fueron creadas correctamente")
    
    # Paso 5: Configurar tamaño de batch
    n_samples = X.shape[0]
    batch_size = self.config.batch_size if self.config.batch_size != -1 else n_samples
    
    # Paso 6: Bucle de entrenamiento
    for epoch in range(self.config.epochs):
        # Paso 6a: Mezclar datos cada época
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Paso 6b: Descenso de gradiente mini-batch
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:batch_end]
            y_batch = y_shuffled[i:batch_end]
            
            # Pase hacia adelante
            output = self._forward_pass(X_batch)
            
            # Calcular gradiente de error
            error_gradient = output - y_batch
            
            # Pase hacia atrás
            self._backward_pass(error_gradient)
            
            # Actualizar pesos
            for layer in self.layers:
                layer.update_weights(self.config.learning_rate)
    
    # Paso 7: Marcar como entrenado
    self.is_fitted = True
```

**Flujo de Pase Hacia Adelante** (`BackpropNetwork._forward_pass()` en `src/models/backprop/network.py`):

```
1. Inicializar activación con entrada X
   - activation = X
   - a_0 = X
   ↓
2. Para cada capa en self.layers:
   a. Calcular pase hacia adelante de la capa
      - activation = layer.forward(activation)
      - a_l = phi(a_{l-1} @ W_l + b_l)
   ↓
3. Retornar activación final
   - return activation
   - y_pred = a_L
```

**Implementación en Código**:
```python
def _forward_pass(self, X: np.ndarray) -> np.ndarray:
    """
    Calcular pase hacia adelante a través de todas las capas.
    
    Fórmula: a_l = phi(z_l), donde z_l = a_{l-1} @ W_l + b_l
    """
    activation = X
    for layer in self.layers:
        activation = layer.forward(activation)
    return activation
```

**Flujo de Pase Hacia Atrás** (`BackpropNetwork._backward_pass()` en `src/models/backprop/network.py`):

```
1. Inicializar gradiente con gradiente de salida
   - gradient = output_gradient
   - delta_L = (y_pred - y)
   ↓
2. Para cada capa en orden inverso (de salida a entrada):
   a. Calcular pase hacia atrás de la capa
      - gradient = layer.backward(gradient)
      - delta_l = delta_{l+1} @ W_{l+1}^T * phi'(z_l)
   ↓
3. Los gradientes se almacenan en cada capa (weights_gradient, bias_gradient)
```

**Implementación en Código**:
```python
def _backward_pass(self, output_gradient: np.ndarray) -> None:
    """
    Calcular pase hacia atrás a través de todas las capas.
    
    Fórmula: delta_l = delta_{l+1} @ W_{l+1} @ phi'(z_l)
    """
    gradient = output_gradient
    for layer in reversed(self.layers):
        gradient = layer.backward(gradient)
```

**Mapeo Código-Matemáticas**:
- `layer_sizes = [input_size] + hidden_layers + [output_size]` → Construcción de arquitectura
- `layer.forward(activation)` → `a_l = phi(a_{l-1} @ W_l + b_l)` : activación de cada capa
- `output - y_batch` → `y_pred - y` : error de predicción
- `layer.backward(gradient)` → `delta_l = delta_{l+1} @ W_{l+1}^T * phi'(z_l)` : propagación de delta
- `layer.update_weights(learning_rate)` → `W_{t+1} = W_t - alpha * gradient_W` : actualización de pesos
- `np.random.permutation(n_samples)` → Mezcla aleatoria de datos para SGD

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
