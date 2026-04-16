# Operaciones Matemáticas

Este documento explica todas las operaciones matemáticas utilizadas en el código de manera sencilla y paso a paso. Para cada operación, verás:
- **Qué hace** en lenguaje simple
- **Por qué se usa** en el contexto del proyecto
- **Cómo funciona** paso a paso
- **La fórmula matemática** para referencia técnica

## Tabla de Contenidos

1. [Álgebra Lineal](#álgebra-lineal)
2. [Funciones de Activación](#funciones-de-activación)
3. [Distancias y Métricas](#distancias-y-métricas)
4. [Optimización y Gradientes](#optimización-y-gradientes)
5. [Inicialización de Pesos](#inicialización-de-pesos)
6. [Métricas de Evaluación](#métricas-de-evaluación)

---

## Álgebra Lineal

### 1. Pseudoinversa de Moore-Penrose (pinv)

**¿Qué hace en palabras simples?**

La pseudoinversa es como una "inversa generalizada" que funciona incluso cuando una matriz no es cuadrada o no tiene inversa normal. Imagina que quieres resolver un sistema de ecuaciones donde tienes más ecuaciones que incógnitas (como cuando tienes 100 datos pero solo 20 parámetros). La pseudoinversa encuentra la mejor solución posible.

**Por qué se usa en este proyecto:**

En redes RBF, necesitamos encontrar los pesos de salida que hagan que la red prediga lo más cerca posible de los valores objetivo. Tenemos muchos datos de entrenamiento (filas) pero pocos centros RBF (columnas), así que no podemos usar la inversa normal. La pseudoinversa nos da la solución óptima en un solo paso, sin necesidad de iterar múltiples veces como en otros métodos.

**Cómo funciona paso a paso:**

1. **Sin regularización:** El código toma la matriz Φ (la matriz de diseño con las activaciones RBF) y calcula su pseudoinversa directamente usando una función de NumPy. Luego multiplica esta pseudoinversa por los valores objetivo y para obtener los pesos W.

2. **Con regularización (versión manual):** Cuando hay regularización, el código hace lo siguiente paso a paso:
   - Paso 1: Multiplica la matriz Φ por su transpuesta (Φ^T Φ) - esto crea una matriz cuadrada
   - Paso 2: Crea una matriz identidad I (una matriz diagonal con 1s)
   - Paso 3: Multiplica la matriz identidad por el parámetro de regularización λ
   - Paso 4: Suma el resultado al paso 1: Φ^T Φ + λI - esto estabiliza la matriz
   - Paso 5: Invierte la matriz resultante - ahora es invertible gracias a la regularización
   - Paso 6: Multiplica la inversa por la transpuesta de Φ
   - Paso 7: Multiplica todo por y para obtener los pesos finales W

**La fórmula matemática:**

```
A⁺ = (A^T A)^(-1) A^T  (versión simple)
A⁺ = (A^T A + λI)^(-1) A^T  (con regularización)
```

Donde:
- A: La matriz de entrada (Φ en nuestro caso)
- A^T: La transpuesta de A (filas se vuelven columnas)
- λ: Parámetro de regularización (un pequeño número positivo)
- I: Matriz identidad (unos en la diagonal, ceros fuera)
- (-1): Inversa matricial

**Diferencia con métodos clásicos:**

Los métodos tradicionales de entrenamiento neuronal (como descenso de gradiente) funcionan así:
- Empiezan con pesos aleatorios
- Calculan el error
- Ajustan los pesos un poco
- Repiten muchas veces (épocas) hasta mejorar

La pseudoinversa es diferente:
- Calcula la solución óptima en un solo paso
- No requiere múltiples iteraciones
- Siempre da el mismo resultado (determinista)
- Es más rápida para problemas pequeños/medianos

**Implementación en Código:**

Archivo: `src/models/rbf/solver.py`

```python
def solve_pseudoinverse(Phi: np.ndarray, y: np.ndarray, regularization: float = 0.0) -> np.ndarray:
    """
    Resolver pesos óptimos usando pseudoinversa.
    """
    n_samples, n_centers = Phi.shape
    
    if regularization > 0:
        # VERSIÓN CON REGULARIZACIÓN (manual paso a paso)
        
        # Paso 1: Multiplicar matriz por su transpuesta
        Phi_T_Phi = Phi.T @ Phi  # Esto crea una matriz cuadrada
        
        # Paso 2: Crear matriz identidad (diagonal de 1s)
        identity = np.eye(n_centers)
        
        # Paso 3: Agregar regularización (estabiliza la matriz)
        regularized_matrix = Phi_T_Phi + regularization * identity
        
        # Paso 4: Invertir la matriz regularizada
        inv_regularized = np.linalg.inv(regularized_matrix)
        
        # Paso 5: Multiplicar por transpuesta de Phi
        pseudoinverse = inv_regularized @ Phi.T
        
        # Paso 6: Multiplicar por y para obtener pesos finales
        W = pseudoinverse @ y
    else:
        # VERSIÓN SIN REGULARIZACIÓN (directa con NumPy)
        
        # Paso 1: Calcular pseudoinversa directamente
        pseudoinverse = np.linalg.pinv(Phi)
        
        # Paso 2: Multiplicar por y para obtener pesos
        W = pseudoinverse @ y
    
    return W
```

**Referencia en Guía:** Ver sección [Por qué Pseudoinversa](guide.md#por-qu-pseudoinversa) en `guide.md`

---

### 2. Multiplicación Matricial (@)

**¿Qué hace en palabras simples?**

La multiplicación matricial combina dos matrices para crear una nueva matriz. Es como multiplicar listas de números de una manera específica: cada elemento del resultado se obtiene multiplicando filas de la primera matriz por columnas de la segunda y sumando los productos.

**Por qué se usa en este proyecto:**

Es la operación fundamental en redes neuronales. Se usa para:
- Calcular la salida de una capa: `salida = entrada × pesos`
- Propagar errores hacia atrás: `gradiente = error × pesos`
- Calcular gradientes: `gradiente_pesos = entrada^T × error`

**Cómo funciona paso a paso:**

Para multiplicar dos matrices A y B:
1. Toma la primera fila de A
2. Toma la primera columna de B
3. Multiplica cada par de números correspondientes
4. Suma todos los productos
5. Este resultado es el primer elemento de la matriz resultado
6. Repite para todas las combinaciones fila-columna

**La fórmula matemática:**

Para matrices A (m×n) y B (n×p):

```
C[i,j] = Σ(k=1 to n) A[i,k] × B[k,j]
```

Esto significa: el elemento en la fila i, columna j de C es la suma de multiplicar los elementos de la fila i de A por los elementos de la columna j de B.

**Implementación en Código:**

Archivo: `src/models/backprop/layer.py`

```python
# Pase hacia adelante: z = X @ W + b
z = X @ self.weights  # Multiplicar entrada por pesos

# Gradiente de pesos: ∇W = X^T @ δ
self.weights_gradient = self.last_input.T @ delta  # Multiplicar transpuesta por error

# Gradiente de entrada: ∇X = δ @ W^T
input_gradient = delta @ self.weights.T  # Multiplicar error por transpuesta de pesos
```

---

### 3. Matriz Identidad (np.eye)

**¿Qué hace en palabras simples?**

La matriz identidad es como el número 1 para matrices. Es una matriz cuadrada que tiene 1s en la diagonal principal (de esquina a esquina) y 0s en todos los demás lugares. Cuando multiplicas cualquier matriz por la identidad, obtienes la misma matriz original.

**Por qué se usa en este proyecto:**

Se usa en regularización para estabilizar matrices que casi no tienen inversa. Al agregar una pequeña cantidad de la matriz identidad multiplicada por un parámetro λ, nos aseguramos de que la matriz sea invertible.

**Cómo funciona paso a paso:**

1. NumPy crea una matriz cuadrada del tamaño especificado
2. Pone 1.0 en todas las posiciones donde el número de fila equals el número de columna
3. Pone 0.0 en todas las demás posiciones
4. El resultado es una matriz que actúa como el "neutro" en multiplicación matricial

**La fórmula matemática:**

```
I[i,j] = 1 si i = j
I[i,j] = 0 si i ≠ j
```

Ejemplo para tamaño 3×3:
```
[1 0 0]
[0 1 0]
[0 0 1]
```

**Implementación en Código:**

Archivo: `src/models/rbf/solver.py`

```python
# Crear matriz identidad del tamaño del número de centros
identity = np.eye(n_centers)  # Crea matriz con 1s en diagonal

# Usarla en regularización
regularized_matrix = Phi_T_Phi + regularization * identity
# Esto suma un poco de "estabilidad" a la matriz
```

---

### 4. Inversa Matricial (np.linalg.inv)

**¿Qué hace en palabras simples?**

La inversa de una matriz es como el "recíproco" para matrices. Si tienes una matriz A y su inversa A^(-1), cuando las multiplicas obtienes la matriz identidad (el equivalente matricial de 1). No todas las matrices tienen inversa.

**Por qué se usa en este proyecto:**

Se usa en la implementación manual de pseudoinversa con regularización. Necesitamos invertir la matriz Φ^T Φ + λI para calcular la pseudoinversa.

**Cómo funciona paso a paso:**

NumPy usa algoritmos numéricos complejos (como descomposición LU) para:
1. Verificar si la matriz es invertible
2. Descomponer la matriz en factores más simples
3. Calcular la inversa usando esos factores
4. Devolver la matriz inversa

**La fórmula matemática:**

```
A @ A^(-1) = A^(-1) @ A = I
```

Donde I es la matriz identidad.

**Implementación en Código:**

Archivo: `src/models/rbf/solver.py`

```python
# Invertir la matriz regularizada
inv_regularized = np.linalg.inv(regularized_matrix)
# Esto calcula (Φ^T Φ + λI)^(-1)
```

---

### 5. Norma Euclidiana (np.linalg.norm)

**¿Qué hace en palabras simples?**

La norma euclidiana es la longitud de un vector. Es como medir la distancia en línea recta desde el origen hasta un punto en el espacio. Para dos puntos, mide la distancia entre ellos.

**Por qué se usa en este proyecto:**

Se usa en el algoritmo k-means para verificar si los centroides han convergido. Si el cambio en la posición de los centroides es menor que una tolerancia pequeña, el algoritmo termina.

**Cómo funciona paso a paso:**

Para un vector [x, y, z]:
1. Eleva cada componente al cuadrado: x², y², z²
2. Suma todos los cuadrados: x² + y² + z²
3. Toma la raíz cuadrada de la suma: √(x² + y² + z²)
4. El resultado es la longitud del vector

**La fórmula matemática:**

Para un vector x:

```
||x|| = √(Σ(i=1 to n) x[i]²)
```

Para la distancia entre dos puntos x y y:

```
||x - y|| = √(Σ(i=1 to n) (x[i] - y[i])²)
```

**Implementación en Código:**

Archivo: `src/training/initializer.py`

```python
# Calcular cuánto cambiaron los centroides
centroid_shift = np.linalg.norm(new_centroids - centroids)

# Si el cambio es muy pequeño, convergimos
if centroid_shift < self.tolerance:
    break  # Terminar el bucle
```

---

## Funciones de Activación

### 6. Función Sigmoide (sigmoid / logsig)

**¿Qué hace en palabras simples?**

La función sigmoide toma cualquier número (positivo, negativo, o cero) y lo convierte en un valor entre 0 y 1. Es como una "campana" suave: valores muy negativos dan casi 0, valores muy positivos dan casi 1, y valores cercanos a 0 dan 0.5.

**Por qué se usa en este proyecto:**

Es útil cuando quieres que la salida represente una probabilidad (entre 0 y 1). También se usa en capas ocultas de redes clásicas porque su derivada es fácil de calcular.

**Cómo funciona paso a paso:**

Para un valor de entrada z:
1. Calcula e^(-z) (la exponencial negativa de z)
2. Suma 1 a ese resultado: 1 + e^(-z)
3. Divide 1 entre ese resultado: 1 / (1 + e^(-z))
4. El resultado está siempre entre 0 y 1

**La fórmula matemática:**

```
φ(z) = 1 / (1 + e^(-z))
```

**Derivada (para backpropagation):**

```
φ'(z) = φ(z) × (1 - φ(z))
```

Esto es útil porque puedes calcular la derivada usando el valor de la función misma.

**Implementación en Código:**

Archivo: `src/core/activation.py`

```python
class SigmoidActivation(ActivationFunction):
    def compute(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        # Paso 1: Calcular e^(-x)
        # Paso 2: Sumar 1
        # Paso 3: Dividir 1 entre el resultado
        return 1.0 / (1.0 + np.exp(-x))
    
    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        s = self.compute(x)
        return s * (1.0 - s)  # Derivada usando el valor de la función
```

---

### 7. Tangente Hiperbólica (tanh / tansig)

**¿Qué hace en palabras simples?**

La tangente hiperbólica es similar a la sigmoide pero en lugar de dar valores entre 0 y 1, da valores entre -1 y 1. Valores muy negativos dan casi -1, valores muy positivos dan casi 1, y cero da 0.

**Por qué se usa en este proyecto:**

Es útil porque centra los datos alrededor de cero, lo cual ayuda a que el entrenamiento sea más estable en redes profundas. También evita problemas donde los gradientes se vuelven muy pequeños.

**Cómo funciona paso a paso:**

Para un valor de entrada z:
1. Calcula e^z y e^(-z)
2. Resta: e^z - e^(-z)
3. Suma: e^z + e^(-z)
4. Divide la resta entre la suma: (e^z - e^(-z)) / (e^z + e^(-z))
5. El resultado está entre -1 y 1

**La fórmula matemática:**

```
φ(z) = tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

**Derivada:**

```
φ'(z) = 1 - tanh²(z)
```

**Implementación en Código:**

Archivo: `src/core/activation.py`

```python
class TanhActivation(ActivationFunction):
    def compute(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        # NumPy ya tiene una función optimizada para esto
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return 1.0 - np.tanh(x) ** 2
```

---

### 8. ReLU (Rectified Linear Unit)

**¿Qué hace en palabras simples?**

ReLU es muy simple: si el valor es positivo, lo deja igual; si es negativo, lo convierte a cero. Es como un interruptor que solo deja pasar valores positivos.

**Por qué se usa en este proyecto:**

Es muy popular en redes profundas porque:
- Es muy rápida de calcular (solo una comparación)
- No sufre del problema de gradientes desvanecidos para valores positivos
- Hace que algunas neuronas se "apaguen" (sean cero), lo cual crea esparsidad

**Cómo funciona paso a paso:**

Para un valor de entrada z:
1. Compara z con 0
2. Si z > 0, devuelve z
3. Si z ≤ 0, devuelve 0

**La fórmula matemática:**

```
φ(z) = max(0, z)
```

**Derivada:**

```
φ'(z) = 1 si z > 0
φ'(z) = 0 si z ≤ 0
```

**Implementación en Código:**

Archivo: `src/core/activation.py`

```python
class ReLUActivation(ActivationFunction):
    def compute(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        # max(0, x) devuelve el mayor entre 0 y x
        return np.maximum(0.0, x)
    
    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        # (x > 0) crea True/False, astype(float) lo convierte a 1.0/0.0
        return (x > 0.0).astype(float)
```

---

### 9. Leaky ReLU

**¿Qué hace en palabras simples?**

Leaky ReLU es una variante de ReLU que permite un pequeño gradiente para valores negativos. En lugar de convertir negativos a cero, los multiplica por un número muy pequeño (típicamente 0.01).

**Por qué se usa en este proyecto:**

Previene el problema de "neuronas muertas" en ReLU (neuronas que nunca se activan y nunca se recuperan). Al permitir un pequeño flujo de gradiente para valores negativos, las neuronas pueden recuperarse.

**Cómo funciona paso a paso:**

Para un valor de entrada z:
1. Multiplica z por un pequeño valor α (típicamente 0.01)
2. Compara z con α × z
3. Devuelve el mayor de los dos

**La fórmula matemática:**

```
φ(z) = max(αz, z)
```

Donde α es típicamente 0.01.

**Derivada:**

```
φ'(z) = 1 si z > 0
φ'(z) = α si z ≤ 0
```

**Implementación en Código:**

Archivo: `src/core/activation.py`

```python
class LeakyReLUActivation(ActivationFunction):
    def compute(self, x: np.ndarray, sigma: float = 1.0, alpha: float = 0.01) -> np.ndarray:
        # max(alpha * x, x) permite un pequeño gradiente para negativos
        return np.maximum(alpha * x, x)
    
    def derivative(self, x: np.ndarray, sigma: float = 1.0, alpha: float = 0.01) -> np.ndarray:
        # Si x > 0, derivada es 1; si x ≤ 0, derivada es alpha
        return np.where(x > 0.0, 1.0, alpha)
```

---

### 10. Función Lineal (linear / purelin)

**¿Qué hace en palabras simples?**

La función lineal es la más simple de todas: devuelve exactamente el mismo valor que recibe. No hace ninguna transformación.

**Por qué se usa en este proyecto:**

Se usa típicamente en la capa de salida para regresión porque:
- Permite que la red prediga cualquier valor real (sin limitación de rango)
- Su derivada es siempre 1, lo cual simplifica mucho el cálculo de gradientes

**Cómo funciona paso a paso:**

Para un valor de entrada z:
1. Devuelve z sin cambios

**La fórmula matemática:**

```
φ(z) = z
```

**Derivada:**

```
φ'(z) = 1
```

**Implementación en Código:**

Archivo: `src/core/activation.py`

```python
class LinearActivation(ActivationFunction):
    def compute(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return x  # Sin cambios
    
    def derivative(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return np.ones_like(x)  # Siempre 1
```

---

### 11. Función Gaussiana (RBF)

**¿Qué hace en palabras simples?**

La función gaussiana toma una distancia y la convierte en un valor entre 0 y 1 usando una forma de campana. Distancias pequeñas dan valores cercanos a 1, distancias grandes dan valores cercanos a 0.

**Por qué se usa en este proyecto:**

Es la función más común en redes RBF porque:
- Es suave y diferenciable en todas partes
- Decae exponencialmente con la distancia (puntos lejanos tienen poca influencia)
- Tiene "soporte local" (solo afecta puntos cercanos al centro)

**Cómo funciona paso a paso:**

Para una distancia r y un parámetro de ancho σ:
1. Divide la distancia entre el ancho: r / σ
2. Eleva al cuadrado: (r/σ)²
3. Cambia el signo: -(r/σ)²
4. Calcula la exponencial: exp(-(r/σ)²)
5. El resultado está entre 0 y 1

**La fórmula matemática:**

```
φ(r, σ) = exp(-(r/σ)²)
```

**Implementación en Código:**

Archivo: `src/core/activation.py`

```python
class GaussianActivation(ActivationFunction):
    def compute(self, distances: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        # Paso 1: Normalizar distancias por sigma
        # Paso 2: Elevar al cuadrado
        # Paso 3: Negar
        # Paso 4: Calcular exponencial
        return np.exp(-(distances / sigma) ** 2)
```

---

## Distancias y Métricas

### 12. Distancia Euclidiana

**¿Qué hace en palabras simples?**

La distancia euclidiana es la distancia en línea recta entre dos puntos en el espacio. Es como medir con una regla la distancia directa entre dos ubicaciones.

**Por qué se usa en este proyecto:**

En redes RBF, se usa para calcular qué tan lejos está cada punto de datos de cada centro RBF. Esta distancia se usa luego en las funciones de activación RBF.

**Cómo funciona paso a paso:**

Para dos puntos x₁ y x₂ con n dimensiones:
1. Calcula la diferencia en cada dimensión: x₁[i] - x₂[i]
2. Eleva cada diferencia al cuadrado
3. Suma todos los cuadrados
4. Toma la raíz cuadrada de la suma

**La fórmula matemática:**

```
d(x₁, x₂) = √(Σ(i=1 to n) (x₁[i] - x₂[i])²)
```

**Implementación en Código:**

Archivo: `src/core/distance.py`

```python
def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    # Paso 1: Calcular diferencias
    # Paso 2: Elevar al cuadrado
    # Paso 3: Sumar
    # Paso 4: Raíz cuadrada
    return np.sqrt(np.sum((x1 - x2) ** 2))
```

---

### 13. Raíz Cuadrada (np.sqrt)

**¿Qué hace en palabras simples?**

La raíz cuadrada de un número x es el número que, multiplicado por sí mismo, da x. Por ejemplo, la raíz cuadrada de 9 es 3 porque 3 × 3 = 9.

**Por qué se usa en este proyecto:**

Se usa en:
- Cálculo de distancias euclidianas (último paso)
- Cálculo de RMSE (para convertir MSE de vuelta a las unidades originales)
- Funciones de activación RBF multicuadráticas

**Cómo funciona paso a paso:**

NumPy usa algoritmos numéricos eficientes para calcular la raíz cuadrada.

**La fórmula matemática:**

```
√x = y tal que y² = x, con y ≥ 0
```

**Implementación en Código:**

Archivo: `src/evaluation/metrics.py`

```python
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse_val = mse(y_true, y_pred)
    return np.sqrt(mse_val)  # Convierte MSE a RMSE
```

---

### 14. Exponencial (np.exp)

**¿Qué hace en palabras simples?**

La exponencial e^x crece muy rápido. e es aproximadamente 2.718. e^1 ≈ 2.718, e^2 ≈ 7.389, e^10 ≈ 22026. Para valores negativos, e^(-x) se vuelve muy pequeño (casi cero).

**Por qué se usa en este proyecto:**

Es fundamental en:
- Función sigmoide: 1/(1+e^(-x)) - la parte e^(-x) hace que sigmoide funcione
- Función gaussiana: e^(-x) - crea la forma de campana

**Cómo funciona paso a paso:**

NumPy calcula la exponencial usando la serie de Taylor o métodos numéricos optimizados.

**La fórmula matemática:**

```
e^x = Σ(n=0 to ∞) x^n / n!
```

**Implementación en Código:**

Archivo: `src/core/activation.py`

```python
# En sigmoide
return 1.0 / (1.0 + np.exp(-x))  # e^(-x) hace que valores grandes negativos den casi 0

# En gaussiana
return np.exp(-(distances / sigma) ** 2)  # e^(-x) crea decaimiento exponencial
```

---

## Optimización y Gradientes

### 15. Descenso de Gradiente

**¿Qué hace en palabras simples?**

Imagina que estás en una montaña y quieres bajar al punto más bajo. El descenso de gradiente es como mirar a tu alrededor, ver en qué dirección la pendiente es más pronunciada hacia abajo, y dar un paso en esa dirección. Repites esto hasta llegar al fondo.

**Por qué se usa en este proyecto:**

Es el algoritmo principal para entrenar redes de retropropagación. Permite ajustar los pesos de la red para minimizar el error de predicción.

**Cómo funciona paso a paso:**

1. Calcula el gradiente del error con respecto a los pesos (la dirección y magnitud del cambio)
2. Multiplica el gradiente por la tasa de aprendizaje (α) - esto controla qué tan grande es el paso
3. Resta ese valor de los pesos actuales
4. Repite hasta que el error sea pequeño o se alcance el número máximo de iteraciones

**La fórmula matemática:**

```
θ(t+1) = θ(t) - α × ∇L(θ(t))
```

Donde:
- θ: Parámetros del modelo (pesos y bias)
- α: Tasa de aprendizaje (qué tan grandes son los pasos)
- ∇L: Gradiente de la función de pérdida (la dirección del cambio)
- t: Iteración actual

**Implementación en Código:**

Archivo: `src/models/backprop/layer.py`

```python
def update_weights(self, learning_rate: float) -> None:
    # Paso 1: Multiplicar gradiente por learning rate
    # Paso 2: Restar de los pesos actuales
    self.weights -= learning_rate * self.weights_gradient  # W(t+1) = W(t) - α∇W
    
    if self.use_bias and self.bias_gradient is not None:
        self.bias -= learning_rate * self.bias_gradient  # b(t+1) = b(t) - α∇b
```

---

### 16. Retropropagación (Backpropagation)

**¿Qué hace en palabras simples?**

La retropropagación es como propagar el error hacia atrás a través de la red. Si la red cometió un error en la salida, la retropropagación calcula cuánto contribuyó cada neurona y cada peso a ese error, para poder ajustarlos adecuadamente.

**Por qué se usa en este proyecto:**

Permite calcular gradientes eficientemente usando la regla de la cadena. Sin retropropagación, tendríamos que calcular gradientes de manera muy ineficiente.

**Cómo funciona paso a paso:**

1. Calcula el error en la salida (diferencia entre predicción y valor real)
2. Propaga ese error hacia atrás capa por capa
3. Para cada capa:
   - Multiplica el error por la derivada de la función de activación
   - Multiplica por los pesos de la capa siguiente
   - Esto da el error que se debe propagar a la capa anterior
4. Calcula los gradientes de pesos y bias usando estos errores

**La fórmula matemática:**

Para una capa l:

```
δ_l = δ_(l+1) @ W_(l+1)^T × φ'_l(z_l)
```

Gradientes:

```
∇W_l = a_(l-1)^T @ δ_l
∇b_l = Σ(δ_l)
```

**Implementación en Código:**

Archivo: `src/models/backprop/layer.py`

```python
def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    # Paso 1: Calcular derivada de activación
    activation_grad = self.activation.derivative(self.last_z)
    
    # Paso 2: Multiplicar error por derivada
    delta = output_gradient * activation_grad  # δ = gradient_y × φ'(z)
    
    # Paso 3: Calcular gradiente de pesos
    self.weights_gradient = self.last_input.T @ delta  # ∇W = X^T @ δ
    
    # Paso 4: Calcular gradiente de bias
    if self.use_bias:
        self.bias_gradient = np.sum(delta, axis=0)  # ∇b = Σ(δ)
    
    # Paso 5: Calcular gradiente para propagar a capa anterior
    input_gradient = delta @ self.weights.T  # ∇X = δ @ W^T
    
    return input_gradient
```

---

## Métricas de Evaluación

### 17. Error Cuadrático Medio (MSE)

**¿Qué hace en palabras simples?**

El MSE calcula el promedio de los errores al cuadrado. Primero calcula la diferencia entre cada predicción y el valor real, la eleva al cuadrado (para que los errores positivos y negativos no se cancelen), y luego promedia todos esos cuadrados.

**Por qué se usa en este proyecto:**

Es la función de pérdida más común para regresión. Penaliza errores grandes más fuertemente (porque se elevan al cuadrado), lo cual ayuda a que el modelo se enfoque en corregir los errores más grandes.

**Cómo funciona paso a paso:**

1. Calcula la diferencia: y_real - y_predicho
2. Eleva cada diferencia al cuadrado
3. Suma todos los cuadrados
4. Divide por el número de muestras

**La fórmula matemática:**

```
MSE = (1/n) × Σ(i=1 to n) (y[i] - ŷ[i])²
```

**Implementación en Código:**

Archivo: `src/evaluation/metrics.py`

```python
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Paso 1: Calcular diferencias
    # Paso 2: Elevar al cuadrado
    # Paso 3: Promediar
    return np.mean((y_true - y_pred) ** 2)
```

---

### 18. Raíz del Error Cuadrático Medio (RMSE)

**¿Qué hace en palabras simples?**

El RMSE es simplemente la raíz cuadrada del MSE. La diferencia importante es que RMSE está en las mismas unidades que los datos originales, mientras que MSE está en unidades al cuadrado.

**Por qué se usa en este proyecto:**

Es más fácil de interpretar que MSE porque está en las mismas unidades. Por ejemplo, si predices precios de casas en dólares, RMSE también está en dólares, mientras que MSE está en dólares al cuadrado.

**Cómo funciona paso a paso:**

1. Calcula el MSE
2. Toma la raíz cuadrada

**La fórmula matemática:**

```
RMSE = √(MSE) = √((1/n) × Σ(i=1 to n) (y[i] - ŷ[i])²)
```

**Implementación en Código:**

Archivo: `src/evaluation/metrics.py`

```python
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse_val = mse(y_true, y_pred)
    return np.sqrt(mse_val)  # Convierte a las unidades originales
```

---

### 19. Coeficiente R² (R-squared)

**¿Qué hace en palabras simples?**

R² mide qué proporción de la variación en los datos puede explicar el modelo. R² = 1 significa que el modelo explica perfectamente todos los datos. R² = 0 significa que el modelo no es mejor que simplemente predecir el promedio. R² negativo significa que el modelo es peor que predecir el promedio.

**Por qué se usa en este proyecto:**

Es útil para entender qué tan bien está funcionando el modelo en términos relativos. Un R² de 0.9 significa que el modelo explica el 90% de la variación en los datos.

**Cómo funciona paso a paso:**

1. Calcula la suma de los errores al cuadrado (SS_res)
2. Calcula la suma total de cuadrados (SS_tot) - cuánto varían los datos del promedio
3. Divide SS_res entre SS_tot
4. Resta ese resultado de 1

**La fórmula matemática:**

```
R² = 1 - (Σ(y - ŷ)² / Σ(y - ȳ)²)
```

Donde ȳ es el promedio de y.

**Implementación en Código:**

Archivo: `src/evaluation/metrics.py`

```python
def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Paso 1: Calcular suma de residuos al cuadrado
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Paso 2: Calcular suma total de cuadrados
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Paso 3: Calcular R²
    return 1 - (ss_res / ss_tot)
```

---

## Referencias

Para más detalles sobre la implementación y uso de estas operaciones, consulta:

- **Guía Principal**: `docs/guide.md` - Explicación detallada de arquitectura y flujo
- **API Reference**: `api/reference.md` - Documentación de la API de usuario
- **Código Fuente**: `src/` - Implementación de todas las operaciones
