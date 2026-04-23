# Cómo Agregar un Nuevo Modelo al Sistema de Registro

## Resumen del Sistema de Registro

El sistema de registro dinámico permite agregar nuevos modelos sin modificar el código central de la API (`api/neural_network.py`). Este sistema usa el patrón Factory con un registro centralizado (singleton) que mapea identificadores de strings a factories de modelos.

**Arquitectura del sistema:**
- `api/core/base.py` - Interfaz abstracta `ModelFactory` que todas las factories deben implementar
- `api/core/registry.py` - `ModelRegistry` (singleton) que mantiene el diccionario de factories registradas
- `api/factories_v2.py` - Implementaciones existentes: `RBFModelFactory` y `BackpropModelFactory`

Para agregar un nuevo modelo, necesitas:
1. `src/models/nuevo_modelo/` - Crear carpeta con network.py, config.py
2. `api/factories_nuevo.py` - Crear factory que implemente `ModelFactory`
3. Una línea de código para registrar el modelo

**Total: 3 archivos a crear + 1 línea de código**

## Cómo Funciona el Sistema de Registro

### Flujo Interno

Cuando llamas a `NeuralNetwork.train(X, y)`, el flujo interno es:

```
1. NeuralNetwork.train(X, y)
   ↓
2. _setup_model(X, y)
   a. model_type_str = self.model_type.value  # 'rbf' o 'backprop'
   b. factory = ModelRegistry.get_factory(model_type_str)
      - Busca en el diccionario interno _factories
      - Si no existe, lanza ValueError con modelos disponibles
   c. self.model = factory.create_network(X, y, self.config)
      - La factory crea la instancia del modelo configurado
   d. self.trainer = factory.create_trainer(self.config)
      - La factory crea el entrenador apropiado
   ↓
3. trainer.train(model, X, y)
   - Ejecuta el algoritmo de entrenamiento específico
```

### Interfaz ModelFactory

Todas las factories deben implementar esta interfaz definida en `api/core/base.py`:

```python
class ModelFactory(ABC):
    """Interfaz abstracta para factories de modelos."""
    
    @abstractmethod
    def create_network(self, X: np.ndarray, y: np.ndarray, config: Any):
        """Crear una instancia del modelo."""
        pass
    
    @abstractmethod
    def create_trainer(self, config: Any):
        """Crear una instancia del entrenador."""
        pass
    
    @abstractmethod
    def get_config_class(self) -> type:
        """Obtener la clase de configuración del modelo."""
        pass
```

### ModelRegistry

El registro es un singleton definido en `api/core/registry.py`:

```python
class ModelRegistry:
    """Registro centralizado de modelos."""
    
    _factories: Dict[str, ModelFactory] = {}
    
    @classmethod
    def register(cls, model_type: str, factory: ModelFactory) -> None:
        """Registrar un nuevo modelo."""
        cls._factories[model_type] = factory
    
    @classmethod
    def get_factory(cls, model_type: str) -> ModelFactory:
        """Obtener la factory de un modelo."""
        if model_type not in cls._factories:
            raise ValueError(
                f"Modelo '{model_type}' no registrado. "
                f"Modelos disponibles: {list(cls._factories.keys())}"
            )
        return cls._factories[model_type]
    
    @classmethod
    def list_models(cls) -> list:
        """Listar todos los modelos registrados."""
        return list(cls._factories.keys())
    
    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """Verificar si un modelo está registrado."""
        return model_type in cls._factories
```

## Pasos Detallados

### Paso 1: Crear la estructura del modelo

Crea la carpeta `src/models/nuevo_modelo/` con:
- `__init__.py` - Exports del modelo
- `config.py` - Configuración del modelo (hereda de `BaseConfig`)
- `network.py` - Implementación del modelo (hereda de `BaseModel`)

Ejemplo de `config.py`:
```python
from src.models.config import BaseConfig

class NuevoModelConfig(BaseConfig):
    def __init__(self, param1: float, param2: int, random_state: int = 42):
        self.param1 = param1
        self.param2 = param2
        self.random_state = random_state
    
    def validate(self) -> None:
        """Validar configuración."""
        if self.param1 <= 0:
            raise ValueError("param1 debe ser positivo")
```

### Paso 2: Crear la Factory

Crea `api/factories_nuevo.py` con una clase que implemente la interfaz `ModelFactory`:

```python
from api.core.base import ModelFactory
from src.models.nuevo_modelo import NuevoModelNetwork, NuevoModelConfig
from src.training.nuevo_trainer import NuevoModelTrainer

class NuevoModelFactory(ModelFactory):
    def create_network(self, X, y, config):
        """
        Crear instancia del modelo configurado.
        
        Flujo:
            1. Extraer parámetros de config
            2. Crear NuevoModelConfig específico
            3. Retornar NuevoModelNetwork instanciado
        """
        return NuevoModelNetwork(config)
    
    def create_trainer(self, config):
        """
        Crear instancia del entrenador configurado.
        
        Flujo:
            1. Crear NuevoModelTrainer con parámetros de config
            2. Retornar el entrenador
        """
        return NuevoModelTrainer()
    
    def get_config_class(self):
        """Obtener clase de configuración del modelo."""
        return NuevoModelConfig
```

**Nota importante:** La factory debe importar desde `api.core.base`, no desde `api.base`.

### Paso 3: Registrar el modelo

Agrega una línea para registrar el modelo. Puedes hacerlo de dos formas:

**Opción A: Registro manual**
```python
from api.core.registry import ModelRegistry
from api.factories_nuevo import NuevoModelFactory

ModelRegistry.register('nuevo_modelo', NuevoModelFactory())
```

**Opción B: Registro con decorador**
```python
from api.core.registry import register_model

@register_model('nuevo_modelo')
class NuevoModelFactory(ModelFactory):
    # ... implementación ...
    pass
```

**Nota importante:** Los imports deben ser desde `api.core.registry`, no desde `api.registry`.

## Uso del Nuevo Modelo

Una vez registrado, puedes usar el nuevo modelo de dos formas:

### Forma 1: Usando el enum ModelType (requiere modificar el enum)
```python
from api import NeuralNetwork, ModelType

net = NeuralNetwork(model_type=ModelType.NUEVO_MODELO, param1=1.0, param2=10)
result = net.train(X, y)
```

### Forma 2: Usando el sistema de registro directamente (sin modificar el enum)
```python
from api.registry import ModelRegistry

factory = ModelRegistry.get_factory('nuevo_modelo')
config = NuevoModelConfig(param1=1.0, param2=10)
model = factory.create_network(X, y, config)
trainer = factory.create_trainer(config)
result = trainer.train(model, X, y)
```

## Ventajas del Sistema de Registro

1. **Menos acoplamiento**: No necesitas tocar `api/neural_network.py`
2. **Más fácil de mantener**: El código central no crece con cada modelo nuevo
3. **Modular**: Cada modelo tiene su propio factory y config
4. **Extensible**: Puedes registrar modelos desde cualquier lugar (incluso plugins externos)
5. **Dinámico**: Puedes registrar modelos en tiempo de ejecución

## Verificar que el Modelo Está Registrado

```python
from api.registry import ModelRegistry

# Listar todos los modelos
print(ModelRegistry.list_models())
# ['rbf', 'backprop', 'nuevo_modelo']

# Verificar si un modelo está registrado
print(ModelRegistry.is_registered('nuevo_modelo'))
# True
```

## Ejemplo Completo

Ver el archivo `api/examples/simple_linear.py` para un ejemplo completo de cómo agregar un modelo lineal al sistema de registro.
