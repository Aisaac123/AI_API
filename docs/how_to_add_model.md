# Cómo Agregar un Nuevo Modelo al Sistema de Registro

## Resumen del Sistema de Registro

El sistema de registro dinámico permite agregar nuevos modelos sin modificar el código central de la API (`api/neural_network.py`). Para agregar un nuevo modelo, necesitas:
1. `src/models/nuevo_modelo/` - Crear carpeta con network.py, config.py
2. `api/factories_nuevo.py` - Crear factory que implemente `ModelFactory`
3. Una línea de código para registrar el modelo

**Total: 3 archivos a crear + 1 línea de código**

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
from api.base import ModelFactory
from src.models.nuevo_modelo import NuevoModelNetwork, NuevoModelConfig
from src.training.nuevo_trainer import NuevoModelTrainer

class NuevoModelFactory(ModelFactory):
    def create_network(self, X, y, config):
        """Crear instancia del modelo."""
        return NuevoModelNetwork(config)
    
    def create_trainer(self, config):
        """Crear instancia del entrenador."""
        return NuevoModelTrainer()
    
    def get_config_class(self):
        """Obtener clase de configuración."""
        return NuevoModelConfig
```

### Paso 3: Registrar el modelo

Agrega una línea para registrar el modelo. Puedes hacerlo de dos formas:

**Opción A: Registro manual**
```python
from api.registry import ModelRegistry
from api.factories_nuevo import NuevoModelFactory

ModelRegistry.register('nuevo_modelo', NuevoModelFactory())
```

**Opción B: Registro con decorador**
```python
from api.registry import register_model

@register_model('nuevo_modelo')
class NuevoModelFactory(ModelFactory):
    # ... implementación ...
    pass
```

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
