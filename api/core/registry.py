"""
Sistema de registro de modelos dinámico.
Permite agregar nuevos modelos sin modificar el código central.
"""

from typing import Dict, Type, Callable, Any
import numpy as np
from .base import ModelFactory


class ModelRegistry:
    """
    Registro centralizado de modelos.
    
    Permite registrar nuevos modelos dinámicamente sin modificar
    el código central de la API.
    """
    
    _factories: Dict[str, ModelFactory] = {}
    
    @classmethod
    def register(cls, model_type: str, factory: ModelFactory) -> None:
        """
        Registrar un nuevo modelo.
        
        Args:
            model_type: Identificador único del modelo (ej: 'rbf', 'backprop', 'lstm')
            factory: Factory del modelo que implementa ModelFactory
        """
        cls._factories[model_type] = factory
    
    @classmethod
    def get_factory(cls, model_type: str) -> ModelFactory:
        """
        Obtener la factory de un modelo.
        
        Args:
            model_type: Identificador del modelo
            
        Returns:
            Factory del modelo
            
        Raises:
            ValueError: Si el modelo no está registrado
        """
        if model_type not in cls._factories:
            raise ValueError(
                f"Modelo '{model_type}' no registrado. "
                f"Modelos disponibles: {list(cls._factories.keys())}"
            )
        return cls._factories[model_type]
    
    @classmethod
    def list_models(cls) -> list:
        """
        Listar todos los modelos registrados.
        
        Returns:
            Lista de identificadores de modelos registrados
        """
        return list(cls._factories.keys())
    
    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """
        Verificar si un modelo está registrado.
        
        Args:
            model_type: Identificador del modelo
            
        Returns:
            True si el modelo está registrado, False en caso contrario
        """
        return model_type in cls._factories


def register_model(model_type: str):
    """
    Decorador para registrar un modelo automáticamente.
    
    Uso:
        @register_model('mi_modelo')
        class MiModeloFactory(ModelFactory):
            ...
    """
    def decorator(factory_class: Type[ModelFactory]) -> Type[ModelFactory]:
        ModelRegistry.register(model_type, factory_class())
        return factory_class
    return decorator


def register_default_models():
    """
    Registrar los modelos por defecto (RBF y Backprop).
    
    Esta función se llama después de resolver las importaciones
    para evitar dependencias circulares.
    """
    from ..factories_v2 import RBFModelFactory, BackpropModelFactory
    
    ModelRegistry.register('rbf', RBFModelFactory())
    ModelRegistry.register('backprop', BackpropModelFactory())
