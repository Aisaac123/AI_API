"""
API compacta de redes neuronales.
Proporciona una interfaz simple y unificada para entrenar y simular redes neuronales.
"""

import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Union
from .core.model_type import ModelType
from .config import NeuralNetworkConfig
from .validators import InputValidator
from .factories import ModelFactory, TrainerFactory
from .factories_v2 import RBFModelFactory, BackpropModelFactory
from .core.registry import ModelRegistry
from .core.results import TrainingResult, EvaluationResult, LayerWeights, ModelSummary, ConfusionMatrixResult
from src.evaluation import Evaluator


class NeuralNetwork:
    """
    Clase compacta de red neuronal con interfaz estilo MATLAB.
    
    Esta clase proporciona una interfaz unificada para redes RBF y backpropagation
    con parametrización extensiva y capacidades de registro de entrenamiento.
    
    Uso:
        net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20, sigma=0.5)
        training_log = net.train(X_train, y_train, verbose=True, log_gradients=True)
        predictions = net.predict(X_test)
        metrics = net.evaluate(X_test, y_test)
    """
    
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
            activation_rbf: Función de activación RBF ('gaussian', 'multiquadratic', 'inverse_multiquadratic', 'thin_plate')
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
        
        # Registros de entrenamiento
        self.training_log = {
            'error_history': [],
            'gradient_norms': [],
            'epoch_times': [],
            'performance_metrics': []
        }
    
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
        
        Soporta múltiples salidas generando una matriz por columna.
        
        Args:
            y_true: Valores verdaderos de forma (n_samples,) o (n_samples, n_outputs)
            y_pred: Valores predichos opcionales de forma (n_samples,) o (n_samples, n_outputs)
            X: Datos de entrada opcionales para generar predicciones si y_pred no se proporciona
            
        Returns:
            Si y_true.shape[1] == 1: ConfusionMatrixResult único
            Si y_true.shape[1] > 1: Dict[int, ConfusionMatrixResult] (una por columna)
            
        Raises:
            ValueError: Si se proporcionan tanto y_pred como X, o ninguno
            RuntimeError: Si el modelo no está entrenado y se requiere predicción
        """
        # Validar que no se proporcionen ambos y_pred y X
        if y_pred is not None and X is not None:
            raise ValueError("Proporciona solo y_pred o X, no ambos.")
        
        # Si no se proporciona y_pred, hacer predicciones
        if y_pred is None:
            if X is None:
                raise ValueError("Debes proporcionar y_pred o X para calcular la matriz de confusión.")
            self._ensure_fitted()
            y_pred = self.predict(X)
        
        # Validar y convertir a arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Usar el evaluador para calcular la matriz de confusión
        return self.evaluator.confusion_matrix(y_true, y_pred)
    
    def get_weights(self) -> Dict[str, Any]:
        """
        Obtener los pesos del modelo.
        
        Returns:
            Diccionario que contiene los pesos del modelo
        """
        self._ensure_fitted()
        
        if self.model_type == ModelType.RBF:
            return {
                'weights': self.model.weights,
                'bias': self.model.bias,
                'centers': self.model.centers
            }
        else:
            weights = {}
            for i, layer in enumerate(self.model.layers):
                weights[f'layer_{i}'] = {
                    'weights': layer.weights,
                    'bias': layer.bias
                }
            return weights
    
    def get_layer_weights(self, layer_index: int) -> LayerWeights:
        """
        Obtener pesos y bias de una capa específica (solo para backpropagation).

        Args:
            layer_index: Índice de la capa (0 para primera capa oculta, -1 para capa de salida)

        Returns:
            LayerWeights con pesos, bias, y función de activación de la capa

        Raises:
            RuntimeError: Si el modelo no está entrenado
            IndexError: Si el índice de capa es inválido
            ValueError: Si el modelo es RBF (no tiene capas múltiples)
        """
        self._ensure_fitted()

        if self.model_type == ModelType.RBF:
            raise ValueError("get_layer_weights no está disponible para modelos RBF. Usa get_weights() en su lugar.")

        if not self.model.layers:
            raise RuntimeError("El modelo no tiene capas configuradas.")

        # Manejar índices negativos (como -1 para última capa)
        if layer_index < 0:
            layer_index = len(self.model.layers) + layer_index

        if layer_index < 0 or layer_index >= len(self.model.layers):
            raise IndexError(
                f"Índice de capa inválido: {layer_index}. "
                f"El modelo tiene {len(self.model.layers)} capas (índices 0-{len(self.model.layers)-1})"
            )

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
        Obtener información detallada de todas las capas del modelo.
        
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
                'weights_shape': self.model.weights.shape if self.model.weights is not None else None,
                'bias': self.model.bias,
                'centers_shape': self.model.centers.shape if self.model.centers is not None else None
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
    
    def summary(self) -> ModelSummary:
        """
        Obtener un resumen del modelo.

        Returns:
            ModelSummary con información del modelo
        """
        if self.model is None:
            return ModelSummary(
                model_type=self.model_type.value,
                is_fitted=False,
                configuration=self._get_config()
            )

        model_summary = self.model.summary()
        return ModelSummary(
            model_type=self.model_type.value,
            is_fitted=True,
            configuration=self._get_config(),
            architecture=model_summary.get('architecture'),
            n_parameters=model_summary.get('n_parameters')
        )
    
    def save(self, filepath: str) -> None:
        """
        Guardar el modelo entrenado en disco.

        Args:
            filepath: Ruta donde guardar el modelo (ej: 'model.pkl')

        Raises:
            RuntimeError: Si el modelo no ha sido entrenado
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
    
    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Establecer semilla aleatoria para reproducibilidad.

        Controla numpy.random para asegurar resultados reproducibles.

        Args:
            seed: Semilla aleatoria
        """
        np.random.seed(seed)
    
    def _get_config(self) -> Dict[str, Any]:
        """Obtener la configuración actual como diccionario."""
        config_dict = self.config.to_dict()
        config_dict['model_type'] = self.model_type.value
        return config_dict
    
    def _setup_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Configurar modelo y entrenador según tipo.

        Usa el sistema de registro dinámico para crear modelos y entrenadores.
        Esto permite agregar nuevos modelos sin modificar este código.
        """
        # Convertir ModelType enum a string
        model_type_str = self.model_type.value
        
        # Obtener factory del registro
        factory = ModelRegistry.get_factory(model_type_str)
        
        # Crear modelo y entrenador usando la factory
        self.model = factory.create_network(X, y, self.config)
        self.trainer = factory.create_trainer(self.config)
    
    def _ensure_fitted(self) -> None:
        """Verificar que el modelo ha sido entrenado."""
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado aún. Llame a train() primero.")
