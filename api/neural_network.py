"""
API compacta de redes neuronales.
Proporciona una interfaz simple y unificada para entrenar y simular redes neuronales.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .model_type import ModelType
from .config import NeuralNetworkConfig
from .validators import InputValidator
from .factories import ModelFactory, TrainerFactory
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
    ) -> Dict[str, Any]:
        """
        Entrenar la red neuronal.
        
        Args:
            X: Matriz de entrada de forma (n_samples, n_features)
            y: Matriz de salida objetivo de forma (n_samples, n_outputs)
            verbose: Si imprimir progreso de entrenamiento
            
        Returns:
            Diccionario con resultados de entrenamiento y registros
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
        
        return {
            'training_time': training_time,
            'final_error': result.final_error,
            'epochs': result.epochs,
            'error_history': result.error_history,
            'converged': result.converged,
            'metadata': result.metadata
        }
    
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
    ) -> Dict[str, Any]:
        """
        Evaluar el modelo en datos de prueba.
        
        Args:
            X: Matriz de entrada de prueba
            y: Valores objetivo de prueba
            detailed: Si devolver métricas detalladas
            
        Returns:
            Diccionario con métricas de evaluación
        """
        self._ensure_fitted()
        X, y = InputValidator.validate_input_pair(X, y)
        
        report = self.evaluator.evaluate(self.model, X, y)
        
        result = {
            'mse': report.mse,
            'mae': report.mae,
            'rmse': report.rmse,
            'r2': report.r2,
            'accuracy': report.accuracy
        }
        
        if detailed:
            result['predictions'] = report.predictions
            result['metadata'] = report.metadata
        
        return result
    
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
    
    def get_layer_weights(self, layer_index: int) -> Dict[str, Any]:
        """
        Obtener pesos y bias de una capa específica (solo para backpropagation).
        
        Args:
            layer_index: Índice de la capa (0 para primera capa oculta, -1 para capa de salida)
            
        Returns:
            Diccionario con pesos, bias, y función de activación de la capa
            
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
    
    def summary(self) -> Dict[str, Any]:
        """
        Obtener un resumen del modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        if self.model is None:
            return {
                'model_type': self.model_type.value,
                'is_fitted': False,
                'configuration': self._get_config()
            }
        
        summary = self.model.summary()
        summary['configuration'] = self._get_config()
        return summary
    
    def _get_config(self) -> Dict[str, Any]:
        """Obtener la configuración actual como diccionario."""
        config_dict = self.config.to_dict()
        config_dict['model_type'] = self.model_type.value
        return config_dict
    
    def _setup_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Configurar modelo y entrenador según tipo.
        
        Usa el patrón Factory para crear modelos y entrenadores,
        aplicando el principio de responsabilidad única.
        """
        if self.model_type == ModelType.RBF:
            self.model = ModelFactory.create_rbf_network(X, self.config)
            self.trainer = TrainerFactory.create_rbf_trainer(self.config)
        else:
            self.model = ModelFactory.create_backprop_network(X, y, self.config)
            self.trainer = TrainerFactory.create_backprop_trainer(self.config)
    
    def _ensure_fitted(self) -> None:
        """Verificar que el modelo ha sido entrenado."""
        if self.model is None:
            raise RuntimeError("El modelo no ha sido entrenado aún. Llame a train() primero.")
