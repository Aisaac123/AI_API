"""
Tests para el REPL de redes neuronales.
Verifica que el contexto del REPL se cargue correctamente y que todos los imports funcionen.
"""

import sys
import os

# Agregar directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_repl_context_loading():
    """Verificar que el contexto del REPL se cargue correctamente."""
    # Ejecutar el script del REPL
    repl_script = os.path.join(project_root, 'repl', 'neural_repl.py')
    
    # Crear un diccionario para el contexto
    context = {}
    
    # Ejecutar el script
    with open(repl_script, 'r', encoding='utf-8') as f:
        exec(f.read(), context)
    
    # Verificar que los componentes principales estén cargados
    required_imports = [
        'np',
        'NeuralNetwork',
        'ModelType',
        'NeuralNetworkConfig',
        'RBFNetwork',
        'RBFConfig',
        'BackpropNetwork',
        'BackpropConfig',
        'GaussianActivation',
        'MultiquadraticActivation',
        'InverseMultiquadraticActivation',
        'ThinPlateSplineActivation',
        'RBFTrainer',
        'BackpropTrainer',
        'KMeansInitializer',
        'RandomInitializer',
        'Evaluator',
        'metrics',
        'NotFittedError',
        'InvalidConfigError',
        'InvalidInputError',
        'ConvergenceError',
        'euclidean_distance',
        'euclidean_distance_matrix',
    ]
    
    for import_name in required_imports:
        assert import_name in context, f"{import_name} no está cargado en el contexto del REPL"
        assert context[import_name] is not None, f"{import_name} es None en el contexto del REPL"
    
    print("✅ Todos los imports requeridos están cargados en el contexto del REPL")


def test_repl_api_functionality():
    """Verificar que la API funcione correctamente en el contexto del REPL."""
    # Ejecutar el script del REPL
    repl_script = os.path.join(project_root, 'repl', 'neural_repl.py')
    
    # Crear un diccionario para el contexto
    context = {}
    
    # Ejecutar el script
    with open(repl_script, 'r', encoding='utf-8') as f:
        exec(f.read(), context)
    
    # Crear datos de prueba
    X = context['np'].random.randn(50, 2)
    y = context['np'].random.randn(50, 1)
    
    # Crear red RBF
    net = context['NeuralNetwork'](model_type=context['ModelType'].RBF, n_centers=10)
    
    # Entrenar
    net.train(X, y)
    
    # Predecir
    predictions = net.predict(X)
    
    # Evaluar
    metrics = net.evaluate(X, y)
    
    # Verificar que las predicciones tengan la forma correcta
    assert predictions.shape == (50, 1), f"Forma incorrecta: {predictions.shape}"
    
    # Verificar que las métricas estén presentes
    assert 'mse' in metrics, "MSE no está en las métricas"
    assert 'mae' in metrics, "MAE no está en las métricas"
    assert 'rmse' in metrics, "RMSE no está en las métricas"
    assert 'r2' in metrics, "R2 no está en las métricas"
    
    print("✅ La API funciona correctamente en el contexto del REPL")


def test_repl_config_class():
    """Verificar que la clase de configuración funcione correctamente."""
    # Ejecutar el script del REPL
    repl_script = os.path.join(project_root, 'repl', 'neural_repl.py')
    
    # Crear un diccionario para el contexto
    context = {}
    
    # Ejecutar el script
    with open(repl_script, 'r', encoding='utf-8') as f:
        exec(f.read(), context)
    
    # Crear configuración
    config = context['NeuralNetworkConfig'](
        n_centers=20,
        sigma=0.5,
        activation_rbf='gaussian',
        learning_rate=0.01
    )
    
    # Validar configuración
    config.validate()
    
    # Convertir a diccionario
    config_dict = config.to_dict()
    
    # Verificar que el diccionario tenga los campos correctos
    assert 'n_centers' in config_dict
    assert 'sigma' in config_dict
    assert 'activation_rbf' in config_dict
    assert 'learning_rate' in config_dict
    
    print("✅ La clase de configuración funciona correctamente en el contexto del REPL")


def test_repl_activation_functions():
    """Verificar que las funciones de activación funcionen correctamente."""
    # Ejecutar el script del REPL
    repl_script = os.path.join(project_root, 'repl', 'neural_repl.py')
    
    # Crear un diccionario para el contexto
    context = {}
    
    # Ejecutar el script
    with open(repl_script, 'r', encoding='utf-8') as f:
        exec(f.read(), context)
    
    # Crear distancias de prueba
    distances = context['np'].array([0.0, 0.5, 1.0, 2.0])
    
    # Probar cada función de activación
    gaussian = context['GaussianActivation']()
    gaussian_output = gaussian.compute(distances, sigma=1.0)
    assert gaussian_output.shape == distances.shape
    
    multiquadratic = context['MultiquadraticActivation']()
    multiquadratic_output = multiquadratic.compute(distances, sigma=1.0)
    assert multiquadratic_output.shape == distances.shape
    
    print("✅ Las funciones de activación funcionan correctamente en el contexto del REPL")


if __name__ == '__main__':
    print("Ejecutando tests del REPL...")
    print()
    
    test_repl_context_loading()
    test_repl_api_functionality()
    test_repl_config_class()
    test_repl_activation_functions()
    
    print()
    print("Todos los tests del REPL pasaron exitosamente.")
