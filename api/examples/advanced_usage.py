"""
Advanced Neural Network Example using the Compact API.
This demonstrates advanced features including different activation functions, parameter tuning, and detailed logging.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import NeuralNetwork, ModelType


def main():
    """Advanced neural network example with multiple configurations."""
    print("Advanced Neural Network Example (Compact API)")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    y = np.sin(X).flatten() + 0.1 * np.random.randn(100)
    y = y.reshape(-1, 1)
    
    # Split data
    n_train = int(0.8 * X.shape[0])
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Example 1: RBF with different activation functions
    print("Example 1: RBF with different activation functions")
    print("-" * 60)
    
    activations = ['gaussian', 'multiquadratic', 'inverse_multiquadratic', 'thin_plate']
    
    for activation in activations:
        net = NeuralNetwork(
            model_type=ModelType.RBF,
            n_centers=20,
            sigma=0.5,
            activation_rbf=activation,
            regularization=0.01
        )
        
        net.train(X_train, y_train, verbose=False)
        metrics = net.evaluate(X_test, y_test)
        
        print(f"{activation:20s} - MSE: {metrics['mse']:.6f}, R2: {metrics['r2']:.6f}")
    
    print()
    
    # Example 2: Backpropagation with different architectures
    print("Example 2: Backpropagation with different architectures")
    print("-" * 60)
    
    architectures = [
        [10],
        [20, 10],
        [30, 20, 10]
    ]
    
    for hidden_layers in architectures:
        net = NeuralNetwork(
            model_type=ModelType.BACKPROP,
            hidden_layers=hidden_layers,
            learning_rate=0.01,
            epochs=500,
            activation_backprop='sigmoid'
        )
        
        net.train(X_train, y_train, verbose=False)
        metrics = net.evaluate(X_test, y_test)
        
        print(f"{str(hidden_layers):20s} - MSE: {metrics['mse']:.6f}, R2: {metrics['r2']:.6f}")
    
    print()
    
    # Example 3: Parameter tuning with RBF
    print("Example 3: Parameter tuning with RBF")
    print("-" * 60)
    
    sigmas = [0.3, 0.5, 0.8, 1.0]
    n_centers_list = [10, 20, 30]
    
    best_mse = float('inf')
    best_config = None
    
    for sigma in sigmas:
        for n_centers in n_centers_list:
            net = NeuralNetwork(
                model_type=ModelType.RBF,
                n_centers=n_centers,
                sigma=sigma,
                activation_rbf='gaussian',
                regularization=0.01
            )
            
            net.train(X_train, y_train, verbose=False)
            metrics = net.evaluate(X_test, y_test)
            
            if metrics['mse'] < best_mse:
                best_mse = metrics['mse']
                best_config = (sigma, n_centers)
            
            print(f"sigma={sigma:.1f}, n_centers={n_centers:2d} - MSE: {metrics['mse']:.6f}")
    
    print(f"\nBest configuration: sigma={best_config[0]}, n_centers={best_config[1]}")
    print(f"Best MSE: {best_mse:.6f}")
    print()
    
    # Example 4: Model inspection
    print("Example 4: Model inspection and weights")
    print("-" * 60)
    
    net = NeuralNetwork(
        model_type=ModelType.RBF,
        n_centers=20,
        sigma=0.5,
        activation_rbf='gaussian'
    )
    
    net.train(X_train, y_train, verbose=True)
    
    # Get summary
    summary = net.summary()
    print("Model Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get weights
    weights = net.get_weights()
    print(f"\nWeights shape: {weights['weights'].shape}")
    print(f"Centers shape: {weights['centers'].shape}")
    print(f"Bias shape: {weights['bias'].shape}")
    
    # Get training log
    print(f"\nTraining time: {net.training_log['training_time']:.4f} seconds")
    print(f"Final error: {net.training_log['error_history'][-1]:.6f}")


if __name__ == "__main__":
    main()
