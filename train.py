"""
Main entry point for neural network training using the compact API.
This script demonstrates the MATLAB-style interface for both RBF and backpropagation networks.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api import NeuralNetwork, ModelType


def main():
    """Main function to demonstrate the compact neural network API."""
    print("Neural Network API Demo (MATLAB-style)")
    print("=" * 60)
    
    # Generate synthetic data (sine wave)
    np.random.seed(42)
    X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    y = np.sin(X).flatten() + 0.1 * np.random.randn(100)
    y = y.reshape(-1, 1)
    
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} feature(s)")
    
    # Split into train and test
    n_train = int(0.8 * X.shape[0])
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print()
    
    # Example 1: RBF Network
    print("Example 1: RBF Network")
    print("-" * 60)
    
    net_rbf = NeuralNetwork(
        model_type=ModelType.RBF,
        n_centers=20,
        sigma=0.5,
        activation_rbf='gaussian',
        regularization=0.01
    )
    
    training_log = net_rbf.train(X_train, y_train, verbose=True)
    predictions_rbf = net_rbf.predict(X_test)
    metrics_rbf = net_rbf.evaluate(X_test, y_test)
    
    print("RBF Test Results:")
    print(f"MSE:  {metrics_rbf['mse']:.6f}")
    print(f"MAE:  {metrics_rbf['mae']:.6f}")
    print(f"RMSE: {metrics_rbf['rmse']:.6f}")
    print(f"R2:   {metrics_rbf['r2']:.6f}")
    print()
    
    # Example 2: Backpropagation Network
    print("Example 2: Backpropagation Network")
    print("-" * 60)
    
    net_bp = NeuralNetwork(
        model_type=ModelType.BACKPROP,
        hidden_layers=[20, 10],
        learning_rate=0.01,
        epochs=1000,
        activation_backprop='sigmoid'
    )
    
    training_log_bp = net_bp.train(X_train, y_train, verbose=True)
    predictions_bp = net_bp.predict(X_test)
    metrics_bp = net_bp.evaluate(X_test, y_test)
    
    print("Backprop Test Results:")
    print(f"MSE:  {metrics_bp['mse']:.6f}")
    print(f"MAE:  {metrics_bp['mae']:.6f}")
    print(f"RMSE: {metrics_bp['rmse']:.6f}")
    print(f"R2:   {metrics_bp['r2']:.6f}")
    print()
    
    # Example 3: Compact usage with different parameters
    print("Example 3: Compact usage with different activation")
    print("-" * 60)
    
    net_rbf2 = NeuralNetwork(
        model_type=ModelType.RBF,
        n_centers=15,
        sigma=0.8,
        activation_rbf='multiquadratic'
    )
    
    net_rbf2.train(X_train, y_train)
    predictions_rbf2 = net_rbf2.predict(X_test)
    
    print("RBF with Multiquadratic activation trained successfully")
    print()
    
    # Example 4: Get model summary and weights
    print("Example 4: Model information")
    print("-" * 60)
    
    summary = net_rbf.summary()
    print("RBF Network Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    weights = net_rbf.get_weights()
    print(f"  Weights shape: {weights['weights'].shape}")
    print(f"  Centers shape: {weights['centers'].shape}")
    print()
    
    print("Demo completed successfully!")
    print()
    print("Compact API Usage:")
    print("  - net = NeuralNetwork(model_type=ModelType.RBF, n_centers=20)")
    print("  - net.train(X_train, y_train, verbose=True)")
    print("  - predictions = net.predict(X_test)")
    print("  - metrics = net.evaluate(X_test, y_test)")
    print()
    print("For more examples, see api_examples/ directory")


if __name__ == "__main__":
    main()
