"""
Classification Example using the Compact API.
This demonstrates using both RBF and backpropagation networks for classification tasks.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import NeuralNetwork, ModelType


def generate_classification_data(n_samples=200):
    """
    Generate synthetic classification data.
    
    Creates two circular clusters for binary classification.
    """
    np.random.seed(42)
    
    # Class 0: inner circle
    radius_0 = 0.5 + 0.1 * np.random.randn(n_samples // 2)
    angle_0 = 2 * np.pi * np.random.rand(n_samples // 2)
    X0 = np.column_stack([radius_0 * np.cos(angle_0), radius_0 * np.sin(angle_0)])
    y0 = np.zeros((n_samples // 2, 1))
    
    # Class 1: outer circle
    radius_1 = 1.5 + 0.1 * np.random.randn(n_samples // 2)
    angle_1 = 2 * np.pi * np.random.rand(n_samples // 2)
    X1 = np.column_stack([radius_1 * np.cos(angle_1), radius_1 * np.sin(angle_1)])
    y1 = np.ones((n_samples // 2, 1))
    
    # Combine
    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def main():
    """Classification example."""
    print("Classification Example (Compact API)")
    print("=" * 60)
    
    # Generate data
    X, y = generate_classification_data(n_samples=200)
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y == 0)} class 0, {np.sum(y == 1)} class 1")
    
    # Split data
    n_train = int(0.8 * X.shape[0])
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print()
    
    # Example 1: RBF for classification
    print("Example 1: RBF Network for Classification")
    print("-" * 60)
    
    net_rbf = NeuralNetwork(
        model_type=ModelType.RBF,
        n_centers=30,
        sigma=0.8,
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
    print(f"Accuracy: {metrics_rbf['accuracy']:.4f}")
    print()
    
    # Example 2: Backpropagation for classification
    print("Example 2: Backpropagation Network for Classification")
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
    print(f"Accuracy: {metrics_bp['accuracy']:.4f}")
    print()
    
    # Example 3: Compare different activation functions
    print("Example 3: Compare RBF activation functions for classification")
    print("-" * 60)
    
    activations = ['gaussian', 'multiquadratic', 'inverse_multiquadratic']
    
    for activation in activations:
        net = NeuralNetwork(
            model_type=ModelType.RBF,
            n_centers=30,
            sigma=0.8,
            activation_rbf=activation,
            regularization=0.01
        )
        
        net.train(X_train, y_train, verbose=False)
        metrics = net.evaluate(X_test, y_test)
        
        print(f"{activation:25s} - Accuracy: {metrics['accuracy']:.4f}, R2: {metrics['r2']:.6f}")


if __name__ == "__main__":
    main()
