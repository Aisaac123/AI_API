"""
Basic Backpropagation Network Example using the Compact API.
This demonstrates the simplest way to use the backpropagation network with the MATLAB-style interface.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import NeuralNetwork, ModelType


def main():
    """Basic backpropagation network example."""
    print("Basic Backpropagation Network Example (Compact API)")
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
    
    # Create and train backpropagation network
    net = NeuralNetwork(
        model_type=ModelType.BACKPROP,
        hidden_layers=[20, 10],
        learning_rate=0.01,
        epochs=1000,
        activation_backprop='sigmoid'
    )
    
    # Train
    training_log = net.train(X_train, y_train, verbose=True)
    
    # Predict
    predictions = net.predict(X_test)
    
    # Evaluate
    metrics = net.evaluate(X_test, y_test)
    
    print("Test Results:")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R2:   {metrics['r2']:.6f}")


if __name__ == "__main__":
    main()
