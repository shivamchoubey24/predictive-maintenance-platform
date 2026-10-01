"""Numpy-only autoencoder inference (keeps the serving layer free of torch)."""
import numpy as np


def ae_forward(weights, X: np.ndarray) -> np.ndarray:
    """Numpy forward pass of the autoencoder (ReLU between layers, linear output)."""
    h = X.astype(np.float64)
    for i, (W, b) in enumerate(weights):
        h = h @ W.T + b
        if i < len(weights) - 1:
            h = np.maximum(h, 0)
    return h


def ae_error(weights, X: np.ndarray) -> np.ndarray:
    return np.mean((ae_forward(weights, X) - X) ** 2, axis=1)
