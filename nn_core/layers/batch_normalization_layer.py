from typing import Optional

import numpy as np
from numpy.typing import NDArray

from nn_core.layers.layer import Layer


class BatchNormalizationLayer(Layer):
    def __init__(self, momentum: float = 0.9, eps: float = 1e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.running_mean: Optional[NDArray] = None
        self.running_var: Optional[NDArray] = None

        # Cache for backward pass
        self._x_normalized: Optional[NDArray] = None
        self._variance: Optional[NDArray] = None
        self._x_centered: Optional[NDArray] = None
        self._inv_std: Optional[NDArray] = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        # Initialize running statistics if needed
        if self.running_mean is None or self.running_var is None:
            D = x.shape[1]
            self.running_mean = np.zeros((1, D))
            self.running_var = np.ones((1, D))

        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)

            # Update running statistics with momentum
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

            # Use batch statistics for normalization
            mean = batch_mean
            variance = batch_var
        else:
            # Use running statistics for normalization
            mean = self.running_mean
            variance = self.running_var

        # Center the input
        x_centered = x - mean

        # Compute safe inverse standard deviation
        # Add eps before sqrt to prevent sqrt(0), and clamp to prevent division by 0
        inv_std = 1.0 / np.sqrt(np.maximum(variance + self.eps, self.eps))

        # Normalize
        x_normalized = x_centered * inv_std

        # Cache values for backward pass (only in training mode)
        if training:
            self._x_normalized = x_normalized
            self._variance = variance
            self._x_centered = x_centered
            self._inv_std = inv_std

        return x_normalized

    def backward(self, grad: NDArray, lr: float) -> NDArray:
        if (
            self._x_normalized is None
            or self._variance is None
            or self._x_centered is None
            or self._inv_std is None
        ):
            raise ValueError(
                "Cached inputs not found. Make sure to perform forward pass in training mode before backward."
            )

        N, D = grad.shape

        # Get cached values
        x_normalized = self._x_normalized
        x_centered = self._x_centered
        inv_std = self._inv_std

        # Compute gradients using numerically stable formulation
        # This follows the standard batch norm backward pass derivation

        # Gradient w.r.t. normalized input
        dx_normalized = grad

        # Gradient w.r.t. variance
        # Safe computation to prevent overflow in inv_std^3
        inv_std_safe = np.minimum(inv_std, 1e6)  # Clamp to prevent overflow
        dvariance = np.sum(
            dx_normalized * x_centered * (-0.5) * inv_std_safe**3, axis=0, keepdims=True
        )

        # Gradient w.r.t. mean
        dmean = (
            np.sum(dx_normalized * (-inv_std), axis=0, keepdims=True)
            + dvariance * np.sum(-2.0 * x_centered, axis=0, keepdims=True) / N
        )

        # Gradient w.r.t. input
        dx = dx_normalized * inv_std + dvariance * 2.0 * x_centered / N + dmean / N

        return dx

    def reset_running_stats(self) -> None:
        """Reset running statistics to initial values."""
        if self.running_mean is not None:
            self.running_mean.fill(0.0)
        if self.running_var is not None:
            self.running_var.fill(1.0)

    def clear_cache(self) -> None:
        """Clear cached values from forward pass."""
        self._x_normalized = None
        self._variance = None
        self._x_centered = None
        self._inv_std = None
