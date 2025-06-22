import numpy as np

from nn_core.layers.layer import Layer


class DropoutLayer(Layer):
    def __init__(self, drop_prob: float = 0.5):
        """
        Dropout Layer.

        Args:
            drop_prob (float): Probability of dropping a neuron (0.0 - 1.0).
        """
        self.drop_prob = drop_prob
        self.mask = None

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        if training:
            self.mask = (np.random.rand(*x.shape) > self.drop_prob).astype(float)
            y = x * self.mask
            self._x = x
            self._y = y
            return y
        else:
            return x

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        """
        Backpropagation for dropout.

        Args:
            grad: Gradient from the next layer.
            lr: Learning rate (unused for dropout).

        Returns:
            Gradient to propagate backward.
        """
        return grad * self.mask  # Pass gradient only where neuron was not dropped
