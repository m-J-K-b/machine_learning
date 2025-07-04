import numpy as np
from numpy.typing import NDArray

from nn_core.layer.layer import Layer
from nn_core.optimizer.optimizer import Optimizer


class DropoutLayer(Layer):
    def __init__(self, drop_prob: float = 0.5):
        super().__init__()

        self.drop_prob = drop_prob
        self.mask = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        if training:
            self.mask = (np.random.rand(*x.shape) > self.drop_prob).astype(float)
            y = x * self.mask
            self._x = x
            self._y = y
            return y
        return x

    def backward(self, grad: NDArray, optimizer: Optimizer) -> NDArray:
        return grad * self.mask  # Pass gradient only where neuron was not dropped
