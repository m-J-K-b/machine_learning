from numpy.typing import NDArray

from nn_core.layer.activation import Activation
from nn_core.layer.layer import Layer
from nn_core.optimizer.optimizer import Optimizer


class ActivationLayer(Layer):
    def __init__(self, activation: Activation) -> None:
        super().__init__()
        self.activation: Activation = activation

    def forward(self, x: NDArray, training: bool) -> NDArray:
        y = self.activation.f(x)
        if training:
            self._x = x
            self._y = y
        return y

    def backward(self, grad: NDArray, optimizer: Optimizer) -> NDArray:
        return grad * self.activation.d(self._x)
