from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from nn_core.layer.layer import Layer
from nn_core.optimizer.optimizer import Optimizer


class DenseLayer(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_initializer: Optional[Callable[[int, int], NDArray]] = None,
    ) -> None:
        super().__init__()

        if weight_initializer is None:
            weight_initializer = DenseLayer.he_normal

        self.weights: NDArray = weight_initializer(output_size, input_size)
        self.bias: NDArray = np.zeros((output_size))

    def forward(self, x: NDArray, training: bool) -> NDArray:
        # (n, inputs) @ (inputs, outputs) + (outputs) = (n, outputs)
        y = x @ self.weights.T + self.bias
        if training:
            self._x = x
            self._y = y
        return y

    def backward(self, grad: NDArray, optimizer: Optimizer) -> NDArray:
        # (batch_size, outputs) @ (outputs, inputs) -> (batch_size, inputs)
        input_grad = grad @ self.weights

        # (inputs, batch_size) @ (batch_size, outputs) -> (inputs, outputs) add transpose to get (outputs, inputs)
        wg = (self._x.T @ grad).T
        bg = np.sum(grad, axis=0)

        self.weights = optimizer.update(self.weights, wg, name=f"{id(self)}_W")
        self.bias = optimizer.update(self.bias, bg, name=f"{id(self)}_b")

        # # momentum to converge faster
        # self.v_w = momentum * self.v_w + (1 - momentum) * wg
        # self.v_b = momentum * self.v_b + (1 - momentum) * bg

        # # adagrad to adjust lr
        # self.w_m += wg**2
        # self.b_m += bg**2
        # wlr = lr / (np.sqrt(self.w_m) + 1e-5)
        # blr = lr / (np.sqrt(self.b_m) + 1e-5)

        # self.weights -= self.v_w * wlr
        # self.bias -= self.v_b * blr

        return input_grad

    @staticmethod
    def random_normal(output_size: int, input_size: int, std: float = 0.01) -> NDArray:
        return np.random.randn(output_size, input_size) * std

    @staticmethod
    def xavier_uniform(output_size: int, input_size: int) -> NDArray:
        limit = np.sqrt(6 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (output_size, input_size))

    @staticmethod
    def he_normal(output_size: int, input_size: int) -> NDArray:
        std = np.sqrt(2 / input_size)
        return np.random.randn(output_size, input_size) * std
