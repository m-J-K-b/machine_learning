from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from nn_core.layers.layer import Layer


class DenseInitializers:
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


class DenseLayer(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_initializer: Optional[Callable[[int, int], NDArray]] = None,
    ) -> None:
        super().__init__()

        if weight_initializer is None:
            weight_initializer = lambda o, i: np.random.randn(o, i) * 0.001

        self.weights: NDArray = weight_initializer(output_size, input_size)
        self.bias: NDArray = np.zeros((output_size))

    def forward(self, x: NDArray, training: bool) -> NDArray:
        """Forward propagation

        Args:
            x (NDArray): Input data of shape (batch_size, input_size).

        Returns:
            NDArray: Output data of shape (batch_size, output_size) after applying the linear transformation.
        """
        y = (
            x @ self.weights.T + self.bias
        )  # (n, inputs) @ (inputs, outputs) + (outputs) = (n, outputs)
        if training:
            self._x = x
            self._y = y
        return y

    def backward(self, grad: NDArray, lr: float) -> NDArray:
        """Backward propagation

        Args:
            grad (NDArray): Gradient of the loss with respect to the output of this layer,
                               shape (batch_size, output_size).
            lr (float): Learning rate for weight and bias updates.

        Returns:
            NDArray: Gradient of the loss with respect to the input of this layer,
                        shape (batch_size, input_size).
        """
        self._check_cached_inputs()

        input_grad = (
            grad @ self.weights
        )  # (batch_size, outputs) @ (outputs, inputs) -> (batch_size, inputs)
        wg = (
            self._x.T @ grad
        )  # (inputs, batch_size) @ (batch_size, outputs) -> (inputs, outputs)

        self.weights -= wg.T * lr  # Transpose to match (outputs, inputs)
        self.bias -= np.sum(grad, axis=0) * lr  # Sum over batch, shape (outputs,)

        return input_grad
