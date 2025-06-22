import numpy as np
from numpy.typing import NDArray

from nn_core.layers.layer import Layer


class DenseLayer(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initialize a Dense (fully connected) layer.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features (neurons in this layer).
        """
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(
            1.0 / input_size
        )

        self.bias = np.zeros((output_size))

    def forward(self, x: NDArray, training: bool) -> NDArray:
        """
        Perform forward propagation through the layer.

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
        """
        Perform backward propagation and update the layer's weights and biases.

        Args:
            grad (NDArray): Gradient of the loss with respect to the output of this layer,
                               shape (batch_size, output_size).
            lr (float): Learning rate for weight and bias updates.

        Returns:
            NDArray: Gradient of the loss with respect to the input of this layer,
                        shape (batch_size, input_size).
        """
        input_grad = (
            grad @ self.weights
        )  # (batch_size, outputs) @ (outputs, inputs) -> (batch_size, inputs)
        wg = (
            self._x.T @ grad
        )  # (inputs, batch_size) @ (batch_size, outputs) -> (inputs, outputs)

        self.weights -= wg.T * lr  # Transpose to match (outputs, inputs)
        self.bias -= np.sum(grad, axis=0) * lr  # Sum over batch, shape (outputs,)

        return input_grad
