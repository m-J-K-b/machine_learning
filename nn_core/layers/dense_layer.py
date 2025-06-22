import numpy as np

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

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation through the layer.

        Args:
            x (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output data of shape (batch_size, output_size) after applying the linear transformation.
        """
        y = (
            x @ self.weights.T + self.bias
        )  # (n, inputs) @ (inputs, outputs) + (outputs) = (n, outputs)
        if training:
            self._x = x
            self._y = y
        return y

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        """
        Perform backward propagation and update the layer's weights and biases.

        Args:
            grad (np.ndarray): Gradient of the loss with respect to the output of this layer,
                               shape (batch_size, output_size).
            lr (float): Learning rate for weight and bias updates.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer,
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

    def to_dict(self):
        return {
            "type": "DenseLayer",
            "weights": self.weights.tolist(),
            "biases": self.bias.tolist(),
        }
