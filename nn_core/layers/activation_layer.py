from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from nn_core.layers.layer import Layer


class Activation:
    @abstractmethod
    def f(self, x: NDArray) -> NDArray:
        """
        Apply the activation function element-wise to the input.

        Args:
            x (NDArray): Input data of shape (batch_size, features).

        Returns:
            NDArray: Output after applying the activation function, same shape as input.
        """
        pass

    @abstractmethod
    def d(self, x: NDArray) -> NDArray:
        """
        Compute the element-wise derivative of the activation function with respect to input.

        Args:
            x (NDArray): Input data of shape (batch_size, features).

        Returns:
            NDArray: Derivative of the activation function, same shape as input.
        """
        pass


class ReLu(Activation):
    def f(self, x: NDArray) -> NDArray:
        return np.maximum(0, x)

    def d(self, x: NDArray) -> NDArray:
        return (x > 0).astype(float)


class LeakyReLu(Activation):
    def __init__(self, alpha=0.01):
        """
        LeakyReLU activation function.

        Args:
            alpha (float): Slope for x < 0. Default is 0.01.
        """
        self.alpha = alpha

    def f(self, x: NDArray) -> NDArray:
        return np.where(x > 0, x, self.alpha * x)

    def d(self, x: NDArray) -> NDArray:
        dx = np.ones_like(x)
        dx[x < 0] = self.alpha
        return dx


class GELU(Activation):
    def f(self, x: NDArray) -> NDArray:
        return (
            0.5
            * x
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        )

    def d(self, x: NDArray) -> NDArray:
        # derivative approximation
        tanh_term = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
        left = 0.5 * (1 + tanh_term)
        sech2 = 1 - tanh_term**2
        right = 0.5 * x * sech2 * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        return left + right


class Sigmoid(Activation):
    def f(self, x: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-x))

    def d(self, x: NDArray) -> NDArray:
        sig = self.f(x)
        return sig * (1 - sig)


class Softmax(Activation):
    def f(self, x: NDArray) -> NDArray:
        exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return softmax

    def d(self, x: NDArray) -> NDArray:
        # Not used directly when combined with CrossEntropyLoss
        return np.ones_like(x)  # Safe dummy derivative


class ActivationLayer(Layer):
    def __init__(self, activation):
        """
        Activation layer applying a given activation function.

        Args:
            activation (Activation): An instance of an Activation subclass (e.g., ReLU, Sigmoid).
        """
        self.activation = activation

    def forward(self, x: NDArray, training: bool) -> NDArray:
        """
        Apply the activation function to the inputs during the forward pass.

        Args:
            x (NDArray): Input data. Shape: (batch_size, features).
            store_input (bool): Whether to store input/output for backward.

        Returns:
            NDArray: Output of activation function, same shape as input.
        """
        y = self.activation.f(x)
        if training:
            self._x = x
            self._y = y
        return y

    def backward(self, grad: NDArray, lr: float) -> NDArray:
        """
        Compute the gradient of the loss with respect to the input of this layer.

        Args:
            grad (NDArray): Gradient from the next layer. Shape: (batch_size, features).
            learning_rate (float): Learning rate (not used in activation layers but kept for interface consistency).

        Returns:
            NDArray: Gradient to pass to the previous layer. Shape: (batch_size, features).
        """
        # Element-wise multiplication: chain rule application
        return grad * self.activation.d(
            self._x
        )  # Shapes match since activation layers don't alter dimensionality
