from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Activation(ABC):
    @abstractmethod
    def f(self, x: NDArray) -> NDArray:
        """
        Apply the activation function element-wise.
        """
        pass

    @abstractmethod
    def d(self, x: NDArray) -> NDArray:
        """
        Compute the element-wise derivative of the activation function.
        """
        pass


class ReLU(Activation):
    def f(self, x: NDArray) -> NDArray:
        return np.maximum(0, x)

    def d(self, x: NDArray) -> NDArray:
        return (x > 0).astype(float)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha

    def f(self, x: NDArray) -> NDArray:
        return np.where(x > 0, x, self.alpha * x)

    def d(self, x: NDArray) -> NDArray:
        dx = np.ones_like(x)
        dx[x < 0] = self.alpha
        return dx


class GELU(Activation):
    def f(self, x: NDArray) -> NDArray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def d(self, x: NDArray) -> NDArray:
        tanh_term = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))
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


# class Softmax(Activation): # TODO: Needs Fixing!
#     def f(self, x: NDArray) -> NDArray:
#         exp = np.exp(x - np.max(x, axis=1, keepdims=True))
#         return exp / np.sum(exp, axis=1, keepdims=True)

#     def d(self, x: NDArray) -> NDArray:
#         # Not used directly unless explicitly computing full Jacobian
#         return np.ones_like(x)
