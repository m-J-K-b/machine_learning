from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Layer(ABC):
    def __init__(self):
        self._x = None
        self._y = None

    @abstractmethod
    def forward(self, x: NDArray, training: bool) -> NDArray:
        """
        Forward pass for a single layer.
        Stores the input data in self._x.

        Args:
            x (NDArray): Input data of shape (N, ...) where N is batch size.

        Returns:
            NDArray: Output data after this layer's transformation.

        """
        pass

    @abstractmethod
    def backward(self, grad: NDArray, lr: float) -> NDArray:
        """
        Backward pass for a single layer.

        Args:
            grad (NDArray): Gradient of the loss with respect to the output of this layer. Shape: (N, ...) where N is batch size.
            learning_rate (float): Learning rate to update the layer's parameters (if any).

        Returns:
            NDArray: Gradient of the loss with respect to the input of this layer (shape same as input of forward()).
        """
        pass
