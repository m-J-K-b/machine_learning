from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self):
        self._x = None
        self._y = None

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        """
        Forward pass for a single layer.
        Stores the input data in self._x.

        Args:
            x (np.ndarray): Input data of shape (N, ...) where N is batch size.

        Returns:
            np.ndarray: Output data after this layer's transformation.

        """
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        """
        Backward pass for a single layer.

        Args:
            grad (np.ndarray): Gradient of the loss with respect to the output of this layer. Shape: (N, ...) where N is batch size.
            learning_rate (float): Learning rate to update the layer's parameters (if any).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer (shape same as input of forward()).
        """
        pass
