from abc import ABC, abstractmethod
from typing import Optional

from numpy.typing import NDArray

from nn_core.optimizer.optimizer import Optimizer


class Layer(ABC):
    def __init__(self):
        self._x: Optional[NDArray] = None
        self._y: Optional[NDArray] = None

    @abstractmethod
    def forward(self, x: NDArray, training: bool) -> NDArray:
        """Forward propagation
        Stores the input data in self._x if training has been set to True

        Args:
            x (NDArray): Input data of shape (N, F) where N is batch size and F is the number of features.

        Returns:
            NDArray: Output data after this layer's transformation.

        """
        pass

    @abstractmethod
    def backward(self, grad: NDArray, optimizer: Optimizer) -> NDArray:
        """Backward propagation
        Adjusts layer properties based on the output gradient.
        Calculates and passes through the input gradient

        Args:
            grad (NDArray): Gradient of the loss with respect to the output of this layer. Shape: (N, ...) where N is batch size.
            learning_rate (float): Learning rate to update the layer's parameters (if any).

        Returns:
            NDArray: Gradient of the loss with respect to the input of this layer (shape same as input of forward()).
        """
        pass
