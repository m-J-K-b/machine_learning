from abc import ABC, abstractmethod
from typing import Optional

from numpy.typing import NDArray


class Layer(ABC):
    def __init__(self):
        self._x: Optional[NDArray] = None
        self._y: Optional[NDArray] = None

    def _check_cached_inputs(self) -> None:
        assert (
            self._x is not None and self._y is not None
        ), f"{self.__class__.__name__}: forward(training=True) must be called before backward()."

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
    def backward(self, grad: NDArray, lr: float) -> NDArray:
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
