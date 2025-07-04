from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from nn_core.util import softmax


class Loss:
    @abstractmethod
    def f(self, y: NDArray, y_pred: NDArray) -> float:
        """
        Compute the scalar loss value over the batch.

        Args:
            y (NDArray): Ground truth targets. Shape: (batch_size, features).
            y_pred (NDArray): Predictions. Shape: (batch_size, features).

        Returns:
            float: Scalar loss value averaged over the batch and features.
        """
        pass

    @abstractmethod
    def d(self, y: NDArray, y_pred: NDArray) -> NDArray:
        """
        Compute the derivative of the loss with respect to the predictions y_pred.

        Args:
            y (NDArray): Ground truth targets. Shape: (batch_size, features).
            y_pred (NDArray): Predictions. Shape: (batch_size, features).

        Returns:
            NDArray: Gradient of the loss with respect to y_pred. Shape: (batch_size, features).
        """
        pass
