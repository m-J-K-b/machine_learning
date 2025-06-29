from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray


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


class MSE(Loss):
    def f(self, y: NDArray, y_pred: NDArray) -> float:
        return np.mean((y - y_pred) ** 2).astype(float) / 2

    def d(self, y: NDArray, y_pred: NDArray) -> NDArray:
        return (y_pred - y) / y.shape[0]  # Normalize by batch size


# TODO: design choice leave softmax in or make it seperate
class CrossEntropyLoss(Loss):
    def __init__(self, apply_softmax=True):
        self._apply_softmax = apply_softmax

    @classmethod
    def softmax(cls, y_pred: NDArray) -> NDArray:
        exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def f(self, y: NDArray, y_pred: NDArray) -> float:
        if self._apply_softmax:
            y_pred = self.softmax(y_pred)
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.mean(np.sum(y * np.log(y_pred), axis=1))

    def d(self, y: NDArray, y_pred: NDArray) -> NDArray:
        return (self.softmax(y_pred) - y) / y.shape[0]
