from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray


class Loss:
    @abstractmethod
    def f(self, y: NDArray, y_hat: NDArray) -> float:
        """
        Compute the scalar loss value over the batch.

        Args:
            y (NDArray): Ground truth targets. Shape: (batch_size, features).
            y_hat (NDArray): Predictions. Shape: (batch_size, features).

        Returns:
            float: Scalar loss value averaged over the batch and features.
        """
        pass

    @abstractmethod
    def d(self, y: NDArray, y_hat: NDArray) -> NDArray:
        """
        Compute the derivative of the loss with respect to the predictions y_hat.

        Args:
            y (NDArray): Ground truth targets. Shape: (batch_size, features).
            y_hat (NDArray): Predictions. Shape: (batch_size, features).

        Returns:
            NDArray: Gradient of the loss with respect to y_hat. Shape: (batch_size, features).
        """
        pass


class MSE(Loss):
    def f(self, y: NDArray, y_hat: NDArray) -> float:
        return np.mean((y - y_hat) ** 2).astype(float) / 2

    def d(self, y: NDArray, y_hat: NDArray) -> NDArray:
        return (y_hat - y) / y.shape[0]  # Normalize by batch size


class CrossEntropyLoss(Loss):
    def f(self, y, y_hat):
        epsilon = 1e-12
        y_hat = np.clip(y_hat, epsilon, 1.0 - epsilon)
        return -np.mean(np.sum(y * np.log(y_hat), axis=1))

    def d(self, y, y_hat):
        epsilon = 1e-12
        y_hat = np.clip(y_hat, epsilon, 1.0 - epsilon)
        return -(y / y_hat) / y.shape[0]
