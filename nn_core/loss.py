from abc import abstractmethod

import numpy as np


class Loss:
    @abstractmethod
    def f(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Compute the scalar loss value over the batch.

        Args:
            y (np.ndarray): Ground truth targets. Shape: (batch_size, features).
            y_hat (np.ndarray): Predictions. Shape: (batch_size, features).

        Returns:
            float: Scalar loss value averaged over the batch and features.
        """
        pass

    @abstractmethod
    def d(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the loss with respect to the predictions y_hat.

        Args:
            y (np.ndarray): Ground truth targets. Shape: (batch_size, features).
            y_hat (np.ndarray): Predictions. Shape: (batch_size, features).

        Returns:
            np.ndarray: Gradient of the loss with respect to y_hat. Shape: (batch_size, features).
        """
        pass


class MSE(Loss):
    def f(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Compute Mean Squared Error (MSE) loss.

        Args:
            y (np.ndarray): Ground truth targets. Shape: (batch_size, features).
            y_hat (np.ndarray): Predictions. Shape: (batch_size, features).

        Returns:
            float: Scalar MSE loss averaged over the batch and features.
        """
        return np.mean((y - y_hat) ** 2).astype(float) / 2

    def d(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Compute derivative of Mean Squared Error (MSE) loss with respect to predictions.

        Args:
            y (np.ndarray): Ground truth targets. Shape: (batch_size, features).
            y_hat (np.ndarray): Predictions. Shape: (batch_size, features).

        Returns:
            np.ndarray: Gradient of MSE loss w.r.t y_hat. Shape: (batch_size, features).
        """
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
