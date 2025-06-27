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
    @classmethod
    def softmax(cls, logits: NDArray) -> NDArray:
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def f(self, y: NDArray, logits: NDArray) -> float:
        """
        Cross-entropy loss with softmax.
        Args:
            y: One-hot encoded true labels. Shape: (batch_size, num_classes)
            logits: Raw model outputs (pre-softmax). Shape: (batch_size, num_classes)
        Returns:
            Scalar loss value
        """
        probs = self.softmax(logits)
        epsilon = 1e-12
        probs = np.clip(probs, epsilon, 1.0 - epsilon)
        return -np.mean(np.sum(y * np.log(probs), axis=1))

    def d(self, y: NDArray, logits: NDArray) -> NDArray:
        """
        Gradient of cross-entropy with softmax.
        Args:
            y: One-hot encoded true labels. Shape: (batch_size, num_classes)
            logits: Raw model outputs (pre-softmax). Shape: (batch_size, num_classes)
        Returns:
            Gradient of loss w.r.t. logits. Same shape as input.
        """
        probs = self.softmax(logits)
        return (probs - y) / y.shape[0]
