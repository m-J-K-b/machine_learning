import numpy as np
from numpy.typing import NDArray

from nn_core.loss.loss import Loss
from nn_core.util import softmax


class CrossEntropy(Loss):
    def f(self, y: NDArray, y_pred: NDArray) -> float:
        y_pred = softmax(y_pred)
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.mean(np.sum(y * np.log(y_pred), axis=1))

    def d(self, y: NDArray, y_pred: NDArray) -> NDArray:
        return (softmax(y_pred) - y) / y.shape[0]
