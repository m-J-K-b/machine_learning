import numpy as np
from numpy.typing import NDArray

from nn_core.loss.loss import Loss


class MeanSquaredError(Loss):
    def f(self, y: NDArray, y_pred: NDArray) -> float:
        return np.mean((y - y_pred) ** 2).astype(float) / 2

    def d(self, y: NDArray, y_pred: NDArray) -> NDArray:
        return (y_pred - y) / y.shape[0]
