from typing import Optional

import numpy as np
from numpy.typing import NDArray

from nn_core.layers.layer import Layer


class BatchNormalizationLayer(Layer):
    def __init__(self, momentum: float = 0.9, eps: float = 1e-5):
        self.momentum = momentum
        self.eps = eps

        self.running_mean: Optional[NDArray] = None
        self.running_var: Optional[NDArray] = None

    def forward(self, x: NDArray, training: bool) -> NDArray:
        if self.running_mean is None or self.running_var is None:
            D = x.shape[1]
            self.running_mean = np.zeros((1, D))
            self.running_var = np.ones((1, D))

        assert self.running_mean is not None
        assert self.running_var is not None

        if training:
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

            mean = batch_mean
            std = np.sqrt(batch_var + self.eps)

            self._x = x
            self._mean = mean
            self._std = std
        else:
            mean = self.running_mean
            std = np.sqrt(self.running_var + self.eps)

        x_hat = (x - mean) / std
        return x_hat

    def backward(self, grad: NDArray, lr: float) -> NDArray:
        """
        grad: dL/dy from next layer
        returns: dL/dx to pass to previous layer
        """
        x, mean, std = self._x, self._mean, self._std
        N, D = grad.shape

        dx_hat = grad

        dvar = np.sum(
            dx_hat * (x - mean) * -0.5 * std ** (-3),
            axis=0,
            keepdims=True,
        )

        dmean = np.sum(
            dx_hat * -1 / std,
            axis=0,
            keepdims=True,
        ) + dvar * np.mean(-2 * (x - mean), axis=0, keepdims=True)

        dx = dx_hat / std + dvar * 2 * (x - mean) / N + dmean / N
        return dx
