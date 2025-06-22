import numpy as np

from nn_core.layers.layer import Layer


class BatchNormalizationLayer(Layer):
    def __init__(self, momentum=0.9, eps=1e-5):
        # Hyperparameters
        self.momentum = momentum
        self.eps = eps

        # Running (inference) stats, initialized once on first forward
        self.running_mean = None
        self.running_var = None

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        # Initialize running stats on first call
        if self.running_mean is None:
            D = x.shape[1]
            self.running_mean = np.zeros((1, D))
            self.running_var = np.ones((1, D))

        if training:
            # Compute batch stats
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)

            # Update running stats
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

            mean = batch_mean
            std = np.sqrt(batch_var + self.eps)

            # Store for backward
            self._x = x
            self._mean = mean
            self._std = std

        else:
            # In inference, use the running stats
            mean = self.running_mean
            std = np.sqrt(self.running_var + self.eps)

        # Normalize
        x_hat = (x - mean) / std
        return x_hat

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        """
        grad: dL/dy from next layer
        returns: dL/dx to pass to previous layer
        """
        x, mean, std = self._x, self._mean, self._std
        N, D = grad.shape

        # Gradient of normalized x
        dx_hat = grad

        # dvar
        dvar = np.sum(
            dx_hat * (x - mean) * -0.5 * std ** (-3),
            axis=0,
            keepdims=True,
        )

        # dmean
        dmean = np.sum(
            dx_hat * -1 / std,
            axis=0,
            keepdims=True,
        ) + dvar * np.mean(-2 * (x - mean), axis=0, keepdims=True)

        # dx
        dx = dx_hat / std + dvar * 2 * (x - mean) / N + dmean / N
        return dx
