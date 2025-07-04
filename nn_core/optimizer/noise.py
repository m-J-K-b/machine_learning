import numpy as np
from numpy.typing import NDArray

from nn_core.optimizer.optimizer import Optimizer


class Noise(Optimizer):
    def __init__(self, lr: float = 0.01, noise_scale: float = 1.0) -> None:
        self.lr = lr
        self.noise_scale = noise_scale

    def update(self, param: NDArray, grad: NDArray, name: str) -> NDArray:
        noise = np.random.normal(0, np.std(grad) * self.noise_scale, size=grad.shape)
        return param - self.lr * (grad + noise)
