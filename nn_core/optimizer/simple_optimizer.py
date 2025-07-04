import numpy as np
from numpy.typing import NDArray

from nn_core.optimizer.optimizer import Optimizer


class SimpleOptimizer(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def update(self, param: NDArray, grad: NDArray, name: str) -> NDArray:
        return param - grad * self.lr


class ClippedGradients(Optimizer):
    def __init__(self, lr: float = 0.01, max_grad=1.0) -> None:
        self.lr = lr
        self.max_grad = max_grad

    def update(self, param: NDArray, grad: NDArray, name: str) -> NDArray:
        return param - np.clip(grad, -self.max_grad, self.max_grad) * self.lr
