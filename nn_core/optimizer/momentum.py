import numpy as np
from numpy.typing import NDArray

from nn_core.optimizer.optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def update(self, param: NDArray, grad: NDArray, name: str) -> NDArray:
        if name not in self.v:
            self.v[name] = np.zeros_like(param)
        self.v[name] = self.v[name] * self.momentum + grad * (1 - self.momentum)
        return param - self.v[name] * self.lr
