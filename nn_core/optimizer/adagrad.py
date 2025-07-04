from typing import Optional

import numpy as np
from numpy.typing import NDArray

from nn_core.optimizer.optimizer import Optimizer


class AdaGrad(Optimizer):
    def __init__(self, lr: float = 0.01, epsilon: float = 0.0001) -> None:
        self.lr = lr
        self.epsilon = epsilon
        self.g2 = {}

    def update(self, param: NDArray, grad: NDArray, name: str) -> NDArray:
        if name not in self.g2:
            self.g2[name] = np.zeros_like(param)

        self.g2[name] += grad**2

        lr = self.lr / (np.sqrt(self.g2[name]) + self.epsilon)
        return param - grad * lr
