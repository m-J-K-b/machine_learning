from numpy.typing import NDArray

from nn_core.optimizer.optimizer import Optimizer


class WeightDecay(Optimizer):
    def __init__(self, lr=0.01, decay=0.01) -> None:
        self.decay = decay
        self.lr = lr

    def update(self, param: NDArray, grad: NDArray, name: str) -> NDArray:
        return param - self.lr * grad - self.lr * self.decay * param
