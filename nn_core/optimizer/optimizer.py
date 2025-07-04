from abc import ABC, abstractmethod

from numpy.typing import NDArray


class Optimizer(ABC):
    @abstractmethod
    def update(self, param: NDArray, grad: NDArray, name: str) -> NDArray:
        pass
