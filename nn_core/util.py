import numpy as np
from numpy.typing import NDArray


def softmax(y_pred: NDArray) -> NDArray:
    exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
