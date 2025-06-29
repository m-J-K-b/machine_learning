import numpy as np
from numpy.typing import NDArray


def one_hot_encode(y: NDArray, num_classes: int):
    return np.eye(num_classes)[y]


def compute_accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    """
    Computes accuracy between one-hot true labels and predicted probabilities.
    """
    true = np.argmax(y_true, axis=1)
    pred = np.argmax(y_pred, axis=1)
    return np.mean(true == pred)
