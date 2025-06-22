from logging import info

import numpy as np


def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


def compute_accuracy(y_true, y_pred):
    """
    Computes accuracy between one-hot true labels and predicted probabilities.
    """
    true = np.argmax(y_true, axis=1)
    pred = np.argmax(y_pred, axis=1)
    return np.mean(true == pred)


def load_mnist_dataset():
    # TODO: Flatten the image data with a locality preserving curve (hilbert / z)
    data = np.load("./datasets/mnist/mnist_data.npz")
    x_train, y_train = data["train_images"], data["train_labels"]
    x_test, y_test = data["test_images"], data["test_labels"]

    # Preprocess
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0
    y_train = one_hot_encode(y_train, 10)
    y_test = one_hot_encode(y_test, 10)

    return x_train, y_train, x_test, y_test
