from typing import Optional

import numpy as np
from numpy.typing import NDArray

from nn_core.layers import Layer
from nn_core.loss import MSE, Loss


class Network:
    def __init__(self):
        self.layers: list[Layer] = []
        self.loss: Optional[Loss] = None

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def set_loss(self, loss: Loss):
        self.loss = loss

    def forward(self, input_data: NDArray, training: bool = True):
        """
        Perform forward propagation through all layers of the network.

        Args:
            input_data (NDArray): Input data. Shape: (batch_size, input_features)

        Returns:
            NDArray: Output data after passing through all layers.
                        Shape: (batch_size, output_features)
        """
        y_hat = input_data
        for layer in self.layers:
            y_hat = layer.forward(y_hat, training)
        return y_hat

    def backward(self, y: NDArray, y_hat: NDArray, lr: float):
        """
        Perform backward propagation through all layers of the network.

        Args:
            y (NDArray): Ground truth targets. Shape: (batch_size, output_features)
            y_hat (NDArray): Predicted outputs from the network. Shape: (batch_size, output_features)
            lr (float): Learning rate for updating layer parameters.
        """
        if self.loss is None:
            raise RuntimeError(f"Loss function was not set on network")

        grad = self.loss.d(y, y_hat)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
