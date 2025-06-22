import numpy as np

from nn_core.layers import Layer
from nn_core.loss import Loss


class Network:
    def __init__(self):
        self.layers: list[Layer] = []
        self.loss: Loss = None

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def set_loss(self, loss: Loss):
        self.loss = loss

    def forward(self, input_data: np.ndarray, training: bool = True):
        """
        Perform forward propagation through all layers of the network.

        Args:
            input_data (np.ndarray): Input data. Shape: (batch_size, input_features)

        Returns:
            np.ndarray: Output data after passing through all layers.
                        Shape: (batch_size, output_features)
        """
        y_hat = input_data
        for layer in self.layers:
            y_hat = layer.forward(y_hat, training)
        return y_hat

    def backward(self, y, y_hat, lr):
        """
        Perform backward propagation through all layers of the network.

        Args:
            y (np.ndarray): Ground truth targets. Shape: (batch_size, output_features)
            y_hat (np.ndarray): Predicted outputs from the network. Shape: (batch_size, output_features)
            lr (float): Learning rate for updating layer parameters.
        """
        grad = self.loss.d(y, y_hat)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
