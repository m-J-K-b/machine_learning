from typing import Optional

import numpy as np
from numpy.typing import NDArray

from nn_core.layer import Layer
from nn_core.loss.loss import Loss
from nn_core.optimizer.optimizer import Optimizer
from nn_core.util import softmax


class Network:
    def __init__(self):
        self.layers: list[Layer] = []
        self.loss: Optional[Loss] = None

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def set_loss(self, loss: Loss):
        self.loss = loss

    def forward(self, x: NDArray, training: bool = False):
        """
        Perform forward propagation through all layers of the network.

        Args:
            input_data (NDArray): Input data. Shape: (batch_size, input_features)

        Returns:
            NDArray: Output data after passing through all layers.
                        Shape: (batch_size, output_features)
        """
        y = x
        for layer in self.layers:
            y = layer.forward(y, training)

        return y

    def backward(self, y: NDArray, y_pred: NDArray, optimizer: Optimizer):
        grad = self.loss.d(y, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, optimizer)

    def predict(self, x: NDArray) -> NDArray:
        """
        Returns the raw logits.
        """
        return self.forward(x, training=False)

    def predict_probabilities(self, x: NDArray) -> NDArray:
        """
        Returns class probabilities by applying softmax to the logits.
        """
        return softmax(self.forward(x, training=False))

    def predict_classes(self, x: NDArray) -> NDArray:
        """
        Returns the most-likely class index for each sample.
        """
        probs = self.predict_probabilities(x)
        return np.argmax(probs, axis=1)
