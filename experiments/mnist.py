import copy
import pickle
import threading
import time

import h5py
import log_util
import numpy as np
from matplotlib import pyplot as plt

from nn_core import Network, Progress, layer, loss, mini_batch_gd, optimizer, train
from nn_core.util import softmax
from util.util import compute_accuracy


def main():
    ### set up logging
    logger = log_util.Logger(log_util.LogLevel.INFO)

    ### load dataset
    with h5py.File("./datasets/fashion-mnist/fashion-mnist.h5", "r") as f:
        train_images = f["train/images"][:]
        train_labels = f["train/labels"][:]
        test_images = f["test/images"][:]
        test_labels = f["test/labels"][:]

    # Normalize images
    train_images = train_images.astype(np.float64) / 255.0
    test_images = test_images.astype(np.float64) / 255.0

    # one hot encode label
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]

    ### Build Network
    nn = Network()
    nn.set_loss(loss.CrossEntropy())
    nn.add_layer(layer.DenseLayer(train_images.shape[-1], 512))
    nn.add_layer(layer.ActivationLayer(layer.activation.LeakyReLU()))
    nn.add_layer(layer.DenseLayer(512, 256))
    nn.add_layer(layer.ActivationLayer(layer.activation.LeakyReLU()))
    nn.add_layer(layer.DenseLayer(256, 128))
    nn.add_layer(layer.ActivationLayer(layer.activation.LeakyReLU()))
    nn.add_layer(layer.DenseLayer(128, 10))
    nn.add_layer(layer.ActivationLayer(layer.activation.LeakyReLU()))

    EPOCHS = 100
    LR = 0.01
    BATCH = 256

    lock = threading.Lock()
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    p = Progress(EPOCHS)

    def _next_epoch():
        p.current_epoch += 1
        nn_c = copy.deepcopy(nn)
        train_y_pred = nn_c.predict(train_images)
        test_y_pred = nn_c.predict(test_images)

        with lock:
            train_acc.append(compute_accuracy(train_labels, softmax(train_y_pred)))
            test_acc.append(compute_accuracy(test_labels, softmax(test_y_pred)))
            train_loss.append(nn.loss.f(train_labels, train_y_pred))
            test_loss.append(nn.loss.f(test_labels, test_y_pred))

        logger.clear_block()
        logger.info(
            f"Epoch: {p.current_epoch} / {p.total_epochs}, train acc: {train_acc[-1] * 100:.3f}%, test acc: {test_acc[-1] * 100:.3f}%, train loss: {train_loss[-1]:.5f}, test loss: {test_loss[-1]:.5f}"
        )

    p.next_epoch = _next_epoch

    logger.info("Starting training")
    logger.start_block()
    train(
        nn,
        train_images,
        train_labels,
        EPOCHS,
        optimizer.Momentum(lr=LR, momentum=0.9),
        mini_batch_gd(BATCH),
        progress=p,
    )
    while not p.get_finished():
        time.sleep(1)
    logger.end_block()

    plt.plot(np.arange(EPOCHS - 1), test_acc, label="test")
    plt.plot(np.arange(EPOCHS - 1), train_acc, label="train")
    plt.show()
    plt.plot(np.arange(EPOCHS - 1), test_loss, label="test")
    plt.plot(np.arange(EPOCHS - 1), train_loss, label="train")
    plt.show()

    ### save model
    with open("./models/mnist.pkl", "wb") as f:
        pickle.dump(nn, f)


if __name__ == "__main__":
    main()
