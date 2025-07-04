import copy
import threading
import time

import h5py
import log_util
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

from nn_core import Network, Progress, layer, loss, mini_batch_gd, optimizer, train
from util.util import compute_accuracy, one_hot_encode


def run_experiment(
    nn_factory,
    optimizer,
    train_images,
    train_labels,
    test_images,
    test_labels,
    epochs: int,
    batch_size: int,
    logger,
) -> dict[str, list[float]]:
    """
    Runs one full training session with per‐epoch logging.
    Returns a dict with lists ["train_acc","test_acc","train_loss","test_loss"].
    """
    nn = nn_factory()
    progress = Progress(epochs)

    train_acc, test_acc = [], []
    train_loss, test_loss = [], []
    lock = threading.Lock()

    def _next_epoch():
        progress.current_epoch += 1
        # take snapshot before computing metrics
        nn_snapshot = copy.deepcopy(nn)

        acc_tr = compute_accuracy(
            train_labels, nn_snapshot.predict_probabilities(train_images)
        )
        acc_te = compute_accuracy(
            test_labels, nn_snapshot.predict_probabilities(test_images)
        )
        loss_tr = nn.loss.f(train_labels, nn.forward(train_images))
        loss_te = nn.loss.f(test_labels, nn.forward(test_images))

        with lock:
            train_acc.append(acc_tr)
            test_acc.append(acc_te)
            train_loss.append(loss_tr)
            test_loss.append(loss_te)

            # logging
            logger.clear_block()
            logger.info(
                f"epoch: {progress.current_epoch} / {progress.total_epochs}  "
                f"train acc: {acc_tr:.4f}, test acc: {acc_te:.4f}  "
                f"train loss: {loss_tr:.4f}, test loss: {loss_te:.4f}"
            )

    progress.next_epoch = _next_epoch

    # start training
    train(
        nn,
        train_images,
        train_labels,
        epochs,
        optimizer,
        mini_batch_gd(batch_size),
        progress=progress,
    )

    # wait for completion
    while not progress.get_finished():
        time.sleep(0.1)

    # return copied lists
    with lock:
        return {
            "train_acc": train_acc[:],
            "test_acc": test_acc[:],
            "train_loss": train_loss[:],
            "test_loss": test_loss[:],
        }


def main():
    logger = log_util.Logger(log_util.LogLevel.INFO)

    with h5py.File("./datasets/fashion-mnist/fashion-mnist.h5", "r") as f:
        train_images = f["train/images"][:]
        train_labels = f["train/labels"][:]
        test_images = f["test/images"][:]
        test_labels = f["test/labels"][:]

    # normalize images
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # One-hot encode labels
    num_classes = len(np.unique(train_labels))
    train_labels = one_hot_encode(train_labels, num_classes)
    test_labels = one_hot_encode(test_labels, num_classes)

    subset_size = 100
    train_images = train_images[:subset_size]
    train_labels = train_labels[:subset_size]

    ### Build network
    nn = None

    def init_nn():
        nn = Network()
        nn.set_loss(loss.CrossEntropy())
        nn.add_layer(
            layer.DenseLayer(train_images.shape[1], 512, layer.DenseLayer.he_normal)
        )
        nn.add_layer(layer.ActivationLayer(layer.activation.LeakyReLU()))
        nn.add_layer(layer.DenseLayer(512, 128, layer.DenseLayer.he_normal))
        nn.add_layer(layer.ActivationLayer(layer.activation.LeakyReLU()))
        nn.add_layer(layer.DenseLayer(128, 10, layer.DenseLayer.he_normal))
        return nn

    EPOCHS, BATCH, LR = 50, 1, 0.01
    optimizers = {
        "AdaGrad": optimizer.AdaGrad(lr=LR, epsilon=1e-5),
        "Momentum": optimizer.Momentum(lr=LR),
        "Noise": optimizer.Noise(lr=LR, noise_scale=0.01),
        "WeightDecay": optimizer.WeightDecay(lr=LR, decay=0.01),
    }

    REPEATS = 1
    all_records = {}

    for name, opt in optimizers.items():
        runs = []
        for r in range(REPEATS):
            logger.info(f"Running {name} trial {r+1}/{REPEATS}")
            logger.start_block()
            rec = run_experiment(
                init_nn,
                copy.deepcopy(opt),
                train_images,
                train_labels,
                test_images,
                test_labels,
                EPOCHS,
                BATCH,
                logger,
            )
            logger.clear_block()
            logger.end_block()
            runs.append(rec)

        # Now stack into arrays shape (REPEATS, EPOCHS)
        def stack(metric):
            return np.vstack([run[metric] for run in runs])

        all_records[name] = {
            "train_acc": stack("train_acc"),
            "test_acc": stack("test_acc"),
            "train_loss": stack("train_loss"),
            "test_loss": stack("test_loss"),
        }

    epochs = np.arange(1, EPOCHS)

    def plot_metric(key, ylabel, filename):
        plt.figure()
        for name, rec in all_records.items():
            data = rec[key]  # shape (repeats, epochs)
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            plt.plot(epochs, mean, label=name)
            plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} (mean ± std over {REPEATS} runs)")
        plt.legend()
        plt.savefig(filename)
        plt.close()

    import os

    os.makedirs("./plots", exist_ok=True)
    plot_metric("test_acc", "Test Accuracy", "./plots/test_accuracy.png")
    plot_metric("train_acc", "Train Accuracy", "./plots/train_accuracy.png")
    plot_metric("train_loss", "Train Loss", "./plots/train_loss.png")
    plot_metric("test_loss", "Test Loss", "./plots/test_loss.png")


if __name__ == "__main__":
    main()
