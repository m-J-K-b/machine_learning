import copy
import pickle
import time
from pathlib import Path

import h5py
import log_util
import numpy as np
from matplotlib import pyplot as plt

from nn_core import (
    Activation,
    ActivationLayer,
    BatchNormalizationLayer,
    CrossEntropyLoss,
    DenseInitializers,
    DenseLayer,
    DropoutLayer,
    Network,
)
from nn_core.training import mini_batch_gd, train
from util.util import compute_accuracy, one_hot_encode


def log_progress(progress, accuracy, bar_length=30):
    """
    Formats and prints the training progress as a multi-line block.
    """
    loss = progress.loss_history[-1] if progress.loss_history else float("nan")
    frac = progress.get_progress_fraction()
    block = int(bar_length * frac)
    bar = "#" * block + "-" * (bar_length - block)

    elapsed = progress.get_elapsed_time()
    total_est = elapsed / frac if frac else float("nan")
    remaining = total_est - elapsed

    logger.info("=" * 70)
    logger.info(" Training Progress")
    logger.info("" + "-" * 70)
    logger.info(
        f" Epoch               : {progress.current_epoch} / {progress.total_epochs}"
    )
    logger.info(f" Avg Epoch Time      : {progress.get_avg_time_per_epoch():.4f}s")
    logger.info(
        f" Progress            : [{bar}] {progress.get_progress_percent():6.2f}%"
    )
    logger.info(f" Loss                : {loss}")
    logger.info(f" Accuracy            : {accuracy:.4f}")
    logger.info(f" Elapsed Time        : {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f" Estimated Total     : {total_est:.1f}s ({total_est/60:.1f}min)")
    logger.info(f" Estimated Remaining : {remaining:.1f}s ({remaining/60:.1f}min)")
    logger.info("=" * 70)


def monitor_training(
    progress, nn, test_images, test_labels, accuracy_history, update_interval=0.5
):
    """
    Monitors a running training session, periodically logging progress and
    validation accuracy until training is complete.

    Args:
        progress:    Progress object returned by train()
        nn:          The network being trained
        test_images: NumPy array of validation images
        test_labels: NumPy array of one-hot validation labels
        accuracy_history: dict mapping epoch -> [cumulative_acc, count]
        update_interval: float seconds between log updates
    """
    logger.start_block()
    try:
        while not progress.get_finished():
            tick_start = time.perf_counter()

            # Clear last block of output (but keep block open)
            logger.clear_block(clear_lines=False)

            # Snapshot network and compute current validation accuracy
            nn_snapshot = copy.deepcopy(nn)
            acc = compute_accuracy(
                test_labels, nn_snapshot.forward(test_images, training=False)
            )

            # Record into history
            epoch = progress.current_epoch
            accuracy_history.setdefault(epoch, [0, 0])
            accuracy_history[epoch][0] += acc
            accuracy_history[epoch][1] += 1

            # Do the actual logging
            log_progress(progress, acc)

            # Respect update interval
            elapsed = time.perf_counter() - tick_start
            logger.debug(f"Update lag: {max(0.0, update_interval - elapsed):.3f}s")
            time.sleep(max(0.0, update_interval - elapsed))
    finally:
        # Ensure we clean up the log block at the end
        logger.clear_block(clear_lines=False)
        logger.end_block()


def main():
    ### Set log level
    logger.level = log_util.LogLevel.DEBUG

    ### Load dataset
    logger.info("Loading MNIST dataset...")
    with h5py.File("./datasets/mnist/mnist.h5", "r") as f:
        train_images = f["train/images"][:]
        train_labels = f["train/labels"][:]
        test_images = f["test/images"][:]
        test_labels = f["test/labels"][:]
        logger.info(f.attrs["description"])

    # normalize images
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # One-hot encode labels
    num_classes = 10
    train_labels = one_hot_encode(train_labels, num_classes)
    test_labels = one_hot_encode(test_labels, num_classes)

    logger.debug(f"x_train: {train_images.shape}, y_train: {train_labels.shape}")

    ### Build network
    nn = Network()
    nn.set_loss(CrossEntropyLoss(apply_softmax=True))

    # Layer 1
    nn.add_layer(DenseLayer(train_images.shape[1], 512, DenseInitializers.he_normal))
    nn.add_layer(BatchNormalizationLayer())
    nn.add_layer(ActivationLayer(Activation.LeakyReLU()))
    nn.add_layer(DropoutLayer(0.2))  # Drop 20%

    # Layer 2
    nn.add_layer(DenseLayer(512, 256, DenseInitializers.he_normal))
    nn.add_layer(BatchNormalizationLayer())
    nn.add_layer(ActivationLayer(Activation.LeakyReLU()))
    nn.add_layer(DropoutLayer(0.2))

    # Layer 3
    nn.add_layer(DenseLayer(256, 128, DenseInitializers.he_normal))
    nn.add_layer(BatchNormalizationLayer())
    nn.add_layer(ActivationLayer(Activation.LeakyReLU()))
    nn.add_layer(DropoutLayer(0.2))

    # Output Layer
    nn.add_layer(DenseLayer(128, 10, DenseInitializers.he_normal))
    # nn.add_layer(ActivationLayer(Activation.Softmax()))

    ### Training
    # params
    EPOCHS = 100
    LR = 1e-3
    BATCH = 256
    accuracy_history = {}
    logger.info(f"Training config -> epochs: {EPOCHS}, lr: {LR}, batch: {BATCH}")

    # start training
    progress = train(
        nn, train_images, train_labels, EPOCHS, LR, mini_batch_gd(BATCH, 0.99)
    )

    monitor_training(
        progress, nn, test_images, test_labels, accuracy_history, update_interval=0.5
    )

    ### Final evaluation
    final_acc = compute_accuracy(test_labels, nn.forward(test_images, training=False))

    if progress.current_epoch not in accuracy_history:
        accuracy_history[progress.current_epoch] = [0, 0]
    accuracy_history[progress.current_epoch][0] += final_acc
    accuracy_history[progress.current_epoch][1] += 1

    accuracy_history = [a / n for a, n in accuracy_history.values()]

    # final log
    log_progress(progress, final_acc)
    logger.info(f"Training complete. Final test accuracy: {final_acc:.4f}")

    # Plots
    epochs = np.arange(1, len(accuracy_history) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Loss curve
    ax1.plot(progress.loss_history, label="Training Loss", linewidth=2)
    ax1.set_title("Training Loss over Mini-Batches")
    ax1.set_xlabel("Mini-Batch Step")
    ax1.set_ylabel("Loss")
    ax1.grid(linestyle="--", alpha=0.7)
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, accuracy_history, marker="o", linewidth=2, label="Val Accuracy")
    ax2.set_title("Validation Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_xticks(epochs)
    ax2.grid(linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    ### store trained nn
    filename = "clark.pkl"
    logger.info(f"storing trained network to: {filename}")
    with open(filename, "wb") as f:
        pickle.dump(nn, f)


if __name__ == "__main__":
    logger = log_util.Logger(
        file_path=Path("logs/mnist_training.txt"),
        max_bytes=int(1e9),
    )

    main()
