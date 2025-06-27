import copy
import pickle
import time
from pathlib import Path

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
from util import compute_accuracy, load_mnist_dataset

# ------------------ Simple Console Logger ------------------


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


# ------------------ Main Training Script ------------------


def main():
    ### Set log level
    logger.level = log_util.LogLevel.DEBUG

    ### Load dataset
    logger.info("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist_dataset()
    logger.debug(
        f"x_train: {x_train.shape}, y_train: {y_train.shape}, min values: {np.min(x_train)}, max values: {np.max(x_train)}"
    )

    ### Build network
    nn = Network()
    nn.set_loss(CrossEntropyLoss())  # Let loss handle softmax

    # Layer 1
    nn.add_layer(DenseLayer(x_train.shape[1], 512, DenseInitializers.he_normal))
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

    ### Training
    # params
    EPOCHS = 500
    LR = 1e-3
    BATCH = 256
    accuracy_history = {}
    logger.info(f"Training config -> epochs: {EPOCHS}, lr: {LR}, batch: {BATCH}")

    # start training
    progress = train(nn, x_train, y_train, EPOCHS, LR, mini_batch_gd(BATCH, 0.99))

    # Real-time feedback
    logger.start_block()
    UPDATE_INTERVAL = 0.5
    while not progress.get_finished():
        start_time = time.perf_counter()

        logger.clear_block(clear_lines=False)

        nn_snapshot = copy.deepcopy(nn)
        acc = compute_accuracy(y_test, nn_snapshot.forward(x_test, training=False))

        epoch = progress.current_epoch
        if epoch not in accuracy_history:
            accuracy_history[epoch] = [0, 0]
        accuracy_history[epoch][0] += acc
        accuracy_history[epoch][1] += 1

        log_progress(progress, acc)

        elapsed = time.perf_counter() - start_time
        logger.debug(f"Update lag: {abs(min(0, UPDATE_INTERVAL - elapsed))}")
        time.sleep(max(0.0, UPDATE_INTERVAL - elapsed))
    logger.clear_block(clear_lines=False)
    logger.end_block()

    ### Final evaluation
    final_acc = compute_accuracy(y_test, nn.forward(x_test, training=False))

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
    filename = "bert.pkl"
    logger.info(f"storing trained network to: {filename}")
    with open(filename, "wb") as f:
        pickle.dump(nn, f)


if __name__ == "__main__":
    logger = log_util.Logger(
        file_path=Path("logs/mnist_training.txt"),
        max_bytes=int(1e9),
    )

    main()
