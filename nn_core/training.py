import threading
import time
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from nn_core.network import Network
from nn_core.optimizer.optimizer import Optimizer


class Progress:
    def __init__(self, epochs: int):
        self.total_epochs = epochs
        self.current_epoch = 0
        self.start_time = 0
        self.end_time = 0

    def next_epoch(self):
        self.current_epoch += 1

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def get_elapsed_time(self):
        return (self.end_time or time.time()) - self.start_time

    def get_avg_time_per_epoch(self):
        return self.get_elapsed_time() / max(1, self.current_epoch)

    def get_progress_fraction(self):
        return self.current_epoch / self.total_epochs

    def get_progress_percent(self):
        return 100 * (self.current_epoch / self.total_epochs)

    def get_finished(self):
        return self.current_epoch >= self.total_epochs

    def __str__(self):
        return (
            f"Epoch: {self.current_epoch}/{self.total_epochs} | "
            f"Progress: {self.get_progress_percent():.2f}% | "
            f"Elapsed: {self.get_elapsed_time():.2f}s"
        )


def run_in_thread(target_fn, *args, **kwargs):
    t = threading.Thread(target=target_fn, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t


TrainingMethod = Callable[[Network, NDArray, NDArray, int, Optimizer, Progress], None]


def mini_batch_gd(
    batch_size: int,
) -> TrainingMethod:
    def _mini_batch_gd(
        nn: Network,
        x: NDArray,
        y: NDArray,
        epochs: int,
        optimizer: Optimizer,
        progress: Progress,
    ):
        n_samples = x.shape[0]

        for _ in range(epochs):
            progress.next_epoch()
            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, batch_size):
                idx = indices[i : i + batch_size]
                x_sample = x[idx]
                y_sample = y[idx]
                y_pred = nn.forward(x_sample, training=True)
                nn.backward(y_sample, y_pred, optimizer)

    return _mini_batch_gd


def train(
    nn: Network,
    x: NDArray,
    y: NDArray,
    epochs: int,
    optimizer: Optimizer,
    method_fn: TrainingMethod,
    progress: Optional[Progress] = None,
):
    progress = progress or Progress(epochs)

    def _train():
        progress.start()
        method_fn(nn, x, y, epochs, optimizer, progress)
        progress.stop()

    run_in_thread(_train)
    return progress
