import threading
import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from nn_core.network import Network


class Progress:
    def __init__(self, nn: Network, epochs: int):
        self.nn: Network = nn
        self.total_epochs = epochs
        self.current_epoch = 0
        self.loss_history = []
        self.start_time = 0
        self.end_time = 0

    def next_epoch(self, loss=None):
        self.current_epoch += 1
        if loss is not None:
            self.loss_history.append(loss)

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


TrainingMethod = Callable[[Network, NDArray, NDArray, int, float, Progress], None]


def mini_batch_gd(mini_batch_size, decay_rate=0.95, min_lr=1e-9) -> TrainingMethod:
    def _mini_batch_gd(
        nn: Network, x: NDArray, y: NDArray, epochs: int, lr: float, progress: Progress
    ):
        n_samples = x.shape[0]
        initial_lr = lr

        for epoch in range(epochs):
            lr = max(min_lr, initial_lr * (decay_rate**epoch))

            indices = np.random.permutation(n_samples)
            loss_epoch = 0

            for i in range(0, n_samples, mini_batch_size):
                idx = indices[i : i + mini_batch_size]
                x_sample = x[idx]
                y_sample = y[idx]
                y_pred = nn.forward(x_sample, training=True)
                nn.backward(y_sample, y_pred, lr)
                loss_epoch += nn.loss.f(y_sample, y_pred)

            loss_epoch /= n_samples
            progress.next_epoch(loss_epoch)

    return _mini_batch_gd


def stochastic_gd(decay_rate=0.95, min_lr=1e-9) -> TrainingMethod:
    def _stochastic_gd(nn: Network, x, y, epochs, lr, progress: Progress):
        n_samples = x.shape[0]
        initial_lr = lr

        for epoch in range(epochs):
            lr = max(initial_lr * (decay_rate**epoch), min_lr)

            indices = np.random.permutation(n_samples)
            loss_epoch = 0

            for idx in indices:
                x_sample = x[idx : idx + 1]
                y_sample = y[idx : idx + 1]
                y_pred = nn.forward(x_sample, training=True)
                nn.backward(y_sample, y_pred, lr)
                loss_epoch += nn.loss.f(y_sample, y_pred)

            loss_epoch /= n_samples
            progress.next_epoch(loss_epoch)

    return _stochastic_gd


def full_batch_gd() -> TrainingMethod:
    def _full_batch_gd(
        nn: Network, x: NDArray, y: NDArray, epochs: int, lr: float, progress: Progress
    ):
        for _ in range(epochs):
            y_pred = nn.forward(x, training=True)
            nn.backward(y, y_pred, lr)
            progress.next_epoch()

    return _full_batch_gd


def full_batch_random_gd() -> TrainingMethod:
    def _full_batch_random_gd(nn: Network, x, y, epochs, lr, progress: Progress):
        n_samples = x.shape[0]
        for _ in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            y_pred = nn.forward(x_shuffled, training=True)
            nn.backward(y_shuffled, y_pred, lr)
            progress.next_epoch()

    return _full_batch_random_gd


def train(
    nn: Network,
    x: NDArray,
    y: NDArray,
    epochs: int,
    lr: float,
    method_fn: TrainingMethod,
):
    progress = Progress(nn, epochs)

    def _train():
        progress.start()
        method_fn(nn, x, y, epochs, lr, progress)
        progress.stop()

    run_in_thread(_train)
    return progress
