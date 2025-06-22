import threading
import time

import numpy as np

from nn_core.network import Network


class Progress:
    """
    Tracks progress, timing and loss during training.
    """

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
    """
    Runs a target function in a daemon thread.
    """
    t = threading.Thread(target=target_fn, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t


def train(nn: Network, x, y, epochs, lr, method_fn):
    """
    Trains the network using the provided method in a background thread.

    Returns:
        Progress: Progress tracker object.
    """
    progress = Progress(nn, epochs)

    def _train():
        progress.start()
        method_fn(nn, x, y, epochs, lr, progress)
        progress.stop()

    run_in_thread(_train)
    return progress


def mini_batch_gd(mini_batch_size, decay_rate=0.95, min_lr=1e-9):
    """
    Mini-batch Gradient Descent with optional learning rate decay.

    Args:
        mini_batch_size (int): Size of each mini-batch.
        decay_rate (float): Learning rate decay factor per epoch (default 0.95).

    Returns:
        function: Training function.
    """

    def _mini_batch_gd(nn, x, y, epochs, lr, progress):
        n_samples = x.shape[0]
        initial_lr = lr  # Store initial learning rate

        for epoch in range(epochs):
            lr = max(min_lr, initial_lr * (decay_rate**epoch))

            indices = np.random.permutation(n_samples)
            loss_epoch = 0

            for i in range(0, n_samples, mini_batch_size):
                idx = indices[i]
                x_sample = x[idx : idx + mini_batch_size]
                y_sample = y[idx : idx + mini_batch_size]
                y_hat = nn.forward(x_sample)
                nn.backward(y_sample, y_hat, lr)
                loss_epoch += nn.loss.f(y_sample, y_hat)

            loss_epoch /= n_samples
            progress.next_epoch(loss_epoch)

    return _mini_batch_gd


def stochastic_gd(decay_rate=0.95, min_lr=1e-9):
    """
    Stochastic Gradient Descent with learning rate decay and minimal learning rate.

    Args:
        decay_rate (float): Learning rate decay per epoch.
        min_lr (float): Minimum allowed learning rate to prevent vanishing updates.

    Returns:
        function: Training function.
    """

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
                y_hat = nn.forward(x_sample)
                nn.backward(y_sample, y_hat, lr)
                loss_epoch += nn.loss.f(y_sample, y_hat)

            loss_epoch /= n_samples
            progress.next_epoch(loss_epoch)

    return _stochastic_gd


def full_batch_gd():
    def _full_batch_gd(nn: Network, x, y, epochs, lr, progress: Progress):
        for _ in range(epochs):
            y_hat = nn.forward(x)
            nn.backward(y, y_hat, lr)
            progress.next_epoch()

    return _full_batch_gd


def full_batch_random_gd():
    def _full_batch_random_gd(nn: Network, x, y, epochs, lr, progress: Progress):
        n_samples = x.shape[0]
        for _ in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            y_hat = nn.forward(x_shuffled)
            nn.backward(y_shuffled, y_hat, lr)
            progress.next_epoch()

    return _full_batch_random_gd
