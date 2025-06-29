# Source: https://www.cs.toronto.edu/~kriz/cifar.html
import os
import pickle
from pathlib import Path

import h5py
import numpy as np


def unpickle(file):
    with open(file, "rb") as fo:
        batch = pickle.load(fo, encoding="bytes")
    return batch


if __name__ == "__main__":
    path = Path(__file__).parent

    # Load training batches
    xs = []
    ys = []
    for i in range(1, 6):
        batch_path = path / f"data_batch_{i}"
        batch = unpickle(batch_path)
        data = batch[b"data"]  # shape (10000, 3072)
        labels = batch[b"labels"]  # list of 10000 integers
        xs.append(data)
        ys.append(np.array(labels, dtype=np.uint8))

    train_images = np.concatenate(xs, axis=0)  # shape (50000, 3072)
    train_labels = np.concatenate(ys, axis=0)  # shape (50000,)

    # Load test batch
    test_path = path / "test_batch"
    test_batch = unpickle(test_path)
    test_images = test_batch[b"data"]
    test_labels = np.array(test_batch[b"labels"], dtype=np.uint8)

    # Load label names and convert from bytes to str
    meta = unpickle(path / "batches.meta")
    label_names = [name.decode("utf-8") for name in meta[b"label_names"]]
    print("Label names:", label_names)

    # Dataset description
    description = (
        "The CIFAR-10 dataset consists of 50,000 training images and labels, and 10,000 test images and labels. "
        "Each image is a 3072-element numpy array of uint8s. The first 1024 entries contain the red channel values, "
        "the next 1024 the green, and the final 1024 the blue. Images are stored in row-major order: "
        "the first 32 values are the red values of the first row of the image.\n\n"
        "Labels are integers from 0 to 9, representing the image class. Class names corresponding to labels can be found "
        "in the 'label_names' HDF5 attribute.\n\n"
        "The dataset is stored in HDF5 format under the groups: 'train/images', 'train/labels', "
        "'test/images', and 'test/labels'."
    )

    # Write to HDF5
    with h5py.File(path / "cifar10.h5", "w") as f:
        f.create_dataset("train/images", data=train_images)
        f.create_dataset("train/labels", data=train_labels)
        f.create_dataset("test/images", data=test_images)
        f.create_dataset("test/labels", data=test_labels)

        f.attrs["description"] = description
        f.attrs["label_names"] = np.bytes_(label_names)  # Store as bytestring array

    print("CIFAR-10 data written to cifar10.h5")
