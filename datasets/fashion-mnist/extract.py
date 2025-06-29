from pathlib import Path

import h5py
import numpy as np

if __name__ == "__main__":
    path = Path(__file__).parent

    train_labels_path = path / "train-labels-idx1-ubyte"
    train_images_path = path / "train-images-idx3-ubyte"
    test_labels_path = path / "test-labels-idx1-ubyte"
    test_images_path = path / "test-images-idx3-ubyte"

    with open(train_labels_path, "rb") as fb:
        train_labels = np.frombuffer(fb.read(), dtype=np.uint8, offset=8)

    with open(train_images_path, "rb") as fb:
        train_images = np.frombuffer(fb.read(), dtype=np.uint8, offset=16).reshape(
            len(train_labels), 784
        )

    with open(test_labels_path, "rb") as fb:
        test_labels = np.frombuffer(fb.read(), dtype=np.uint8, offset=8)

    with open(test_images_path, "rb") as fb:
        test_images = np.frombuffer(fb.read(), dtype=np.uint8, offset=16).reshape(
            len(test_labels), 784
        )

    description = (
        "The fashion-MNIST dataset consists of 60,000 training images and labels, and 10,000 test images and labels. "
        "Each image is a 28x28 grayscale representation of a clothing article, stored as a flattened uint8 array of length 784. "
        "Each image is stored as a row vector, flattened in row-major order (left to right, top to bottom). "
        "The labels are integers from 0 to 9, where each label indicates which class is shown in the corresponding image, "
        "also stored as a uint8 array. The string representation of the labels is accessible from the label_names attribute in this dataset, "
        "a list of 10 names for each class. The dataset is stored in HDF5 format, with training and test sets organized into "
        "'train/images', 'train/labels', 'test/images', and 'test/labels'."
    )

    label_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # Convert label_names to numpy array of fixed-length ascii bytestrings for HDF5 attribute
    label_names_np = np.array(label_names, dtype="S")

    with h5py.File(path / "fashion-mnist.h5", "w") as f:
        f.create_dataset("train/images", data=train_images)
        f.create_dataset("train/labels", data=train_labels)
        f.create_dataset("test/images", data=test_images)
        f.create_dataset("test/labels", data=test_labels)

        f.attrs["description"] = description
        f.attrs["label_names"] = label_names_np

    print("fashion-mnist.h5 created successfully!")
