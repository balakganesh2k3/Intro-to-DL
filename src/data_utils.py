"""
data_utils.py

Utility functions for loading the simplified MNIST dataset used in
Assignment 0 (16x16 pixel images, values in [-1, 1], digits 0-9).
"""

import os
import numpy as np


def load_data(data_dir="../data"):
    """
    Load train/test input and output CSV files from `data_dir`.

    Expects the following files inside data_dir:
        train_in.csv   (or train_in - Copy.csv)
        train_out.csv  (or train_out - Copy.csv)
        test_in.csv    (or test_in - Copy.csv)
        test_out.csv   (or test_out - Copy.csv)

    Each *_in.csv file has one image per row, 256 comma-separated
    float values (16x16 pixels flattened), no header row.

    Each *_out.csv file has one integer label (0-9) per row, no header.

    Returns
    -------
    train_in : np.ndarray, shape (1707, 256)
    train_out : np.ndarray, shape (1707,)
    test_in : np.ndarray, shape (1000, 256)
    test_out : np.ndarray, shape (1000,)
    """

    def _find(name_options):
        """Return the first existing path among a list of candidate filenames."""
        for name in name_options:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"None of {name_options} found in '{data_dir}'. "
            f"Files present: {os.listdir(data_dir) if os.path.isdir(data_dir) else 'DIR NOT FOUND'}"
        )

    train_in_path = _find(["train_in.csv", "train_in - Copy.csv", "train_in_-_Copy.csv"])
    train_out_path = _find(["train_out.csv", "train_out - Copy.csv", "train_out_-_Copy.csv"])
    test_in_path = _find(["test_in.csv", "test_in - Copy.csv", "test_in_-_Copy.csv"])
    test_out_path = _find(["test_out.csv", "test_out - Copy.csv", "test_out_-_Copy.csv"])

    train_in = np.loadtxt(train_in_path, delimiter=",")
    test_in = np.loadtxt(test_in_path, delimiter=",")

    # out files: one integer per line -> load as 1D int array
    train_out = np.loadtxt(train_out_path, delimiter=",").astype(int)
    test_out = np.loadtxt(test_out_path, delimiter=",").astype(int)

    return train_in, train_out, test_in, test_out


def add_bias_column(X):
    """
    Prepend a column of ones to X, to represent the bias term.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray, shape (n_samples, n_features + 1)
        Column of 1's is placed as the FIRST column (index 0),
        so that if X was (256,) per row, output is (257,)
        with X_aug[:, 0] == 1.
    """
    n_samples = X.shape[0]
    ones = np.ones((n_samples, 1))
    return np.hstack([ones, X])


if __name__ == "__main__":
    # quick self-test
    train_in, train_out, test_in, test_out = load_data(data_dir="../data")
    print("train_in :", train_in.shape)
    print("train_out:", train_out.shape, "unique labels:", np.unique(train_out))
    print("test_in  :", test_in.shape)
    print("test_out :", test_out.shape, "unique labels:", np.unique(test_out))

    train_in_aug = add_bias_column(train_in)
    print("train_in with bias:", train_in_aug.shape)
    assert np.all(train_in_aug[:, 0] == 1), "bias column not all ones!"
    print("Bias column check passed.")