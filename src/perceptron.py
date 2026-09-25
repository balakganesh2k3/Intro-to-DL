"""
Multi-class perceptron: 10 independent linear classifiers sharing one
weight matrix W of shape (num_features, num_classes).

Forward pass is a single matrix multiplication:
    scores = T @ W        # (N, num_features) @ (num_features, num_classes)
    preds  = argmax(scores, axis=1)
"""

import numpy as np


class MultiClassPerceptron:
    def __init__(self, num_features=257, num_classes=10, init="small_random", seed=None):
        rng = np.random.default_rng(seed)
        if init == "zeros":
            self.W = np.zeros((num_features, num_classes))
        elif init == "small_random":
            self.W = rng.normal(0, 0.01, size=(num_features, num_classes))
        elif init == "large_random":
            self.W = rng.normal(0, 1.0, size=(num_features, num_classes))
        else:
            raise ValueError(f"Unknown init strategy: {init}")

    def forward(self, T):
        """T: (N, num_features) -> scores: (N, num_classes)"""
        return T @ self.W

    def predict(self, T):
        """Return predicted class index for each row of T."""
        return np.argmax(self.forward(T), axis=1)

    def accuracy(self, T, labels):
        """labels: (N,) integer class indices."""
        preds = self.predict(T)
        return np.mean(preds == labels)

    def train_epoch(self, T, labels, lr=0.1, shuffle=True, rng=None):
        """One pass over the training data, updating W on every mistake.

        T:      (N, num_features), bias column already included
        labels: (N,) integer class indices in [0, num_classes)
        lr:     learning rate
        """
        n = T.shape[0]
        indices = np.arange(n)
        if shuffle:
            rng = rng or np.random.default_rng()
            rng.shuffle(indices)

        mistakes = 0
        for i in indices:
            x = T[i]
            true_label = labels[i]
            scores = x @ self.W          # (num_classes,)
            pred_label = np.argmax(scores)

            if pred_label != true_label:
                mistakes += 1
                self.W[:, true_label] += lr * x
                self.W[:, pred_label] -= lr * x

        return mistakes / n  # epoch error rate