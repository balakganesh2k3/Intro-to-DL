"""
Training loop: trains a MultiClassPerceptron for a number of epochs,
tracking train/test accuracy after each epoch.
"""

import numpy as np
from perceptron import MultiClassPerceptron


def train(T_train, y_train, T_test, y_test, num_epochs=20, lr=0.1,
          init="small_random", seed=None):
    """Train a perceptron and return the model plus per-epoch history.

    Returns:
        model: trained MultiClassPerceptron
        history: dict with keys 'train_acc', 'test_acc', 'train_error'
    """
    model = MultiClassPerceptron(num_features=T_train.shape[1], init=init, seed=seed)
    rng = np.random.default_rng(seed)

    history = {"train_acc": [], "test_acc": [], "train_error": []}

    for epoch in range(num_epochs):
        error_rate = model.train_epoch(T_train, y_train, lr=lr, rng=rng)

        train_acc = model.accuracy(T_train, y_train)
        test_acc = model.accuracy(T_test, y_test)

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["train_error"].append(error_rate)

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
              f"train_error_rate={error_rate:.4f}")

    return model, history


def run_multiple_seeds(T_train, y_train, T_test, y_test, seeds,
                        num_epochs=20, lr=0.1, init="small_random"):
    """Repeat training across several seeds; return final accuracies."""
    results = []
    for seed in seeds:
        model, history = train(T_train, y_train, T_test, y_test,
                                num_epochs=num_epochs, lr=lr, init=init, seed=seed)
        results.append({
            "seed": seed,
            "init": init,
            "lr": lr,
            "final_train_acc": history["train_acc"][-1],
            "final_test_acc": history["test_acc"][-1],
        })
    return results