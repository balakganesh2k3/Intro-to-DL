"""
task2_perceptron.py

Introduction to Deep Learning - Assignment 0, Task 2
Multi-class perceptron trained from scratch on the 16x16 MNIST-like dataset.

Run from the `notebooks` folder (or adjust DATA_DIR below):
    python task2_perceptron.py

Everything needed lives in this single file:
    - data loading
    - bias-augmentation
    - multi-class perceptron (train / predict)
    - epoch-by-epoch loss & accuracy tracking (train + test)
    - multi-seed reliability experiment
    - learning rate / init strategy comparison
    - plots saved to ./results/
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_DIR = "../data"       # folder containing the csv files
RESULTS_DIR = "../results"  # where plots get saved
NUM_CLASSES = 10
NUM_EPOCHS = 20
DEFAULT_LR = 0.1
DEFAULT_INIT = "small_random"
DEFAULT_SEED = 0
NUM_RELIABILITY_RUNS = 5


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def load_data(data_dir=DATA_DIR):
    """Load train/test input & output csv files, handling '- Copy' filenames too."""

    def _find(name_options):
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
    train_out = np.loadtxt(train_out_path, delimiter=",").astype(int)
    test_out = np.loadtxt(test_out_path, delimiter=",").astype(int)

    return train_in, train_out, test_in, test_out


def add_bias_column(X):
    """Prepend a column of ones to X (bias term). (N, D) -> (N, D+1)."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


# --------------------------------------------------------------------------
# 2. Multi-class perceptron
# --------------------------------------------------------------------------
def init_weights(n_features, n_classes, strategy="small_random", seed=0):
    """
    strategy:
        'zeros'         -> all-zero weights
        'small_random'  -> N(0, 0.01) per weight
        'large_random'  -> N(0, 1) per weight
        'uniform'       -> U(-1, 1) per weight
    """
    rng = np.random.default_rng(seed)
    if strategy == "zeros":
        return np.zeros((n_features, n_classes))
    elif strategy == "small_random":
        return rng.normal(loc=0.0, scale=0.01, size=(n_features, n_classes))
    elif strategy == "large_random":
        return rng.normal(loc=0.0, scale=1.0, size=(n_features, n_classes))
    elif strategy == "uniform":
        return rng.uniform(-1.0, 1.0, size=(n_features, n_classes))
    else:
        raise ValueError(f"Unknown init strategy: {strategy}")


def predict(T, W):
    """
    T: (N, D+1) augmented inputs
    W: (D+1, C) weight matrix
    Returns predicted class indices, shape (N,)
    """
    scores = T @ W           # (N, C)
    return np.argmax(scores, axis=1)


def accuracy(T, y, W):
    preds = predict(T, W)
    return np.mean(preds == y)


def loss_fn(T, y, W):
    """Simple perceptron-style loss: fraction of misclassified samples."""
    preds = predict(T, W)
    return np.mean(preds != y)


def train_one_epoch(T, y, W, lr):
    """
    One pass over the training set with the standard multi-class
    perceptron update rule:
        if predicted_label != true_label:
            W[:, true_label]      += lr * x
            W[:, predicted_label] -= lr * x
    (Chapter 6.1, UDL, eq. 6.3 / 6.7 — the "reward correct / punish wrong" rule.)
    """
    n_samples = T.shape[0]
    for i in range(n_samples):
        x = T[i]
        true_label = y[i]
        scores = x @ W
        pred_label = np.argmax(scores)
        if pred_label != true_label:
            W[:, true_label] += lr * x
            W[:, pred_label] -= lr * x
    return W


def train(T_train, y_train, T_test, y_test,
          num_epochs=NUM_EPOCHS, lr=DEFAULT_LR,
          init="small_random", seed=DEFAULT_SEED, verbose=True):
    """
    Train the multi-class perceptron and track loss/accuracy per epoch
    on both train and test sets.

    Returns
    -------
    W : trained weight matrix, shape (D+1, C)
    history : dict with keys 'train_acc', 'test_acc', 'train_loss', 'test_loss'
              each a list of length num_epochs
    """
    n_features = T_train.shape[1]
    W = init_weights(n_features, NUM_CLASSES, strategy=init, seed=seed)

    history = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

    for epoch in range(1, num_epochs + 1):
        W = train_one_epoch(T_train, y_train, W, lr)

        tr_acc = accuracy(T_train, y_train, W)
        te_acc = accuracy(T_test, y_test, W)
        tr_loss = loss_fn(T_train, y_train, W)
        te_loss = loss_fn(T_test, y_test, W)

        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)

        if verbose:
            print(f"Epoch {epoch:2d}/{num_epochs} | "
                  f"train_acc={tr_acc:.4f} test_acc={te_acc:.4f} | "
                  f"train_loss={tr_loss:.4f} test_loss={te_loss:.4f}")

        # perceptron on linearly separable data converges to 0 training error
        if tr_loss == 0.0:
            if verbose:
                print(f"Converged (0 training error) at epoch {epoch}.")
            break

    return W, history


# --------------------------------------------------------------------------
# 3. Reliability across multiple random seeds
# --------------------------------------------------------------------------
def run_multiple_seeds(T_train, y_train, T_test, y_test,
                        num_runs=NUM_RELIABILITY_RUNS,
                        num_epochs=NUM_EPOCHS, lr=DEFAULT_LR, init="small_random"):
    """Run training with several seeds, report mean/std of final test accuracy."""
    final_accs = []
    for seed in range(num_runs):
        _, history = train(T_train, y_train, T_test, y_test,
                            num_epochs=num_epochs, lr=lr, init=init,
                            seed=seed, verbose=False)
        final_accs.append(history["test_acc"][-1])
        print(f"  seed={seed}: final test_acc={history['test_acc'][-1]:.4f}")

    final_accs = np.array(final_accs)
    print(f"\nOver {num_runs} seeds: mean test_acc={final_accs.mean():.4f}, "
          f"std={final_accs.std():.4f}")
    return final_accs


# --------------------------------------------------------------------------
# 4. Hyperparameter comparison (learning rate x init strategy)
# --------------------------------------------------------------------------
def compare_hyperparameters(T_train, y_train, T_test, y_test,
                             learning_rates=(0.001, 0.01, 0.1, 1.0),
                             init_strategies=("zeros", "small_random", "large_random"),
                             num_epochs=NUM_EPOCHS, seed=DEFAULT_SEED):
    """Grid comparison; prints a small results table and returns it as a list of dicts."""
    results = []
    for init in init_strategies:
        for lr in learning_rates:
            _, history = train(T_train, y_train, T_test, y_test,
                                num_epochs=num_epochs, lr=lr, init=init,
                                seed=seed, verbose=False)
            results.append({
                "init": init,
                "lr": lr,
                "final_train_acc": history["train_acc"][-1],
                "final_test_acc": history["test_acc"][-1],
                "epochs_run": len(history["train_acc"]),
            })

    print(f"\n{'init':<14}{'lr':<8}{'train_acc':<12}{'test_acc':<12}{'epochs'}")
    for r in results:
        print(f"{r['init']:<14}{r['lr']:<8}{r['final_train_acc']:<12.4f}"
              f"{r['final_test_acc']:<12.4f}{r['epochs_run']}")
    return results


# --------------------------------------------------------------------------
# 5. Plotting
# --------------------------------------------------------------------------
def plot_history(history, out_path=None, title="Multi-class perceptron"):
    epochs = range(1, len(history["train_acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_acc"], label="Train accuracy")
    axes[0].plot(epochs, history["test_acc"], label="Test accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{title} — Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], label="Train loss")
    axes[1].plot(epochs, history["test_loss"], label="Test loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (error rate)")
    axes[1].set_title(f"{title} — Loss")
    axes[1].legend()

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    plt.show()


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    train_in, train_out, test_in, test_out = load_data(DATA_DIR)
    T_train = add_bias_column(train_in)
    T_test = add_bias_column(test_in)
    print("Train inputs:", T_train.shape, "Test inputs:", T_test.shape)

    # --- Single training run with default hyperparameters -----------------
    print("\n=== Single run (default hyperparameters) ===")
    W, history = train(T_train, train_out, T_test, test_out,
                        num_epochs=NUM_EPOCHS, lr=DEFAULT_LR,
                        init=DEFAULT_INIT, seed=DEFAULT_SEED, verbose=True)
    plot_history(history, out_path=os.path.join(RESULTS_DIR, "task2_accuracy_loss.png"))

    # --- Reliability across seeds ------------------------------------------
    print("\n=== Reliability across multiple seeds ===")
    run_multiple_seeds(T_train, train_out, T_test, test_out,
                        num_runs=NUM_RELIABILITY_RUNS,
                        num_epochs=NUM_EPOCHS, lr=DEFAULT_LR, init=DEFAULT_INIT)

    # --- Hyperparameter comparison ------------------------------------------
    print("\n=== Learning rate / init strategy comparison ===")
    compare_hyperparameters(T_train, train_out, T_test, test_out,
                             learning_rates=(0.001, 0.01, 0.1, 1.0),
                             init_strategies=("zeros", "small_random", "large_random"),
                             num_epochs=NUM_EPOCHS, seed=DEFAULT_SEED)

    print("\nDone. Compare these numbers against your Task 1 "
          "nearest-mean / KNN accuracies in the report.")