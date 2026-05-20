"""
ModelArbiter — ensemble wrapper that averages softmax probabilities from
multiple Model instances to improve prediction accuracy.

Loading modes
-------------
1. Manual / programmatic:
       arbiter.add_model(model)          # add an already-built Model object
       arbiter.load_from_file(path)      # load one  .npz file and add it
       arbiter.load_from_files([p1, p2]) # load many .npz files at once

2. Bayesian optimisation (auto):
       arbiter.run_bayes_opt(
           X_train, y_train, X_val, y_val,
           top_k=3,           # how many best models to keep
           n_iterations=20,   # total BayesOpt trials
           **bayes_kwargs,    # forwarded to BayesOptimizer
       )

Inference
---------
   probs  = arbiter.predict_proba(X)   # averaged softmax vectors  (n, 10)
   labels = arbiter.predict(X)         # argmax of averaged probs  (n,)
   acc    = arbiter.accuracy(X, y)     # fraction of correct labels
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Bootstrap the repo root so relative imports work regardless of cwd
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Code.train.model import Model
from Code.Optymalize_Model_Size.Bayes_optymalization import BayesOptimizer


class ModelArbiter:
    """Ensemble of Model objects that votes by averaging softmax probabilities."""

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------
    def __init__(self, models: list[Model] | None = None):
        """
        Parameters
        ----------
        models : list[Model] | None
            Optional initial list of Model objects.  You can always add more
            later with :meth:`add_model` or the various load helpers.
        """
        self._models: list[Model] = list(models) if models else []

    # -----------------------------------------------------------------------
    # Public properties
    # -----------------------------------------------------------------------
    @property
    def size(self) -> int:
        """Number of models currently in the ensemble."""
        return len(self._models)

    # -----------------------------------------------------------------------
    # Adding models
    # -----------------------------------------------------------------------
    def add_model(self, model: Model) -> None:
        """Add a pre-built Model object to the ensemble."""
        if not isinstance(model, Model):
            raise TypeError(f"Expected a Model instance, got {type(model).__name__}")
        self._models.append(model)
        print(f"[Arbiter] Model added  (ensemble size: {self.size})")

    def load_from_file(self, path: str) -> None:
        """Load a single .npz model file and add it to the ensemble."""
        model = Model.load(path)
        self._models.append(model)
        print(f"[Arbiter] Loaded '{path}'  (ensemble size: {self.size})")

    def load_from_files(self, paths: list[str]) -> None:
        """Load multiple .npz model files and add them all to the ensemble."""
        for p in paths:
            self.load_from_file(p)

    def load_from_directory(self, directory: str, pattern: str = "*.npz") -> None:
        """
        Scan *directory* for files matching *pattern* and load all of them.

        Parameters
        ----------
        directory : str   Path to the folder containing .npz files.
        pattern   : str   Glob pattern (default ``"*.npz"``).
        """
        folder = Path(directory)
        if not folder.is_dir():
            raise NotADirectoryError(f"'{directory}' is not a valid directory")
        files = sorted(folder.glob(pattern))
        if not files:
            print(f"[Arbiter] No files matching '{pattern}' found in '{directory}'")
            return
        for f in files:
            self.load_from_file(str(f))

    # -----------------------------------------------------------------------
    # Bayesian-optimisation helper
    # -----------------------------------------------------------------------
    def run_bayes_opt(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        top_k:          int = 3,
        n_iterations:   int = 20,
        n_random_starts: int | None = None,
        **bayes_kwargs,
    ) -> None:
        """
        Run Bayesian optimisation, then keep the *top_k* best models.

        Parameters
        ----------
        X_train, y_train : training data
        X_val,   y_val   : validation data (used for scoring)
        top_k            : number of best models to add to the ensemble
        n_iterations     : total BayesOpt trials (random + GP-guided)
        n_random_starts  : pure-random trials before GP kicks in
                           (defaults to max(5, n_iterations // 5))
        **bayes_kwargs   : forwarded verbatim to :class:`BayesOptimizer`
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        random_starts = (
            n_random_starts
            if n_random_starts is not None
            else max(5, n_iterations // 5)
        )

        print(f"\n[Arbiter] Starting Bayesian optimisation "
              f"({n_iterations} iterations, keeping top {top_k})…")

        optimizer = BayesOptimizer(
            X_train, y_train,
            X_val,   y_val,
            n_random_starts=random_starts,
            n_iterations=n_iterations,
            **bayes_kwargs,
        )
        optimizer.optimise()

        # Sort all trials by validation accuracy (descending) and keep top_k
        ranked = sorted(optimizer.history, key=lambda h: h['score'], reverse=True)
        selected = ranked[:top_k]

        print(f"\n[Arbiter] Adding top {len(selected)} model(s) to the ensemble:")
        for rank, entry in enumerate(selected, start=1):
            acc = entry['score']
            self._models.append(entry['trainer'].model)
            print(f"  #{rank}  Val accuracy: {acc:.4f}  (ensemble size: {self.size})")

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return the element-wise average of each model's softmax output.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Averaged probability distributions.
        """
        if not self._models:
            raise RuntimeError("No models in the ensemble — add at least one model first")

        # Stack softmax outputs and average across models
        probs = np.stack([m._forward(X) for m in self._models], axis=0)  # (M, N, C)
        return probs.mean(axis=0)                                          # (N, C)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted class label for each sample.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)  — integer class labels
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  — true integer labels

        Returns
        -------
        float   Fraction of correctly classified samples.
        """
        return float(np.mean(self.predict(X) == y))

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------
    def evaluate_all(self, X: np.ndarray, y: np.ndarray) -> None:
        """Print individual and ensemble accuracy for quick comparison."""
        print("\n[Arbiter] Per-model accuracy:")
        for i, m in enumerate(self._models):
            acc = m.accuracy(X, y)
            print(f"  Model {i+1:>2}: {acc:.4f}")
        ensemble_acc = self.accuracy(X, y)
        print(f"  Ensemble : {ensemble_acc:.4f}")

    def clear(self) -> None:
        """Remove all models from the ensemble."""
        self._models.clear()
        print("[Arbiter] Ensemble cleared.")

    def __repr__(self) -> str:
        return f"ModelArbiter(size={self.size})"


# ---------------------------------------------------------------------------
# Quick demo / manual test — run this file directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from Code.Import_training_data.import_traing_data import DataLoader

    X_train, y_train = DataLoader.get_train_data()
    X_val,   y_val   = DataLoader.get_val_data()
    X_test,  y_test  = DataLoader.get_test_data()

    arbiter = ModelArbiter()

    print("=" * 55)
    print("  ModelArbiter demo")
    print("=" * 55)
    print("\nChoose how to populate the ensemble:")
    print("  1 - Load .npz files manually (paths typed interactively)")
    print("  2 - Load all .npz files from the saved_models directory")
    print("  3 - Run Bayesian optimisation and pick the top-k models")
    print("  4 - All of the above")
    choice = input("Choice [1/2/3]: ").strip()

    if choice in ("1", "4"):
        raw = input("Enter .npz file path(s), space-separated: ").strip()
        if raw:
            arbiter.load_from_files(raw.split())

    if choice in ("2", "4"):
        models_dir = ROOT_DIR / "saved_models"
        arbiter.load_from_directory(str(models_dir))

    if choice in ("3", "4"):
        raw_k    = input("top_k models to keep  [3]: ").strip()
        raw_iter = input("BayesOpt iterations  [20]: ").strip()
        top_k    = int(raw_k)    if raw_k    else 3
        n_iter   = int(raw_iter) if raw_iter else 20
        arbiter.run_bayes_opt(
            X_train, y_train,
            X_val,   y_val,
            top_k=top_k,
            n_iterations=n_iter,
        )

    if arbiter.size == 0:
        print("[Arbiter] No models loaded — nothing to evaluate.")
    else:
        arbiter.evaluate_all(X_test, y_test)
