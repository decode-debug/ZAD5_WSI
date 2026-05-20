import numpy as np
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Code.Import_training_data.import_traing_data import DataLoader
from Code.train.model import Model


class TrainModel:
    def __init__(self, model: Model, learning_rate=0.01):
        """"Trainer that implements forward and backward passes, and updates weights."""
        self.model = model
        self.learning_rate = learning_rate

    def _cross_entropy_loss(self, y_pred, y_true):
        n = len(y_true)
        correct_class_probs = y_pred[np.arange(n), y_true]
        loss = -np.mean(np.log(correct_class_probs + 1e-9))
        return loss

    def _backward(self, y_pred, y_true):
        """Compute gradients for all layers using backpropagation."""
        n = len(y_true)
        gradients = []

        # Initialize the error for the output layer
        error = y_pred.copy()
        error[np.arange(n), y_true] -= 1
        error /= n

        # Backpropagate through layers in reverse order
        for i in reversed(range(len(self.model.layers))):
            a_in, z, a_out = self.model.cache[i]

            # Compute gradients for weights and biases
            dW = error.T @ a_in
            db = np.sum(error, axis=0)
            gradients.append((i, dW, db))

            # Update error for the next layer down (if not at input layer)
            if i > 0:
                W, _ = self.model.layers[i]._get_weights()
                _, z_prev, _ = self.model.cache[i - 1]

                error = (error @ W) * (z_prev > 0)

        return gradients

    def gradient_descent(self, X, y):
        """Perform one step of gradient descent and return the loss."""
        # Forward pass to get predictions
        y_pred = self.model._forward(X)

        # Compute loss
        loss = self._cross_entropy_loss(y_pred, y)

        # Backward pass to get gradients
        gradients = self._backward(y_pred, y)

        # Update weights for each layer
        for layer_index, dW, db in gradients:
            layer = self.model.layers[layer_index]
            W, b  = layer._get_weights()
            W    -= self.learning_rate * dW
            b    -= self.learning_rate * db
            layer._set_weights(W, b)

        return loss

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size=32, verbose=True):
        """
        Train the model for a specified number of epochs using Mini-Batch Gradient Descent.
        If batch_size is None, it defaults to Full-Batch Gradient Descent.
        """
        self.loss_history    = []
        self.val_acc_history = []
        num_samples = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # 1. Shuffle data at the beginning of each epoch to avoid ordering bias
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses = []

            # If batch_size is None, process the whole dataset at once (Full-Batch)
            current_batch_size = batch_size if batch_size is not None else num_samples

            # 2. Loop over mini-batches
            for i in range(0, num_samples, current_batch_size):
                X_batch = X_shuffled[i : i + current_batch_size]
                y_batch = y_shuffled[i : i + current_batch_size]

                # Perform a single weight update step based on this batch
                batch_loss = self.gradient_descent(X_batch, y_batch)
                epoch_losses.append(batch_loss)

            # Compute the average loss for the entire epoch
            epoch_loss = np.mean(epoch_losses)

            # 3. Validation and logging every 5 epochs
            if epoch % 5 == 0:
                val_acc = self.model.accuracy(X_val, y_val)
                self.loss_history.append(round(epoch_loss, 5))
                self.val_acc_history.append(round(val_acc, 5))
                if verbose:
                    print(f"Epoch {epoch:>4}/{epochs} | Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if verbose:
            print("\nTraining complete.")


class QuasiNewtonTrainer:
    """
    Trains a Model using the L-BFGS-B quasi-Newton optimizer (via scipy).

    L-BFGS builds a low-rank approximation of the inverse Hessian from the
    last m gradient differences, which gives super-linear convergence on smooth
    objectives — typically far fewer iterations than plain SGD/mini-batch GD.

    Usage
    -----
        trainer = QuasiNewtonTrainer(model)
        trainer.train(X_train, y_train, X_val, y_val, max_iter=300)
        acc = trainer.model.accuracy(X_test, y_test)

    The attributes ``loss_history`` and ``val_acc_history`` are filled at every
    L-BFGS callback step, so Plotly visualisations work identically to
    ``TrainModel``.
    """

    def __init__(self, model: Model):
        self.model           = model
        self.loss_history    = []
        self.val_acc_history = []

    # ------------------------------------------------------------------
    # Weight vector <-> model parameter conversion
    # ------------------------------------------------------------------

    def _get_flat_weights(self) -> np.ndarray:
        """Concatenate all W and b arrays into a single 1-D float64 vector."""
        parts = []
        for layer in self.model.layers:
            W, b = layer._get_weights()
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts).astype(np.float64)

    def _set_flat_weights(self, flat: np.ndarray):
        """Write a flat parameter vector back into the model layers."""
        idx = 0
        for layer in self.model.layers:
            W, b = layer._get_weights()
            nW, nb = W.size, b.size
            layer._set_weights(
                flat[idx        : idx + nW     ].reshape(W.shape),
                flat[idx + nW   : idx + nW + nb].reshape(b.shape),
            )
            idx += nW + nb

    # ------------------------------------------------------------------
    # Loss + gradient as a single function (required by scipy.optimize)
    # ------------------------------------------------------------------

    def _loss_and_grad(self, flat: np.ndarray, X: np.ndarray, y: np.ndarray):
        """Return (scalar loss, flat gradient vector) for the given weight vector."""
        self._set_flat_weights(flat)

        # Forward pass
        y_pred = self.model._forward(X)

        # Cross-entropy loss
        n = len(y)
        correct_probs = y_pred[np.arange(n), y]
        loss = float(-np.mean(np.log(correct_probs + 1e-9)))

        # Backward pass — same logic as TrainModel._backward
        error = y_pred.copy()
        error[np.arange(n), y] -= 1
        error /= n

        grad_parts = [None] * len(self.model.layers)
        for i in reversed(range(len(self.model.layers))):
            a_in, z, a_out = self.model.cache[i]
            dW = (error.T @ a_in).ravel()
            db = np.sum(error, axis=0).ravel()
            grad_parts[i] = np.concatenate([dW, db])
            if i > 0:
                W, _ = self.model.layers[i]._get_weights()
                _, z_prev, _ = self.model.cache[i - 1]
                error = (error @ W) * (z_prev > 0)

        grad_flat = np.concatenate(grad_parts).astype(np.float64)
        return loss, grad_flat

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train, y_train, X_val, y_val, max_iter=300, verbose=True):
        """
        Train using the full-batch L-BFGS-B quasi-Newton algorithm.

        Parameters
        ----------
        max_iter : int
            Maximum number of L-BFGS-B iterations.
        verbose : bool
            Print result after optimisation.
        """
        from scipy.optimize import minimize

        self.loss_history    = []
        self.val_acc_history = []

        x0 = self._get_flat_weights()
        result = minimize(
            fun=self._loss_and_grad,
            x0=x0,
            args=(X_train, y_train),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-7},
        )
        self._set_flat_weights(result.x)

        # Record a single history entry for compatibility with Plotly visualisations
        y_pred = self.model._forward(X_train)
        n      = len(y_train)
        loss   = float(-np.mean(np.log(y_pred[np.arange(n), y_train] + 1e-9)))
        val_acc = float(self.model.accuracy(X_val, y_val))
        self.loss_history.append(round(loss, 5))
        self.val_acc_history.append(round(val_acc, 5))

        if verbose:
            print(f"L-BFGS-B: {result.message}")
            print(f"Iterations : {result.nit}  |  Loss: {loss:.4f}  |  Val Accuracy: {val_acc:.4f}")
            print("Training complete.")
