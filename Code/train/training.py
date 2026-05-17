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
