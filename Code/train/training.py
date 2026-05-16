import numpy as np
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Code.Import_training_data.import_traing_data import DataLoader
from Code.train.model import Model


class TrainModel:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate

    def _get_weights(self, layer):
        W = np.array([node.weights for node in layer.nodes])
        b = np.array([node.bias    for node in layer.nodes])
        return W, b

    def _set_weights(self, layer, W, b):
        for i, node in enumerate(layer.nodes):
            node.weights = W[i]
            node.bias    = b[i]

    def _relu(self, z):
        return np.maximum(0, z)

    def _softmax(self, z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z   = np.exp(shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _forward(self, X):
        current = X
        self.cache = []

        for layer in self.model.layers:
            W, b = self._get_weights(layer)

            a_in = current
            z    = current @ W.T + b

            if layer.activation_name == 'softmax':
                a_out = self._softmax(z)
            else:
                a_out = self._relu(z)

            self.cache.append((a_in, z, a_out))
            current = a_out

        return current

    def _cross_entropy_loss(self, y_pred, y_true):
        n = len(y_true)
        correct_class_probs = y_pred[np.arange(n), y_true]
        loss = -np.mean(np.log(correct_class_probs + 1e-9))
        return loss

    def _backward(self, y_pred, y_true):
        n = len(y_true)
        gradients = []

        dout = y_pred.copy()
        dout[np.arange(n), y_true] -= 1
        dout /= n

        for i in reversed(range(len(self.model.layers))):
            a_in, z, a_out = self.cache[i]


            dW = dout.T @ a_in
            db = np.sum(dout, axis=0)
            gradients.append((i, dW, db))

            if i > 0:
                W, _ = self._get_weights(self.model.layers[i])
                _, z_prev, _ = self.cache[i - 1]

                dout = (dout @ W) * (z_prev > 0)

        return gradients

    def gradient_descent(self, X, y):
        y_pred = self._forward(X)

        loss = self._cross_entropy_loss(y_pred, y)

        gradients = self._backward(y_pred, y)

        for layer_index, dW, db in gradients:
            layer = self.model.layers[layer_index]
            W, b  = self._get_weights(layer)
            W    -= self.learning_rate * dW
            b    -= self.learning_rate * db
            self._set_weights(layer, W, b)

        return loss

    def accuracy(self, X, y):
        y_pred      = self._forward(X)
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y)

    def train(self, X_train, y_train, X_val, y_val, epochs, verbose=True):
        self.loss_history    = []
        self.val_acc_history = []

        for epoch in range(1, epochs + 1):
            loss = self.gradient_descent(X_train, y_train)

            if epoch % 5 == 0:
                val_acc = self.accuracy(X_val, y_val)
                self.loss_history.append(round(loss, 5))
                self.val_acc_history.append(round(val_acc, 5))
                if verbose:
                    print(f"Epoch {epoch:>4}/{epochs} | Loss: {loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if verbose:
            print("\nTraining complete.")
