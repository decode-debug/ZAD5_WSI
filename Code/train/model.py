import numpy as np

class Node:
    """Represents a single neuron and stores its state."""
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs

        # Neuron state: weight vector (one per input) and a single bias
        self.weights = np.random.randn(num_inputs)
        self.bias = 0.0

    def __repr__(self):
        return f"Node(inputs={self.num_inputs})"


class Layer:
    """Stores information about an entire layer and its list of neurons."""
    def __init__(self, num_nodes, num_inputs, activation_name):
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.activation_name = activation_name

        # A layer simply holds a list of Node objects
        self.nodes = [Node(num_inputs) for _ in range(num_nodes)]

    def _get_weights(self):
        """Extracts the weight matrix and bias vector for the entire layer."""
        W = np.array([node.weights for node in self.nodes])
        b = np.array([node.bias    for node in self.nodes])
        return W, b

    def _set_weights(self, W, b):
        """Sets the weight matrix and bias vector for the entire layer."""
        for i, node in enumerate(self.nodes):
            node.weights = W[i]
            node.bias    = b[i]

    def __repr__(self):
        return f"Layer(nodes={self.num_nodes}, inputs={self.num_inputs}, activation='{self.activation_name}')"

class Model:
    """Ties everything together by storing the network architecture."""
    def __init__(self, layer_sizes: list[int], activations: list[str]):
        """Initializes the model architecture based on the provided layer sizes and activation functions."""
        self.num_layers = len(layer_sizes) - 1
        self.layers = []
        self.cache = None  # Will be used to store intermediate values during forward pass for backpropagation

        # Build the structure layer by layer
        for i in range(self.num_layers):
            num_inputs = layer_sizes[i]
            num_nodes = layer_sizes[i+1]
            activation = activations[i]

            # Create and store the layer
            layer = Layer(num_nodes, num_inputs, activation)
            self.layers.append(layer)

    def _relu(self, z):
        """ReLU activation function: f(z) = max(0, z)"""
        return np.maximum(0, z)

    def _sigmoid(self, z):
        """Sigmoid activation function: f(z) = 1 / (1 + exp(-z))"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _softmax(self, z):
        """Softmax activation function for classification."""
        # Shift the input to prevent numerical overflow
        # z Matrix - z_maximum Vector - Broadcasting
        shifted = z - np.max(z, axis=1, keepdims=True)
        # Compute exponentials and normalize to get probabilities
        exp_z   = np.exp(shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _forward(self, X):
        """Performs a forward pass through the network and stores intermediate values for backpropagation."""
        A = X  # X becomes the input to the first layer
        self.cache = []

        for layer in self.layers:
            W, b = layer._get_weights()

            A_prev = A  # Save the input to this layer for backpropagation

            # 1. Linear equation: Z = A_prev * W^T + b
            Z = A_prev @ W.T + b

            # 2. Activation equation (non-linear): A = g(Z)
            if layer.activation_name == 'softmax':
                A = self._softmax(Z)
            elif layer.activation_name == 'sigmoid':
                A = self._sigmoid(Z)
            else:
                A = self._relu(Z)

            # Save the mathematical steps to memory: input (A_prev), raw state (Z), and output (A)
            self.cache.append((A_prev, Z, A))

        return A  # Return the final activation matrix from the last layer

    def accuracy(self, X, y):
        y_pred      = self._forward(X)
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y)

    # -------------------------------------------------------------------
    # Serialisation
    # -------------------------------------------------------------------
    def save(self, path: str):
        """
        Save the model weights, biases, and architecture to a .npz file.

        Parameters
        ----------
        path : str   File path (with or without the .npz extension).
        """
        arrays = {}
        layer_sizes  = [self.layers[0].num_inputs]
        activations  = []

        for i, layer in enumerate(self.layers):
            W, b = layer._get_weights()
            arrays[f'W_{i}'] = W
            arrays[f'b_{i}'] = b
            layer_sizes.append(layer.num_nodes)
            activations.append(layer.activation_name)

        arrays['layer_sizes'] = np.array(layer_sizes, dtype=np.int32)
        arrays['activations'] = np.array(activations, dtype=object)

        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> 'Model':
        """
        Load a model that was previously saved with :meth:`save`.

        Parameters
        ----------
        path : str   Path to the .npz file (extension optional).

        Returns
        -------
        Model
        """
        if not path.endswith('.npz'):
            path = path + '.npz'

        data        = np.load(path, allow_pickle=True)
        layer_sizes = data['layer_sizes'].tolist()
        activations = data['activations'].tolist()

        model = cls(layer_sizes, activations)
        for i, layer in enumerate(model.layers):
            layer._set_weights(data[f'W_{i}'], data[f'b_{i}'])

        return model

    def summary(self):
        """Prints a summary of the entire network architecture."""
        print("=== Model Summary ===")
        print(f"Total number of layers (excluding input): {self.num_layers}")

        for i, layer in enumerate(self.layers):
            print(f"  Layer {i+1}:")
            print(f"    - Number of neurons (nodes): {layer.num_nodes}")
            print(f"    - Number of inputs per neuron: {layer.num_inputs}")
            print(f"    - Activation function: {layer.activation_name}")
        print("=====================")

