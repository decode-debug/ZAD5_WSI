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

    def __repr__(self):
        return f"Layer(nodes={self.num_nodes}, inputs={self.num_inputs}, activation='{self.activation_name}')"


class Model:
    """Ties everything together by storing the network architecture."""
    def __init__(self, layer_sizes, activations):
        # layer_sizes includes the input size, so there is one fewer layer than entries
        self.num_layers = len(layer_sizes) - 1
        self.layers = []

        # Build the structure layer by layer
        for i in range(self.num_layers):
            num_inputs = layer_sizes[i]      # Output size of the previous layer
            num_nodes = layer_sizes[i+1]     # Size of the current layer
            activation = activations[i]      # Activation function name

            # Create and store the layer
            layer = Layer(num_nodes, num_inputs, activation)
            self.layers.append(layer)

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

