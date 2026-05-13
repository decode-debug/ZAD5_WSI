import numpy as np

class Node:
    """Reprezentuje pojedynczy neuron i przechowuje jego stan."""
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs

        # Stan neuronu: wektor wag (jeden dla każdego wejścia) i pojedynczy bias
        self.weights = np.random.randn(num_inputs)
        self.bias = 0.0

    def __repr__(self):
        return f"Node(inputs={self.num_inputs})"


class Layer:
    """Przechowuje informacje o całej warstwie oraz listę jej neuronów."""
    def __init__(self, num_nodes, num_inputs, activation_name):
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.activation_name = activation_name

        # Warstwa przechowuje po prostu listę obiektów Node
        self.nodes = [Node(num_inputs) for _ in range(num_nodes)]

    def __repr__(self):
        return f"Layer(nodes={self.num_nodes}, inputs={self.num_inputs}, activation='{self.activation_name}')"


class Model:
    """Spina wszystko w całość, przechowując architekturę sieci."""
    def __init__(self, layer_sizes, activations):
        # layer_sizes zawiera m.in. rozmiar wejścia, więc warstw jest o 1 mniej
        self.num_layers = len(layer_sizes) - 1
        self.layers = []

        # Budowanie struktury warstwa po warstwie
        for i in range(self.num_layers):
            num_inputs = layer_sizes[i]      # Rozmiar wyjścia poprzedniej warstwy
            num_nodes = layer_sizes[i+1]     # Rozmiar obecnej warstwy
            activation = activations[i]      # Nazwa funkcji aktywacji

            # Tworzymy i zapisujemy warstwę
            layer = Layer(num_nodes, num_inputs, activation)
            self.layers.append(layer)

    def summary(self):
        """Wypisuje zebrane informacje o całej zbudowanej sieci."""
        print("=== Struktura Modelu ===")
        print(f"Całkowita liczba warstw (bez wejściowej): {self.num_layers}")

        for i, layer in enumerate(self.layers):
            print(f"  Warstwa {i+1}:")
            print(f"    - Liczba neuronów (nodes): {layer.num_nodes}")
            print(f"    - Liczba wejść do każdego neuronu: {layer.num_inputs}")
            print(f"    - Funkcja aktywacji: {layer.activation_name}")
        print("========================")

