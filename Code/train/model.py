class Node:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = [0.0] * input_size  # Inicjalizacja wag na 0.0
        self.bias = 0.0  # Inicjalizacja biasu na 0.0
class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = [Node(input_size) for _ in range(output_size)]  # Tworzenie warstwy z określoną liczbą neuronów