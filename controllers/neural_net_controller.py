from controller import Controller
from util import sigmoid
import numpy as np


class NNController(Controller):
    def __init__(self, hidden_layers=[10, 5]) -> None:
        super().__init__()
        inputs = 3
        self.params = []
        layers = hidden_layers + [1]  # add output layer

        for outputs in layers:
            weights = np.random.uniform(-.1, .1, (inputs, outputs))
            biases = np.random.uniform(-.1, .1, (1, outputs))
            inputs = outputs
            self.params.append((weights, biases))

    def calculate_control_value(self, error):
        super().calculate_control_value(error)
        activations = np.array([error, self.integral(), self.derivative()])
        for weights, biases in self.params:
            # TODO: don't hard code sigmoid, accept other activation functions
            activations = sigmoid(np.dot(activations, weights) + biases)
        return activations[0]


if __name__ == '__main__':
    con = NNController()
    print(con.calculate_control_value(5))
