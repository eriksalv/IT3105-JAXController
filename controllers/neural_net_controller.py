from controller import Controller
from util import sigmoid, ReLU, tanh
import jax.numpy as jnp
import numpy as np


class NNController(Controller):
    def __init__(self, hidden_layers=[10, 5], activation_funcs=[ReLU, sigmoid, tanh]) -> None:
        super().__init__()
        assert len(activation_funcs) == len(hidden_layers) + 1
        inputs = 3
        layers = hidden_layers + [1]  # add output layer
        self.params = []
        self.activation_funcs = activation_funcs

        for outputs in layers:
            weights = np.random.uniform(-.1, .1, (inputs, outputs))
            biases = np.random.uniform(-.1, .1, (1, outputs))
            inputs = outputs
            self.params.append((weights, biases))

    def calculate_control_value(self, error):
        super().calculate_control_value(error)
        activations = jnp.array([error, self.integral(), self.derivative()])
        for (weights, biases), activation in zip(self.params, self.activation_funcs):
            activations = activation(jnp.dot(activations, weights) + biases)
        return activations[0]


if __name__ == '__main__':
    con = NNController()
    print(con.calculate_control_value(5))
