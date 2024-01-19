from .controller import Controller
from .util import sigmoid, ReLU, tanh
import jax.numpy as jnp
from jax import tree_util
import numpy as np


class NNController(Controller):
    def __init__(self, hidden_layers=[5], activation_funcs=[ReLU, ReLU]) -> None:
        super().__init__()
        assert len(activation_funcs) == len(hidden_layers) + 1
        self.hidden_layers = hidden_layers
        self.activation_funcs = activation_funcs

    def gen_params(self):
        inputs = 3
        layers = self.hidden_layers + [1]  # add output layer
        params = []

        for outputs in layers:
            weights = np.random.uniform(-.1, .1, (inputs, outputs))
            biases = np.random.uniform(-.1, .1, (1, outputs))
            inputs = outputs
            params.append((weights, biases))

        self.treedef = tree_util.tree_structure(params)

        return params

    def initialize(self, params):
        self.params = params

    def calculate_control_value(self, error):
        super().calculate_control_value(error)
        activations = jnp.array([error, self.integral(), self.derivative()])
        for wb, activation in zip(self.params, self.activation_funcs):
            weights, biases = wb
            activations = activation(jnp.dot(activations, weights) + biases)
        return activations[0, 0]

    def update_params(self, params, lrate, gradients):
        updated_params = []
        for param, gradient in zip(params, gradients):
            updated_params.append(param - lrate * gradient)
        return tree_util.tree_unflatten(self.treedef, updated_params)


if __name__ == '__main__':
    con = NNController()
    print(con.calculate_control_value(5))
