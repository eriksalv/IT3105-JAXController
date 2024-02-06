from .controller import Controller
import jax.numpy as jnp
from jax import tree_util
import matplotlib.pyplot as plt
import numpy as np


class PIDController(Controller):
    def gen_params(self):
        kp = np.random.uniform(-1, 1)
        ki = np.random.uniform(-1, 1)
        kd = np.random.uniform(-1, 1)

        self.param_history = []
        self.param_history.append([kp, ki, kd])
        params = (kp, ki, kd)

        self.treedef = tree_util.tree_structure(params)
        return params

    def initialize(self, params):
        self.kp = params[0]
        self.ki = params[1]
        self.kd = params[2]

    def calculate_control_value(self, error) -> float:
        super().calculate_control_value(error)
        return self.kp * error + self.ki * self.integral() + self.kd * self.derivative()

    def update_params(self, params, lrate, gradients):
        kp, ki, kd = params
        kp_new = kp - lrate * gradients[0]
        ki_new = ki - lrate * gradients[1]
        kd_new = kd - lrate * gradients[2]
        self.param_history.append([kp_new, ki_new, kd_new])
        return kp_new, ki_new, kd_new

    def plot_params(self):
        """Plots parameter history for kp, ki, kd
        """
        param_history_array = np.array(self.param_history)

        plt.plot(param_history_array[:, 0], label='kp')
        plt.plot(param_history_array[:, 1], label='ki')
        plt.plot(param_history_array[:, 2], label='kd')

        plt.xlabel('Epoch')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    con = PIDController()
    print(con.calculate_control_value(5))
    print(con.calculate_control_value(2))
