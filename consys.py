import json
from controllers.controller import Controller
from controllers.util import ReLU, linear, sigmoid, tanh
from plants.plant import Plant
from plants.bathtub_plant import BathtubPlant
from plants.cournot_plant import CournotPlant
from plants.cruise_control_plant import CruiseControlPlant

from controllers.pid_controller import PIDController
from controllers.neural_net_controller import NNController
from jax import value_and_grad, random, tree_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


class CONSYS:
    def __init__(self, plant: Plant, controller: Controller, lrate: float = 0.1, epochs=50, timesteps=20, noise_range=(-0.01, 0.01)) -> None:
        self.plant = plant
        self.controller = controller
        self.lrate = lrate
        self.epochs = epochs
        self.timesteps = timesteps
        self.min_noise = noise_range[0]
        self.max_noise = noise_range[1]

    def run_system(self):
        params = self.controller.gen_params()
        grad_func = value_and_grad(self.run_epoch, argnums=0)
        all_mse = []
        for _ in range(self.epochs):
            flat_params = tree_util.tree_leaves(params)
            self.plant.reset()
            self.controller.reset()
            MSE, gradients = grad_func(flat_params)
            print(MSE)

            all_mse.append(MSE)
            params = self.controller.update_params(
                flat_params, self.lrate, gradients)
        self.plot_mse(all_mse)

    def run_epoch(self, flat_params):
        """
        Runs a single epoch and returns MSE
        """
        # TODO: make range adjustable
        params = tree_util.tree_unflatten(self.controller.treedef, flat_params)
        self.controller.initialize(params)
        key = random.PRNGKey(42)
        key, subkey = random.split(key)
        noises = random.uniform(subkey, shape=(
            self.timesteps,), minval=-self.min_noise, maxval=self.max_noise)
        control = 0
        for timestep in range(self.timesteps):
            control = self.run_timestep(control, noises[timestep])
            # print(f'Step {timestep}: control = {control}')
        return jnp.mean(jnp.square(jnp.array(self.controller.error_history)))

    def run_timestep(self, control, noise):
        output = self.plant.process(control, noise)
        error = self.plant.get_target() - output
        # print(f'output = {output}, error = {error}')
        return self.controller.calculate_control_value(error)

    def plot_mse(self, all_mse):
        plt.plot(all_mse)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Learning Progression')

        plt.show()
        return


def read_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    activation_functions = {
        "sigmoid": sigmoid,
        "ReLU": ReLU,
        "tanh": tanh,
        "linear": linear
    }

    controller_type = data["controller"]
    if controller_type == 'NN':
        controller_params = data.get("controller_params", {})
        controller_params["activation_funcs"] = [activation_functions[name]
                                                 for name in controller_params["activation_funcs"]]
    else:
        controller_params = None

    plant_type = data.get("plant", "")
    plant_params = data.get("plant_params", {})
    consys_params = data.get("consys_params", {})

    return controller_type, controller_params, plant_type, plant_params, consys_params


def main(filepath):
    controller_type, controller_params, plant_type, plant_params, consys_params = read_json(
        filepath)

    if controller_type == 'PID':
        controller = PIDController()
    if controller_type == 'NN':
        controller = NNController(**controller_params)
    if plant_type == 'Bathtub':
        plant = BathtubPlant(**plant_params)
    if plant_type == 'Cournot':
        plant = CournotPlant(**plant_params)
    if plant_type == 'CC':
        plant = CruiseControlPlant(**plant_params)

    consys = CONSYS(plant, controller, **consys_params)
    consys.run_system()
    if controller_type == 'PID':
        controller.plot_params()

if __name__ == '__main__':
    np.random.seed(42)
    # main('runs/nn-cournot.json')
    # main('runs/pid-cournot.json')

    # main('runs/nn-cc.json')
    # main('runs/pid-cc.json')

    # main('runs/nn-bathtub.json')
    # main('runs/pid-bathtub.json')
