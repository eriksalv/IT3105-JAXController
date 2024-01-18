from controllers.controller import Controller
from plants.plant import Plant
from plants.bathtub_plant import BathtubPlant
from controllers.pid_controller import PIDController
import numpy as np


class CONSYS:
    def __init__(self, plant: Plant, controller: Controller, lrate: float = 0.01, epochs=50) -> None:
        self.plant = plant
        self.controller = controller
        self.lrate = lrate

    def run_system(self, epochs=50):
        for _ in epochs:
            self.run_epoch()

    def run_epoch(self, timesteps=10):
        self.plant.reset()
        self.controller.reset()
        # TODO: make range adjustable
        noises = np.random.uniform(-0.01, 0.01, timesteps)
        control = 0
        for timestep in range(timesteps):
            control = self.run_timestep(control, noises[timestep])
            print(f'Step {timestep}: control = {control}')
        self.controller.update_params(self.lrate)

    def run_timestep(self, control, noise):
        output = self.plant.output(control, noise)
        error = (self.plant.get_target() - output)**2
        print(f'output = {output}, error = {error}')
        return self.controller.calculate_control_value(error)


if __name__ == '__main__':
    plant = BathtubPlant(100, 1, 50)
    controller = PIDController()
    consys = CONSYS(plant, controller)
    consys.run_epoch()
