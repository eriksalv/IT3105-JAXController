from controllers.controller import Controller
from plants.plant import Plant
from plants.bathtub_plant import BathtubPlant
from controllers.pid_controller import PIDController
import numpy as np
from jax import value_and_grad
import jax.numpy as jnp


class CONSYS:
    def __init__(self, plant: Plant, controller: Controller, lrate: float = 0.01, epochs=50, timesteps=10) -> None:
        self.plant = plant
        self.controller = controller
        self.lrate = lrate
        self.epochs = epochs
        self.timesteps = timesteps

    def run_system(self):
        params = self.controller.gen_params()
        grad_func = value_and_grad(self.run_epoch, argnums=[0,1,2])
        for _ in range(self.epochs):
            self.plant.reset()
            self.controller.reset()
            MSE, gradients = grad_func(*params)
            print(gradients)
            print(MSE)
            params = self.controller.update_params(self.lrate, gradients)

    def run_epoch(self, *params):
        """
        Runs a single epoch and returns MSE
        """
        # TODO: make range adjustable
        self.controller.initialize(*params)
        noises = np.random.uniform(-0.01, 0.01, self.timesteps)
        control = 0
        for timestep in range(self.timesteps):
            control = self.run_timestep(control, noises[timestep])
            #print(f'Step {timestep}: control = {control}')
        return jnp.mean(jnp.sum(jnp.array(self.controller.error_history)))

    def run_timestep(self, control, noise):
        output = self.plant.output(control, noise)
        error = (self.plant.get_target() - output)**2
        #print(f'output = {output}, error = {error}')
        return self.controller.calculate_control_value(error)


if __name__ == '__main__':
    plant = BathtubPlant(100, 1, 50)
    controller = PIDController()
    consys = CONSYS(plant, controller)
    consys.run_system()
    # def test(x,y):
    #     return x**2 + 8*y
    
    # grad_func = value_and_grad(test, argnums=[0,1])
    # print(grad_func(2.0,2.0))
    

