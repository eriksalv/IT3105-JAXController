from .plant import Plant
import jax.numpy as jnp
from jax import random


class BathtubPlant(Plant):
    def __init__(self, A, C, h_0):
        super().__init__()
        self.A = A
        self.C = C
        self.h_0 = h_0
        self.height = h_0

    def calculate_velocity(self):
        gravity = 9.81
        velocity = jnp.sqrt(2 * gravity * self.height)
        return velocity

    def calculate_flow_rate(self):
        flow_rate = self.calculate_velocity()*self.C
        return flow_rate

    def calculate_volume_change(self, U, noise):
        return U + noise - self.calculate_flow_rate()

    def calculate_height_change(self, U, noise):
        return self.calculate_volume_change(U, noise)/self.A

    def process(self, control_signal, noise):
        height_change = self.calculate_height_change(control_signal, noise)
        self.height += height_change
        # kan forkastes senere
        self.height = jnp.maximum(self.height, 0)
        return self.height

    def reset(self):
        self.height = self.h_0

    def get_target(self):
        return self.h_0


if __name__ == '__main__':
    plant = BathtubPlant(20, 5, 10)
    print(plant.process(2))
    print(plant.process(2))
