from .plant import Plant
import jax.numpy as jnp


class BathtubPlant(Plant):
    def __init__(self, A, C, h0):
        self.A = A
        self.C = C
        self.h0 = h0
        self.height = h0

    def process(self, U, noise):
        velocity = jnp.sqrt(2 * 9.81 * self.height)
        flow_rate = velocity * self.C
        volume_change = U + noise - flow_rate

        self.height += volume_change / self.A
        self.height = jnp.maximum(self.height, 0)
        return self.height

    def reset(self):
        self.height = self.h0

    def get_target(self):
        return self.h0


if __name__ == '__main__':
    plant = BathtubPlant(20, 5, 10)
    print(plant.process(2))
    print(plant.process(2))
