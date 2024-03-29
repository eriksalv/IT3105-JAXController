from .plant import Plant
import jax.numpy as jnp


class CournotPlant(Plant):
    def __init__(self, pMax, cm):
        self.pMax = pMax
        self.cm = cm
        self.q1 = 0.5
        self.q2 = 0.5
        self.T = 1
        self.q = None

    def process(self, U, noise):
        self.q1 = jnp.clip(self.q1 + U, 0, 1)
        self.q2 = jnp.clip(self.q2 + noise, 0, 1)

        self.q = self.q1 + self.q2

        price = self.pMax - self.q
        profit = self.q1 * (price - self.cm)
        return profit

    def get_target(self):
        return self.T

    def reset(self):
        self.q1 = 0.5
        self.q2 = 0.5


if __name__ == '__main__':
    plant = CournotPlant(2, 0.1)
    print(plant.process(0.01))
