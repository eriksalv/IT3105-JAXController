from .plant import Plant
import jax.numpy as jnp


class CruiseControlPlant(Plant):
    def __init__(self, v0, m, c, theta):
        self.v0 = v0
        self.v = v0
        self.m = m
        self.c = c
        self.theta = theta

    def process(self, U, noise):
        gravity = self.m * 9.81 * jnp.sin(self.theta + noise)
        drag_resistance = self.c * self.v

        self.v += (U - drag_resistance - gravity) / self.m
        return self.v

    def get_target(self):
        return self.v0

    def reset(self):
        self.v = self.v0
