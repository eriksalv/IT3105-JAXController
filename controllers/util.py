import jax.numpy as jnp


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def ReLU(x):
    return jnp.maximum(0, x)


def tanh(x):
    return jnp.tanh(x)
