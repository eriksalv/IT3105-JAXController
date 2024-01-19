from abc import ABC, abstractmethod

import jax.numpy as jnp

class Controller(ABC):
    def __init__(self) -> None:
        self.error_history = []

    def reset(self) -> None:
        self.error_history = []

    def derivative(self) -> float:
        if len(self.error_history) > 1:
            return self.error_history[-1] - self.error_history[-2]
        else:
            return self.error_history[-1]

    def integral(self) -> float:
        return jnp.sum(jnp.array(self.error_history))

    def compute_gradients(self, error_history):
        pass
    @abstractmethod
    def calculate_control_value(self, error):
        self.error_history.append(error)
