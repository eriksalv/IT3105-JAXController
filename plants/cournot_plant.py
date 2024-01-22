from .plant import Plant
import jax.numpy as jnp

class CournotPlant(Plant):
    def __init__(self, pMax, cm):
        super().__init__()
        self.pMax = pMax
        self.cm = cm
        self.q1 = 0.5
        self.q2 = 0.5
        self.T = 2
    def process(self, control_signal, noise):
        
        self.q1 = self.q1 + control_signal
        self.q2 = self.q2 + noise
        self.q1 = jnp.clip(self.q1, 0, 1)
    
        self.q2 = jnp.clip(self.q2, 0, 1)

        q = self.q1 + self.q2
        price = self.pMax - q
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
   