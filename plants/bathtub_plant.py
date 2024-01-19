from .plant import Plant
import jax.numpy as jnp
from jax import random

class BathtubPlant(Plant):
    def __init__(self, A, C, h_0):
        super().__init__()
        self.A = A
        self.C = C
        self.h_0 = h_0
        self.height=h_0

    def calculate_velocity(self):
        gravity = 9.81
        velocity = jnp.sqrt(2 * gravity * self.height)
        return velocity
    
    def calculate_flow_rate(self):
        flow_rate = self.calculate_velocity()*self.C
        return flow_rate
    
    def calculate_volume_change(self, U):
        key = random.PRNGKey(42)
        key, subkey = random.split(key)
        disturbance = random.uniform(subkey, shape=(1,), minval=-0.01, maxval=0.01)[0]
        return U + disturbance - self.calculate_flow_rate()
    
    def calculate_height_change(self, U):
        return self.calculate_volume_change(U)/self.A

    def process(self, control_signal):
        height_change= self.calculate_height_change(control_signal)   
        self.height = self.height + height_change
        self.height = jnp.max(jnp.array([self.height, 0])) # kan forkastes senere
        return height_change
    
    def reset(self):
        self.height=self.h_0
       
    
if __name__ == '__main__':
    plant = BathtubPlant(20, 5, 10)
    print(plant.process(2))
    print(plant.process(2))