from plant import Plant
import numpy as np

class BathtubPlant(Plant):
    def __init__(self, A, C, h_0):
        super().__init__()
        self.A = A
        self.C = C
        self.h_0 = h_0
        self.height=h_0

    def calculate_velocity(self):
        gravity = 9.81
        velocity = np.sqrt(2*gravity*self.height)
        return velocity
    
    def calculate_flow_rate(self):
        flow_rate = self.calculate_velocity()*self.C
        return flow_rate
    
    def calculate_volume_change(self, U):
        disturbance = np.random.uniform(-0.01, 0.01)
        return U + disturbance - self.calculate_flow_rate()
    
    def calculate_height_change(self, U):
        return self.calculate_volume_change(U)/self.A

    def process(self, control_signal):
        return self.calculate_height_change(control_signal)   
    
    def reset(self):
        self.height=self.h_0
       
    
if __name__ == '__main__':
    plant = BathtubPlant(20, 5, 10)
    print(plant.process(2))
   