from plant import Plant
import numpy as np

class CournotPlant(Plant):
    def __init__(self, pMax, cm):
        super().__init__()
        self.pMax = pMax
        self.cm = cm
        self.q1 = 0.5
        self.q2 = 0.5
        self.T = 2
    def process(self, control_signal):

        self.q_1 = self.q1 + control_signal
        self.q2 = self.q2 + np.random.uniform(-0.01, 0.01)
        self.q1 = np.clip(self.q1, 0, 1)
        self.q2 = np.clip(self.q2, 0, 1)

        q = self.q1 + self.q2
        price = self.pMax - q
        profit = self.q1 * (price - self.cm)
        error  = self.T - profit
        return error
    
if __name__ == '__main__':
    plant = CournotPlant(2, 0.1)
    print(plant.process(0.01))
   