from .controller import Controller
import numpy as np


class PIDController(Controller):
    def gen_params(self):
        kp = np.random.uniform(-1,1)
        ki = np.random.uniform(-1,1)
        kd = np.random.uniform(-1,1)
        return (kp, ki, kd)

    def initialize(self, *params):
        self.kp = params[0]
        self.ki = params[1]
        self.kd = params[2]

    def calculate_control_value(self, error) -> float:
        super().calculate_control_value(error)
        return self.kp * error + self.ki * self.integral() + self.kd * self.derivative()

    def update_params(self, lrate, gradients):
        kp = self.kp - lrate * gradients[0]
        ki = self.ki - lrate * gradients[1]
        kd = self.kd - lrate * gradients[2]
        return (kp, ki, kd)


if __name__ == '__main__':
    con = PIDController()
    print(con.calculate_control_value(5))
    print(con.calculate_control_value(2))
