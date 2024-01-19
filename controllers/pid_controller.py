from .controller import Controller
import jax
import jax.numpy as jnp
class PIDController(Controller):
    def gen_params(self):
        print("Genererer  params")
        kp = jnp.float32(1.0)
        ki = jnp.float32(2.0)
        kd = jnp.float32(3.0)
        return (kp, ki, kd)

    def initialize(self, *params):
        self.kp = params[0]
        self.ki = params[1]
        self.kd = params[2]

    def calculate_control_value(self, error) -> float:
        super().calculate_control_value(error)
        return self.kp * error + self.ki * self.integral() + self.kd * self.derivative()

    def update_params(self, params, lrate, gradients):
        kp, ki, kd = params
        kp_new = kp - lrate * gradients[0]
        ki_new = ki - lrate * gradients[1]
        kd_new = kd - lrate * gradients[2]


        return kp_new, ki_new, kd_new
    
    def get_params(self):
        return 
if __name__ == '__main__':
    con = PIDController()
    print(con.calculate_control_value(5))
    print(con.calculate_control_value(2))

