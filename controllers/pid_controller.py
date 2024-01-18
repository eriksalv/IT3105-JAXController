from .controller import Controller
import jax


class PIDController(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.kp = 1
        self.ki = 2
        self.kd = 3

    def calculate_control_value(self, error) -> float:
        super().calculate_control_value(error)
        return self.kp * error + self.ki * self.integral() + self.kd * self.derivative()

    def update_params(self, lrate):
        return


if __name__ == '__main__':
    con = PIDController()
    print(con.calculate_control_value(5))
    print(con.calculate_control_value(2))
