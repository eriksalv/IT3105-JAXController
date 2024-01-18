from abc import ABC, abstractmethod


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
        return sum(self.error_history)
    
    @abstractmethod
    def gen_params():
        return

    @abstractmethod
    def calculate_control_value(self, error):
        self.error_history.append(error)

    @abstractmethod
    def update_params(self, lrate):
        return
    
    @abstractmethod
    def initialize(self, *params):
        return
