from abc import ABC, abstractmethod


class Plant(ABC):
    @abstractmethod
    def output(self, control_signal, noise):
        return

    @abstractmethod
    def reset(self):
        return

    @abstractmethod
    def get_target(self):
        return
