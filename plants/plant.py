from abc import ABC, abstractmethod


class Plant(ABC):
    @abstractmethod
    def process(self, U, noise):
        return

    @abstractmethod
    def reset(self):
        return
