from abc import ABC, abstractmethod


class Plant:
    @abstractmethod
    def process(self, U, noise):
        return

    @abstractmethod
    def reset(self):
        return
