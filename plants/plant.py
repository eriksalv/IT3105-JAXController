from abc import ABC, abstractmethod


class Plant(ABC):
    @abstractmethod
    def process(self, U: float, noise: float) -> float:
        """updates plant state based on control value (U) and noise

        Parameters
        ----------
        U : float
            control value
        noise : float
            random noise

        Returns
        -------
        float
            new state value
        """
        return

    @abstractmethod
    def reset(self) -> None:
        """resets plant back to initial state
        """
        return

    @abstractmethod
    def get_target(self) -> float:
        """returns the target value for the output of the plant
        """
        return
