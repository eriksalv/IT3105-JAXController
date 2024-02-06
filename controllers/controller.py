from abc import ABC, abstractmethod


class Controller(ABC):
    def __init__(self) -> None:
        self.error_history = []

    def reset(self) -> None:
        """resets error history
        """
        self.error_history = []

    def derivative(self) -> float:
        """calculates the derivative (difference) of the error history

        Returns
        -------
        float
            derivative of the last timestep
        """
        if len(self.error_history) > 1:
            return self.error_history[-1] - self.error_history[-2]
        else:
            return self.error_history[-1]

    def integral(self):
        """calculates the integral (sum) of the error history

        Returns
        -------
        float
            integral of the error history
        """
        return sum(self.error_history)

    @abstractmethod
    def calculate_control_value(self, error: float) -> float:
        """runs error through the controller to produce a new control value

        Parameters
        ----------
        error : float
            new error

        Returns
        -------
        float
            new control value
        """
        self.error_history.append(error)

    @abstractmethod
    def gen_params(self) -> list:
        """Generate initial parameters for the controller
        """
        return

    @abstractmethod
    def update_params(self, params, lrate, gradients):
        """performs gradient descent to update parameters

        Parameters
        ----------
        params : list
            parameters to update
        lrate : float
            learning rate
        gradients : list
            gradients of MSE wrt. parameters

        Returns
        -------
        list
            updated parameters
        """
        return
