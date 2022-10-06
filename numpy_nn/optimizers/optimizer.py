import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Abstract Base Class for any optimizer class.
    All classes that could be interpreted as optimizer function should inherit from Optimizer class.

    Attributes:

    - learning_rate (float): Learning rate hyperparameter
    - cache (dict[str, np.ndarray]): Memory of previous updates
    - clip_th (float): Gradient clipping threshold
    """

    learning_rate: float
    cache: dict[str, np.ndarray] = {}
    clip_th: float

    def clip_grad(self, param_grad: np.ndarray) -> np.ndarray:
        r"""
        Method that performs gradient clipping to prevent gradient explosion.
        Calculations:

        .. math ::
            if  \: ||\nabla|| > th:

            \nabla = th * \nabla / ||\nabla||

        Where th is a clipping threshold.

        Args:
            param_grad (np.ndarray): Gradient

        Returns:
            np.ndarray: Clipped gradient
        """
        if np.linalg.norm(param_grad) >= self.clip_th:
            return self.clip_th * param_grad / np.linalg.norm(param_grad)
        return param_grad

    @abstractmethod
    def update(self, param: np.ndarray, param_grad: np.ndarray, param_name: str) -> np.ndarray:
        """
        Method used for parameter updates.

        Args:
            param (np.ndarray): Parameter to be updated
            param_grad (np.ndarray): Parameter gradient
            param_name (str): Name of the parameter, used for caching updates

        Returns:
            np.ndarray: Updated parameter
        """
        pass
