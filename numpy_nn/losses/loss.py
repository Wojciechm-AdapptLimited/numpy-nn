import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """
    Abstract Base Class for any loss function class.
    All classes that could be interpreted as loss function of the network should inherit from Loss class.
    """

    @staticmethod
    @abstractmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Static method used to calculate the value of the loss function.

        Args:
            y_true (np.ndarray): One-hot encoded array of targets
            y_pred (np.ndarray): Output array from the network

        Returns:
            float: Value of the loss function
        """
        pass

    @staticmethod
    @abstractmethod
    def grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Static used to calculate the gradient from the loss function.

        Args:
            y_true (np.ndarray): One-hot encoded array of targets
            y_pred (np.ndarray): Output array from the network

        Returns:
            np.ndarray: Gradient for the network
        """
        pass
