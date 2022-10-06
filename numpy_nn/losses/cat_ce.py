import numpy as np
from numpy_nn.losses import Loss


class CategoricalCrossEntropy(Loss):
    """
    Class implementing Categorical Cross Entropy loss function.
    """

    @staticmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r"""
        Static method used to calculate the value of the loss function.
        Calculations used:

        .. math::
            - \sum{y \times \log{\hat{y}}} / m

        Where m is the number of samples.

        Args:
            y_true (np.ndarray): One-hot encoded array of targets
            y_pred (np.ndarray): Output array from the network

        Returns:
            float: Value of the loss function
        """
        m = y_pred.shape[-1]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
        return loss

    @staticmethod
    def grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        r"""
        Static used to calculate the gradient from the loss function.
        Calculations used:

        .. math::
            \nabla a = \hat{y} - y

        Args:
            y_true (np.ndarray): One-hot encoded array of targets
            y_pred (np.ndarray): Output array from the network

        Returns:
            np.ndarray: Gradient for the network
        """
        return y_pred - y_true
