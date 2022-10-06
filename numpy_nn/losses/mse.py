import numpy as np
from numpy_nn.losses.loss import Loss


class MeanSquaredError(Loss):
    """
    Class implementing Mean Square Error loss function.
    """

    @staticmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r"""
        Static method used to calculate the value of the loss function.
        Calculations used:

        .. math::
            \sum{(\hat{y} - y)^2 } / m

        Where m is the number of samples.

        Args:
            y_true (np.ndarray): One-hot encoded array of targets
            y_pred (np.ndarray): Output array from the network

        Returns:
            float: Value of the loss function
        """
        m = y_pred.shape[-1]
        loss = np.sum((y_pred - y_true) ** 2) / m
        return loss

    @staticmethod
    def grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        r"""
        Static used to calculate the gradient from the loss function.
        Calculations used:

        .. math::
            \nabla a = 2 * (\hat{y} - y) / m

        Args:
            y_true (np.ndarray): One-hot encoded array of targets
            y_pred (np.ndarray): Output array from the network

        Returns:
            np.ndarray: Gradient for the network
        """
        m = y_pred.shape[-1]
        grad = 2 * (y_pred - y_true) / m
        return grad
