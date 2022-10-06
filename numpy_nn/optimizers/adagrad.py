import numpy as np
from numpy_nn.optimizers import Optimizer


class AdaGrad(Optimizer):
    """
    Implementation of the AdaGrad optimizer function.

    Attributes:

    - learning_rate (float): Learning rate hyperparameter
    - cache (dict[str, np.ndarray]): Memory of previous updates
    - clip_th (float): Gradient clipping threshold
    - eps (float): Smoothing variable, to prevent dividing by zero

    Args:
        lr (float): Learning rate
        eps (float): Smoothing variable, to prevent dividing by zero, 1e-7 by default
        clip_th (float): Gradient clipping threshold, infinity by default
    """

    eps: float

    def __init__(self, lr: float, eps: float = 1e-7, clip_th: float = np.inf) -> None:
        self.learning_rate = lr
        self.eps = eps
        self.clip_th = clip_th

    def update(self, param: np.ndarray, param_grad: np.ndarray, param_name: str) -> np.ndarray:
        r"""
        Method used for parameter updates.
        Calculations:

        .. math ::
            G(t) = V(t - 1) + \nabla \Theta^{2}

            V(t) = \alpha \times \nabla \Theta / (\sqrt{G(t) + \epsilon})

            \Theta = \Theta - V(t)

        Where V(t) is the update in the current iteration, t - iteration number, epsilon - smoothing constant,
        alpha - learning rate and theta - the parameter to be updated


        Args:
            param (np.ndarray): Parameter to be updated
            param_grad (np.ndarray): Parameter gradient
            param_name (str): Name of the parameter, used for caching updates

        Returns:
            np.ndarray: Updated parameter
        """
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param_grad)

        param_grad = self.clip_grad(param_grad)

        self.cache[param_name] += np.power(param_grad, 2)

        update = self.learning_rate * param_grad / (np.sqrt(self.cache[param_name]) + self.eps)
        self.cache[param_name] = update

        return param - update
