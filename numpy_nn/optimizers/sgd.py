import numpy as np
from numpy_nn.optimizers import Optimizer


class SGD(Optimizer):
    """
    Implementation of the Stochastic Gradient Descent optimizer function.

    Attributes:

    - learning_rate (float): Learning rate hyperparameter
    - cache (dict[str, np.ndarray]): Memory of previous updates
    - clip_th (float): Gradient clipping threshold
    - momentum (float): Hyperparameter that determines the influence of the previous update on current update

    Args:
        lr (float): Learning rate
        momentum (float): Influence of the previous update
        clip_th (float): Gradient clipping threshold, infinity by default
    """

    momentum: float

    def __init__(self, lr: float, momentum: float, clip_th: float = np.inf) -> None:
        self.learning_rate = lr
        self.momentum = momentum
        self.clip_th = clip_th

    def update(self, param: np.ndarray, param_grad: np.ndarray, param_name: str) -> np.ndarray:
        r"""
        Method used for parameter updates.
        Calculations:

        .. math ::
            V(t) = \gamma \times V(t-1) + (1 - \gamma) \times \nabla \Theta

            \Theta = \Theta - \alpha \times V(t)

        Where V(t) is the update in the current iteration, t - iteration number, gamma - momentum,
        alpha - learning rate and theta the parameter to be updated


        Args:
            param (np.ndarray): Parameter to be updated
            param_grad (np.ndarray): Parameter gradient
            param_name (str): Name of the parameter, used for caching updates

        Returns:
            np.ndarray: Updated parameter
        """
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param_grad)
        prev_update = self.cache[param_name]

        param_grad = self.clip_grad(param_grad)

        update = self.momentum * prev_update + (1 - self.momentum) * param_grad
        self.cache[param_name] = update

        return param - self.learning_rate*update
