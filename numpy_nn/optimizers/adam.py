import numpy as np
from numpy_nn.optimizers import Optimizer


class Adam(Optimizer):
    """
    Implementation of the Adam optimizer function.

    Attributes:

    - learning_rate (float): Learning rate hyperparameter
    - cache (dict[str, np.ndarray]): Memory of previous updates
    - clip_th (float): Gradient clipping threshold
    - eps (float): Smoothing variable, to prevent dividing by zero
    - decay_1 (float): Exponential decay rate for the first moment estimate
    - decay_2 (float): Exponential decay rate for the second moment estimate

    Args:
        lr (float): Learning rate
        eps (float): Smoothing variable, to prevent dividing by zero, 1e-7 by default
        clip_th (float): Gradient clipping threshold, infinity by default
        d1 (float): Exponential decay rate for the first moment estimate
        d2 (float): Exponential decay rate for the second moment estimate
    """

    decay_1: float
    decay_2: float
    eps: float

    def __init__(self, lr: float, d1: float = 0.9, d2: float = 0.999,
                 eps: float = 1e-7, clip_th: float = np.inf) -> None:
        self.learning_rate = lr
        self.decay_1 = d1
        self.decay_2 = d2
        self.eps = eps
        self.clip_th = clip_th

    def update(self, param: np.ndarray, param_grad: np.ndarray, param_name: str) -> np.ndarray:
        r"""
        Method used for parameter updates.
        Calculations:

        .. math ::
            t = t + 1

            g_t = \nabla \Theta

            m_t = \beta_1 \times m_{t-1} + (1 - \beta_1) \times g_t

            v_t = \beta_2 \times v_{t-1} + (1 - \beta_2) \times g_t^{2}

            \hat{m_t} = m_t / (1 - \beta_1^t)

            \hat{v_t} = v_t / (1 - \beta_2^t)

            V(t) = \alpha \times \hat{m_t} / (\sqrt{\hat{v_t}} + \epsilon)

            \Theta = \Theta - V(t)

        Where V(t) is the update in the current iteration, t - iteration number, beta1 and beta2 - decaying rates
        m and v - moment estimates, epsilon - smoothing constant, alpha - learning rate and theta - the parameter
        to be updated


        Args:
            param (np.ndarray): Parameter to be updated
            param_grad (np.ndarray): Parameter gradient
            param_name (str): Name of the parameter, used for caching updates

        Returns:
            np.ndarray: Updated parameter
        """
        if param_name not in self.cache:
            self.cache[f'{param_name}t'] = np.array([0])
            self.cache[f'{param_name}mean'] = np.zeros_like(param_grad)
            self.cache[f'{param_name}var'] = np.zeros_like(param_grad)

        param_grad = self.clip_grad(param_grad)

        t = self.cache[f'{param_name}t'] + 1
        mean = self.cache[f'{param_name}mean']
        var = self.cache[f'{param_name}var']

        self.cache[f'{param_name}t'] = t
        self.cache[f'{param_name}mean'] = self.decay_1 * mean + (1 - self.decay_1) * param_grad
        self.cache[f'{param_name}var'] = self.decay_2 * var + (1 - self.decay_2) * param_grad ** 2

        m_hat = self.cache[f'{param_name}mean'] / (1 - self.decay_1 ** t)
        v_hat = self.cache[f'{param_name}var'] / (1 - self.decay_2 ** t)
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return param - update
