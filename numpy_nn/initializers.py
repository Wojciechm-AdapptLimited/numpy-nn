import numpy as np
from typing import Callable


Params = tuple[np.ndarray, ...]
Initializer = Callable[[int, int], Params]


def rand_normal_init(input_dim: int, output_dim: int) -> Params:
    """
    Function used in initializing layer parameters using random normal distribution.

    Args:
        input_dim (int): Input dimensions of the layer
        output_dim (int): Output dimensions of the layer

    Returns:
        tuple[np.ndarray, ...]: Initialized weights and biases
    """
    w = np.random.randn(output_dim, input_dim) * 0.1
    b = np.random.randn(output_dim, 1) * 0.1
    return w, b


def rand_uniform_init(input_dim: int, output_dim: int) -> Params:
    """
    Function used in initializing layer parameters using random uniform distribution.

    Args:
        input_dim (int): Input dimensions of the layer
        output_dim (int): Output dimensions of the layer

    Returns:
        tuple[np.ndarray, ...]: Initialized weights and biases
    """
    w = np.random.rand(output_dim, input_dim) - 0.5
    b = np.random.rand(output_dim, 1) - 0.5
    return w, b


def ones_init(input_dim: int, output_dim: int) -> Params:
    """
    Function used in initializing layer parameters setting all values to one.

    Args:
        input_dim (int): Input dimensions of the layer
        output_dim (int): Output dimensions of the layer

    Returns:
        tuple[np.ndarray, ...]: Initialized weights and biases
    """
    w = np.ones(output_dim, input_dim)
    b = np.ones(output_dim, 1)
    return w, b


def zeros_init(input_dim: int, output_dim: int) -> Params:
    """
    Function used in initializing layer parameters setting all values to zero.

    Args:
        input_dim (int): Input dimensions of the layer
        output_dim (int): Output dimensions of the layer

    Returns:
        tuple[np.ndarray, ...]: Initialized weights and biases
    """
    w = np.zeros(output_dim, input_dim)
    b = np.zeros(output_dim, 1)
    return w, b


def get(identifier: str) -> Initializer:
    """
    Function that turns string identifiers into initializer function objects.

    Args:
        identifier (str): String identifier of the activation function.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Activation function object
    """
    functions = {
        'random normal initializer': rand_normal_init,
        'random uniform initializer': rand_uniform_init,
        'ones initializer': ones_init,
        'zeros initializer': zeros_init
    }

    return functions[identifier]
