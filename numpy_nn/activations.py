import numpy as np
from typing import Callable

Activation = Callable[[np.ndarray], np.ndarray]


def softmax(x: np.ndarray) -> np.ndarray:
    r"""
    Calculates the softmax of all elements in the input array, row-wise. Leaves the columns intact.
    Calculations:

    .. math ::
        exp(x) = e^{x - x_{max}}

        \sigma (x) = exp_x / (\sum^{K}_{i}{exp(x)})

    Where K is the number of classes

    Args:
        x (np.ndarray): Input array

    Returns:
        np.ndarray: Output array, element-wise softmax of x
    """
    x_max = np.max(x, axis=0, keepdims=True)
    x_exp = np.exp(x - x_max)
    denominator = np.sum(x_exp, axis=0, keepdims=True)
    return x_exp / denominator


def sigmoid(x: np.ndarray) -> np.ndarray:
    r"""
    Calculates the sigmoid of all elements in the input array.

    .. math::
         S(x) = 1 / (1 - e^{-x})

    Args:
        x (np.ndarray): Input array

    Returns:
        np.ndarray: Output array, sigmoid of x
    """
    x_exp = np.exp(-x)
    denominator = 1 + x_exp
    return 1/denominator


def relu(x: np.ndarray) -> np.ndarray:
    r"""
    Calculates the rectified linear of all elements in the input array.
    Used as an activation function in ReLU layer.

    .. math ::
        f(x) = max(0, x)

    Args:
        x (np.ndarray): Input array

    Returns:
        np.ndarray: Output array, element-wise rectified linear of x
    """
    return np.maximum(0, x)


def relu_prime(x: np.ndarray) -> np.ndarray:
    """
    Calculates the rectified linear derivative of all elements in the input array.
    Used in ReLU layers during backpropagation.

    Returns 1 if x > 0, 0 otherwise.

    Args:
        x (np.ndarray): Input array

    Returns:
        np.ndarray: Output array, element-wise rectified linear derivative of x
    """
    return x > 0


def linear(x: np.ndarray) -> np.ndarray:
    r"""
    Calculates the linear function.
    Used as an activation function in Linear layer.

    .. math ::
        f(x) = x

    Args:
        x (np.ndarray): Input array

    Returns:
        np.ndarray: Output array, element-wise linear function of x
    """
    return x


def linear_prime(x: np.ndarray) -> np.ndarray:
    """
    Calculates the linear function.
    Used in Linear, Softmax and Sigmoid layers during backpropagation.

    Args:
        x (np.ndarray): input array

    Returns:
        np.ndarray: Output array of ones, with the same shape as x
    """
    return np.ones_like(x)


def get(identifier: str) -> Activation:
    """
    Function that turns string identifiers into activation function objects.

    Args:
        identifier (str): String identifier of the activation function.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Activation function object
    """
    functions = {
        'relu': relu,
        'softmax': softmax,
        'linear': linear,
        'sigmoid': sigmoid
    }

    return functions[identifier]


def get_derivative(activation: Activation) -> Activation:
    """
    Function that takes activation function objects and returns their derivatives.

    Args:
        activation (Callable[[np.ndarray], np.ndarray]): Activation function

    Returns:
        Callable[[np.ndarray], np.ndarray]: Activation function derivative
    """
    derivatives = {
        relu: relu_prime,
        linear: linear_prime,
        softmax: linear_prime,
        sigmoid: linear_prime
    }

    return derivatives[activation]
