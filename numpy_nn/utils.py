import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    r"""
    Performs column-wise normalization on the input array.
    Calculation:

    .. math ::
        x = (x - x_{min}) / (x_{max} - x_{min})

    Args:
        x (np.ndarray): Input array

    Returns:
        np.ndarray: Normalized input array
    """
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    return (x - x_min) / (x_max - x_min)


def standardize(x: np.ndarray) -> np.ndarray:
    r"""
    Performs column-wise standardization on the input array.
    Calculation:

    .. math ::
        x = (x - \mu_{x}) / \sigma_{x}

    Args:
        x (np.ndarray): Input array

    Returns:
        np.ndarray: Standardized input array
    """
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / x_std


def one_hot(x: np.ndarray) -> np.ndarray:
    """
    Function used for formatting arrays from ordinal format to one-hot format.

    Args:
        x (np.ndarray): Input array, in ordinal formatting

    Returns:
        np.ndarray: One-hot formatted array
    """
    one_hot_x = np.zeros((x.size, x.max() + 1))
    one_hot_x[np.arange(x.size), x] = 1
    return one_hot_x


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Function for calculating prediction accuracy.

    Args:
        y_true (np.ndarray): Target array, one-hot
        y_pred (np.ndarray): Predictions array

    Returns:
        float: Accuracy score
    """
    class_pred = np.argmax(y_pred, axis=0)
    class_true = np.argmax(y_true, axis=0)
    scores = np.asarray(class_pred == class_true)
    return np.mean(scores)
