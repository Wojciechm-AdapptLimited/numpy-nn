import numpy as np
from abc import ABC, abstractmethod
from numpy_nn.activations import Activation
from numpy_nn.initializers import Initializer

Params = tuple[np.ndarray, ...]
Shape = tuple[int, ...]


class Layer(ABC):
    """
    Abstract Base Class for any layer class.
    All classes that could be interpreted as layer of the network should inherit from Layer class.

    Attributes:

    - units (int): Number of units in the layer
    - shape (tuple[int, ...]): Shape of the layer
    - activation (Activation): Activation function of the layer, linear by default
    - initializer (Initializer): Initializer function of the layer, random normal by default
    - cache_a (np.ndarray): Cache for memorizing the layer input
    - cache_z (np.ndarray): Cache for memorizing the layer output
    - weights (np.ndarray): Layer weights
    - bias (np.ndarray): Layer bias
    """

    units: int
    shape: Shape
    activation: Activation
    initializer: Initializer
    cache_a: np.ndarray
    cache_z: np.ndarray
    weights: np.ndarray
    bias: np.ndarray

    @abstractmethod
    def build(self, prev_shape: Shape) -> None:
        """
        Method for initializing layer parameters

        Args:
            prev_shape (tuple[int, ...]): Shape of the previous layer
        """
        pass

    @abstractmethod
    def forward(self, a: np.ndarray) -> np.ndarray:
        """
        Method used during forward propagation phase.

        Args:
            a (np.ndarray): Input array from the previous layer

        Returns:
            np.ndarray: Output array for the next layer
        """
        pass

    @abstractmethod
    def backward(self, up_grad: np.ndarray) -> Params:
        """
        Method used during backward propagation phase.

        Args:
            up_grad (np.ndarray): Gradient from the next layer

        Returns:
            tuple[np.ndarray, ...]: Output gradient for the previous layer,
             bias and weights gradients for parameters optimization
        """
        pass
