import numpy as np
from numpy_nn.layers import Layer

Params = tuple[np.ndarray, ...]
Shape = tuple[int, ...]


class Input(Layer):
    """
        Layer class representing input layer of the neural network.

        Attributes:

        - shape(tuple[int, ...]): Shape of the layer

        Args:
            shape(tuple[int, ...]): Shape of the layer, depending on the shape of the input
        """

    def __init__(self, shape: Shape) -> None:
        self.shape = shape

    def build(self, prev_shape: Shape) -> None:
        pass

    def forward(self, a: np.ndarray) -> np.ndarray:
        return a

    def backward(self, up_grad: np.ndarray) -> Params:
        pass
