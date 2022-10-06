import numpy as np
from numpy_nn.layers import Layer
from numpy_nn import initializers, activations

Params = tuple[np.ndarray, ...]
Shape = tuple[int, ...]


class Dense(Layer):
    """
    Layer class representing fully connected layer of the neural network.

    Attributes:

    - units (int): Number of units in the layer
    - shape (tuple[int, ...]): Shape of the layer
    - activation (Activation): Activation function of the layer, linear by default
    - initializer (Initializer): Initializer function of the layer, random normal by default
    - cache_a (np.ndarray): Cache for memorizing the layer input
    - cache_z (np.ndarray): Cache for memorizing the layer output
    - weights (np.ndarray): Layer weights
    - bias (np.ndarray): Layer bias

    Args:
        units (int): Number of hidden units
        activation (str): Identifier of the activation function, 'linear' by default
        initializer (str): Identifier of the initializer function, 'random normal initializer' by default
    """

    def __init__(self, units: int, activation: str = 'linear', initializer: str = 'random normal initializer') -> None:
        self.units = units
        self.shape = (1, self.units)
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)

    def build(self, prev_shape: Shape) -> None:
        input_dim = prev_shape[-1]
        self.weights, self.bias = self.initializer(input_dim, self.units)

    def forward(self, a: np.ndarray) -> np.ndarray:
        r"""
        Method used during forward propagation phase.
        Calculations used in this phase:

        .. math::
            z_{l} = w_{l} \cdot a_{l-1} + b_{l}

            a_{l} = f(z_{l})

        Where f is the activation function and l is the layer number

        Args:
            a (np.ndarray): Input array from the previous layer

        Returns:
            np.ndarray: Output array for the next layer
        """
        z = self.weights @ a + self.bias
        self.cache_a = a
        self.cache_z = z
        return self.activation(z)

    def backward(self, up_grad: np.ndarray) -> Params:
        r"""
        Method used during backward propagation phase.
        Calculations used in this phase:

        .. math::
            \nabla z_{l} = \nabla a_{l+1} \times f \prime (z_{l})

            \nabla w_{l} = (\nabla z_{l} \cdot a_{l-1}^T) / m

            \nabla b_{l} = \sum_{i=1}^{m}{\nabla z_{l}^{i}} / m

            \nabla a_{l} = (w_{l}^T \cdot \nabla z_{l}) / m

        Where f is the activation function, l is the layer number and m is the number of samples

        Args:
            up_grad (np.ndarray): Gradient from the next layer

        Returns:
            tuple[np.ndarray, ...]: Output gradient for the previous layer,
             bias and weights gradients for parameters optimization
        """
        d_func = activations.get_derivative(self.activation)
        m = self.cache_a.shape[-1]
        dz = up_grad * d_func(self.cache_z)
        dw = dz @ self.cache_a.T / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        da = self.weights.T @ dz
        return da, dw, db
