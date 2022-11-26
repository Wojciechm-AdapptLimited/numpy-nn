import numpy as np
from typing import Optional
from time import perf_counter
from numpy_nn.data import Dataset
from numpy_nn.optimizers import Optimizer
from numpy_nn.losses import Loss
from numpy_nn.models import Model
from numpy_nn.models.history import History
from numpy_nn.layers import Layer
from numpy_nn.utils import accuracy


class Sequential(Model):
    """
    Class representing the simplest model structure, in which all layers form a one-dimensional array.

    Attributes:

    - optimizer (Optimizer): Optimizer function of the model, used to update layers' parameters
    - loss (Loss): Loss function of the model
    - metrics (dict[str, list[float]]): Dictionary for logging model training history
    - layers (list[Layer]): List of all layers, representing the model architecture

    Args:
        layers (list[Layer]): List of Layer objects
    """

    layers: list[Layer]

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def build(self) -> None:
        input_dim = self.layers[0].shape
        for layer in self.layers[1:]:
            layer.build(input_dim)
            input_dim = layer.shape

    def compile(self, optimizer: Optimizer, loss: Loss, metrics: list[str]) -> None:
        self.optimizer = optimizer
        self.loss = loss
        for metric in metrics:
            self.metrics[metric] = []
        self.build()

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, da: np. ndarray) -> None:
        for idx, layer in enumerate(self.layers[-1:0:-1]):
            da, dw, db = layer.backward(da)
            layer.weights = self.optimizer.update(layer.weights, dw, f'w{idx}')
            layer.bias = self.optimizer.update(layer.bias, db, f'b{idx}')

    def train(self, data: Dataset) -> None:
        losses = []
        accuracies = []
        for x, y in data.load():
            x = self.forward(x)
            losses.append(self.loss.loss(y, x))
            accuracies.append(accuracy(y, x))
            self.backward(self.loss.grad(y, x))
        self.metrics['loss'].append(sum(losses) / len(losses))
        if 'accuracy' in self.metrics:
            self.metrics['accuracy'].append(sum(accuracies) / len(accuracies))

    def validate(self, data: Dataset) -> None:
        losses = []
        accuracies = []
        for x, y in data.load():
            x = self.forward(x)
            losses.append(self.loss.loss(y, x))
            accuracies.append(accuracy(y, x))
        self.metrics['val_loss'].append(sum(losses) / len(losses))
        if 'val_accuracy' in self.metrics:
            self.metrics['val_accuracy'].append(sum(accuracies) / len(accuracies))

    def fit(self, train_data: Dataset, validation_data: Optional[Dataset] = None, epochs: int = 100) -> History:
        self.metrics['loss'] = []
        if validation_data is not None:
            self.metrics['val_loss'] = []

        for epoch in range(epochs):
            start_time = perf_counter()
            self.train(train_data)
            if validation_data is not None:
                self.validate(validation_data)
            end_time = perf_counter()
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'{end_time - start_time:.3f}s', end='')
            for name, metric in self.metrics.items():
                print(f' - {name}: {metric[-1]:.3f}', end='')
            print()

        return History(np.arange(epochs), self.metrics)

    def predict(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.forward(x)
        return y_pred

    def add(self, layer: Layer) -> None:
        """
        Method used for manual addition to the 'layers' attribute.
        Must be called before the compiling of the model.

        Args:
            layer (Layer): Layer object to be added

        """
        self.layers.append(layer)

    def pop(self, index: int) -> Layer:
        """
        Method used for manually removing Layer from the 'layers' attribute.
        Must be called before the compiling of the model.

        Args:
            index (int): Index of the layer to be removed
        Returns:
            Layer: Removed layer
        """
        return self.layers.pop(index)
