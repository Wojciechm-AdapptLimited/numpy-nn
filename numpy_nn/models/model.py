import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from numpy_nn.data import Dataset
from numpy_nn.optimizers import Optimizer
from numpy_nn.losses import Loss
from numpy_nn.models.history import History


class Model(ABC):
    """
    Abstract Base Class for any model class.
    All classes that could be interpreted as network models should inherit from Model class.

    Attributes:

    - optimizer (Optimizer): Optimizer function of the model, used to update layers' parameters
    - loss (Loss): Loss function of the model
    - metrics (dict[str, list[float]]): Dictionary for logging model training history
    """

    optimizer: Optimizer
    loss: Loss
    metrics: dict[str, list[float]] = {}

    @abstractmethod
    def build(self) -> None:
        """
        Method for initializing the layers.
        """
        pass

    @abstractmethod
    def compile(self, optimizer: Optimizer, loss: Loss, metrics: list[str]) -> None:
        """
        Setter method for optimizer, loss and metrics attributes.

        Args:
            optimizer (Optimizer): Optimizer object
            loss (Loss): Loss object
            metrics (list[str]): List of metrics to be calculated and logged. Takes values: 'accuracy', 'val_accuracy'.
             Loss and validation loss are logged by default (latter if validation dataset is given).

        """
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Method for performing forward propagation of each batch.

        Args:
            x (np.ndarray): Input sample from the dataset

        Returns:
            np.ndarray: Model output, predicted values.
        """
        pass

    @abstractmethod
    def backward(self, da: np. ndarray) -> None:
        """
        Method for performing backward propagation of each batch.

        Args:
            da (np.ndarray): Gradient calculated from the loss function

        """
        pass

    @abstractmethod
    def train(self, data: Dataset) -> None:
        """
        Method called each epoch during training.
        Iterates through dataset updating layers parameters each iteration.

        Args:
            data (Dataset): Training dataset

        """
        pass

    @abstractmethod
    def validate(self, data: Dataset):
        """
        Method called each epoch during training if validation dataset is given.

        Args:
            data (Dataset): Validation dataset

        """
        pass

    @abstractmethod
    def fit(self, train_data: Dataset, validation_data: Optional[Dataset] = None, epochs: int = 100) -> History:
        """
        Method managing the training process.
        Calls the 'train' and 'validate' methods each epoch and prints the metrics.

        Args:
            train_data (Dataset): Training data
            validation_data (Dataset): Validation data
            epochs (int): Number of epochs

        Returns:
            History: Logged history of the training process
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Method for predicting new values, called after the training is finished.

        Args:
            x (np.ndarray): Input data

        Returns:
            np.ndarray: Output inference from the model
        """
        pass
