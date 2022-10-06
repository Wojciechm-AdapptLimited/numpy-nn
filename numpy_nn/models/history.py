import numpy as np
from dataclasses import dataclass


@dataclass
class History:
    """
    Dataclass used as an output from the model training.
    Holds data about the training.

    Attributes:

    - epochs (np.ndarray): Array of numbers representing each epoch, 0:number of epochs
    - history (dict[str, list[float]]): Dictionary holding data about history of each metric
    """

    epochs: np.ndarray
    history: dict[str, list[float]]

