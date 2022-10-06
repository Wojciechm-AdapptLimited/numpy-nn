import numpy as np
import pandas as pd
from numpy_nn import normalize, standardize, one_hot


class Dataset:

    """
    Data structure for storing, formatting and loading the dataset.

    Attributes:

    - batch_size (int): Number of loaded samples, set to the length of the dataset by default
    - target (np.ndarray): One-hot formatted array of targets
    - dataset (np.ndarray): Formatted array of features, standardized and normalized numerical data
      and one-hot formatted categorical data

    Args:
        df (pd.DataFrame): Dataset to be formatted and stored
        target_name (str): Identifier of the labels array
        cat_attr_names (list[str]): List of identifiers of the categorical features
        num_attr_names (list[str]): List of identifiers of the numerical features

    """

    batch_size: int
    target: np.ndarray
    dataset: np.ndarray

    def __init__(self, df: pd.DataFrame, target_name: str,
                 cat_attr_names: list[str], num_attr_names: list[str]) -> None:

        self.target = df[target_name].to_numpy()
        if np.max(self.target) > 1:
            self.target = one_hot(self.target)

        self.dataset = standardize(normalize(df[num_attr_names].to_numpy()))

        for name in cat_attr_names:
            cat = df[name].to_numpy()
            if np.max(cat) > 1:
                cat = one_hot(df[name].to_numpy())
            self.dataset = np.concatenate((self.dataset, cat), axis=1)

        self.batch_size = len(self)

    def __len__(self) -> int:
        """
        Getter method for acquiring the size of the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return self.dataset.shape[0]

    def num_of_features(self) -> int:
        """
        Getter method for acquiring the dimensions of the dataset.

        Returns:
            int: Number of features in the dataset
        """
        return self.dataset.shape[1]

    def __getitem__(self, indices: list[int]) -> tuple[np.ndarray, ...]:
        """
        Utility method for accessing samples of given indices in the dataset.

        Args:
            indices (list[int]): List of indices.

        Returns:
            tuple[np.ndarray, ...]: Samples, transposed
        """
        return self.dataset[indices].T, self.target[indices].T

    def batch(self, batch_size: int) -> None:
        """
        Setter method for the batch_size attribute.

        Args:
            batch_size (int): Size of the batch
        """
        self.batch_size = batch_size

    def load(self) -> tuple[np.ndarray, ...]:
        """
        Method for loading batches into the model.
        Works as a generator.

        Returns:
            tuple[np.ndarray, ...]: Batch
        """
        indices = np.arange(len(self))
        np.random.shuffle(indices)

        full_batches = len(self) // self.batch_size
        full_indices = np.reshape(indices[: full_batches * self.batch_size], (full_batches, self.batch_size))
        left_indices = indices[full_batches * self.batch_size:]

        for batch_indices in full_indices:
            yield self[batch_indices]

        yield self[left_indices]
