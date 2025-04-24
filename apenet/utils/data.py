import numpy as np
from collections import Counter
from typing import Optional, Tuple, Generator

def train_test_split(X: np.ndarray,
                     y: np.ndarray,
                     test_size: float = 0.2,
                     shuffle: bool = True,
                     rng: Optional[np.random.Generator] = None,
                     seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : np.ndarray
        Input data.
    y : np.ndarray
        Labels.
    test_size : float, optional
        Proportion of data to use for testing. Default: 0.2
    shuffle : bool, optional
        Whether to shuffle the data before splitting. Default: True
    rng : np.random.Generator, optional
        Numpy random generator. Default: None
    seed : int, optional
        Seed for the random number generator. Default: None

    Returns
    -------
    X_train : np.ndarray
        Training input data.
    X_test : np.ndarray
        Test input data.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    """
    assert 0 < test_size < 1, "test_size must be between 0 and 1"

    if rng is None:
        rng = np.random.default_rng(seed)

    num_samples = X.shape[0]
    num_test = int(num_samples * test_size)
    num_train = num_samples - num_test

    if shuffle:
        indices = rng.permutation(num_samples)
        X = X[indices]
        y = y[indices]

    X_train = X[:num_train]
    X_test = X[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]

    return X_train, X_test, y_train, y_test

def get_batches(X: np.ndarray,
                y: np.ndarray,
                batch_size: int,
                shuffle: bool = True,
                rng: Optional[np.random.Generator] = None,
                seed: Optional[int] = None
    ) -> Generator[np.ndarray, np.ndarray]:
    """
    Generate mini-batches for training.

    Parameters
    ----------
    X : np.ndarray
        Input data.
    y : np.ndarray
        Labels.
    batch_size : int
        Size of each mini-batch.
    shuffle : bool, optional
        Whether to shuffle the data before creating batches. Default: True

    Yields
    ------
    X_batch : np.ndarray
        Mini-batch of input data.
    y_batch : np.ndarray
        Mini-batch of labels.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    num_samples = X.shape[0]
    indices = rng.permutation(num_samples) if shuffle else np.arange(num_samples)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]

def one_hot_encode(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert class labels to one-hot encoding.

    Parameters
    ----------
    y : np.ndarray
        Class labels.
    num_classes : int, optional
        Number of classes. Default: None

    Returns
    -------
    one_hot : np.ndarray
        One-hot encoded labels.
    """
    if num_classes is None:
        num_classes = len(np.unique(y))

    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
    return one_hot


def bootstrap_sample(X: np.ndarray,
                     y: np.ndarray,
                     rng: Optional[np.random.Generator] = None,
                     seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a bootstrap sample from the data.

    Parameters
    ----------
    X : np.ndarray
        Input data.
    y : np.ndarray
        Labels.
    rng : np.random.Generator, optional
        Numpy random generator. Default: None
    seed : int, optional
        Seed for the random number generator. Default: None

    Returns
    -------
    X_sample : np.ndarray
        Bootstrap sample of input data.
    y_sample : np.ndarray
        Bootstrap sample of labels.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    n_samples = X.shape[0]
    idxs = rng.integers(0, n_samples, size=n_samples)
    return X[idxs], y[idxs]

def standardize(X: np.ndarray) -> np.ndarray:
    """
    Standardizes the data in the array X.

    Parameters
    ----------
    X : np.ndarray
        Features array of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        The standardized features array.
    """
    # Convert to float32 for precision and memory efficiency
    X = X.astype('float32')

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std
