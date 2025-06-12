# apenet/utils/data.py
import numpy as np

def train_test_split(X,
                     y,
                     test_size=0.2,
                     shuffle=True,
                     rng=None,
                     seed=None
    ):
    """Split data into training and test sets.

    Parameters
    ----------
    X : ndarray
        Input data.
    y : ndarray
        Labels.
    test_size : float, default=0.2
        Proportion of data to use for testing. 
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    rng : numpy.random.Generator, default=None
        Numpy random generator.
    seed : int, default=None
        Seed for the random number generator.

    Returns
    -------
    X_train : ndarray
        Training input data.
    X_test : ndarray
        Test input data.
    y_train : ndarray
        Training labels.
    y_test : ndarray
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

def get_batches(X,
                y,
                batch_size,
                shuffle=True,
                rng=None,
                seed=None
    ):
    """Generate mini-batches for training.

    Parameters
    ----------
    X : ndarray
        Input data.
    y : ndarray
        Labels.
    batch_size : int
        Size of each mini-batch.
    shuffle : bool, default=True
        Whether to shuffle the data before creating batches.
    rng : numpy.random.Generator, default=None
        Numpy random generator.
    seed : int, default=None
        Seed for the random number generator.

    Yields
    ------
    X_batch : ndarray
        Mini-batch of input data.
    y_batch : ndarray
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

def one_hot_encode(y, num_classes=None):
    """Convert class labels to one-hot encoding.

    Parameters
    ----------
    y : ndarray
        Class labels.
    num_classes : int, default=None
        Number of classes.

    Returns
    -------
    one_hot : ndarray
        One-hot encoded labels.
    """
    if num_classes is None:
        num_classes = len(np.unique(y))

    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
    return one_hot


def bootstrap_sample(X,
                     y,
                     rng=None,
                     seed=None
    ):
    """Generate a bootstrap sample from the data.

    Parameters
    ----------
    X : ndarray
        Input data.
    y : ndarray
        Labels.
    rng : numpy.random.Generator, default=None
        Numpy random generator.
    seed : int, default=None
        Seed for the random number generator.

    Returns
    -------
    X_sample : ndarray
        Bootstrap sample of input data.
    y_sample : ndarray
        Bootstrap sample of labels.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    n_samples = X.shape[0]
    idxs = rng.integers(0, n_samples, size=n_samples)
    return X[idxs], y[idxs]

def standardize(X, eps=1e-8):
    """Standardizes the data in the array X.
    
    Parameters
    ----------
    X : ndarray
        Features array of shape (n_samples, n_features).
    epsilon : float, default=1e-8
        Small value to prevent division by zero.
        
    Returns
    -------
    ndarray
        The standardized features array.
    """
    X = X.astype('float32')
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Use epsilon instead of setting std=1 for zero-variance features
    std = np.where(std < eps, eps, std)
    
    return (X - mean) / std
