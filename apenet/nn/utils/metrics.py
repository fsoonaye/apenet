# apenet/utils/metrics.py
import numpy as np

def accuracy(y_true, y_pred):
    """
    Compute the accuracy of the predictions.

    Parameters:
    - y_true: True class labels.
    - y_pred: Predicted class labels.

    Returns:
    - accuracy: Computed accuracy.
    """
    if y_true.ndim == 1:  # Class indices
        return np.mean(y_pred == y_true)
    else:  # One-hot encoded
        y_indices = np.argmax(y_true, axis=1)
        return np.mean(y_pred == y_indices)

def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared error of the predictions.

    Parameters:
    - y_true: True values.
    - y_pred: Predicted values.

    Returns:
    - mse: Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)
