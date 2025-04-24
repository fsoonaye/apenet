# apenet/utils/metrics.py
import numpy as np
from numpy.typing import ArrayLike

import numpy as np
from numpy.typing import ArrayLike

def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Computes the accuracy of a classification model.

    Parameters
    ----------
    y_true : ArrayLike
        True labels, which can be class indices (1D array) or one-hot encoded arrays (2D array).
    y_pred : ArrayLike
        Predicted labels, which can be class indices (1D array) or logits/class probabilities (2D array).

    Returns
    -------
    float
        The accuracy of the model.
    """
    if y_true.ndim == 1 and y_pred.ndim == 1:  # Both are class indices
        return np.mean(y_true == y_pred)
    elif y_true.ndim == 1 and y_pred.ndim == 2:  # y_true is class indices, y_pred is logits/probabilities
        pred_indices = np.argmax(y_pred, axis=1)
        return np.mean(pred_indices == y_true)
    elif y_true.ndim == 2 and y_pred.ndim == 1:  # y_true is one-hot encoded, y_pred is class indices
        true_indices = np.argmax(y_true, axis=1)
        return np.mean(true_indices == y_pred)
    elif y_true.ndim == 2 and y_pred.ndim == 2:  # Both are one-hot encoded or logits/probabilities
        true_indices = np.argmax(y_true, axis=1)
        pred_indices = np.argmax(y_pred, axis=1)
        return np.mean(pred_indices == true_indices)
    else:
        raise ValueError("Invalid shapes for y_true and y_pred")


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute the mean squared error of the predictions.

    Parameters
    ----------
    y_true : ArrayLike
        True values.
    y_pred : ArrayLike
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)
