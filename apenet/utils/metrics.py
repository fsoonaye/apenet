# apenet/utils/metrics.py
import numpy as np


def accuracy(y_true, y_pred):
    """Computes the accuracy of a classification model.

    Parameters
    ----------
    y_true : ndarray
        True labels, which can be class indices (1D array) or one-hot encoded arrays (2D array).
    y_pred : ndarray
        Predicted labels, which can be class indices (1D array) or logits (2D array).

    Returns
    -------
    float
        The accuracy of the model.
    """
    # Both are class indices
    if y_true.ndim == 1 and y_pred.ndim == 1:  
        return np.mean(y_true == y_pred)
    
    # y_true is class indices, y_pred is logits
    elif y_true.ndim == 1 and y_pred.ndim == 2:  
        pred_indices = np.argmax(y_pred, axis=1)
        return np.mean(pred_indices == y_true)
    
    # y_true is one-hot encoded, y_pred is class indices
    elif y_true.ndim == 2 and y_pred.ndim == 1:  
        true_indices = np.argmax(y_true, axis=1)
        return np.mean(true_indices == y_pred)
    
    # Both are one-hot encoded or logits
    elif y_true.ndim == 2 and y_pred.ndim == 2:  
        true_indices = np.argmax(y_true, axis=1)
        pred_indices = np.argmax(y_pred, axis=1)
        return np.mean(pred_indices == true_indices)
    else:
        raise ValueError("Invalid shapes for y_true and y_pred")


def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between true and predicted values.
    
    Parameters
    ----------
    y_true : ndarray
        True values.
    y_pred : ndarray
        Predicted values.
        
    Returns
    -------
    float
        The mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)
