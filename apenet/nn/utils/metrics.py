# apenet/utils/metrics.py
import torch

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
        return torch.mean((y_pred == y_true).float()).item()
    else:  # One-hot encoded
        y_indices = torch.argmax(y_true, dim=1)
        return torch.mean((y_pred == y_indices).float()).item()

def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared error of the predictions.

    Parameters:
    - y_true: True values.
    - y_pred: Predicted values.

    Returns:
    - mse: Mean squared error.
    """
    return torch.mean((y_true - y_pred) ** 2).item()
