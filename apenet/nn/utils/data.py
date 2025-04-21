# apenet/utils/data.py
import torch


def train_test_split(X, y, test_size=0.2, shuffle=True):
    """
    Split data into training and test sets.
    
    Parameters:
    - X: Input data.
    - y: Labels.
    - test_size: Proportion of data to use for testing.
    - shuffle: Whether to shuffle the data before splitting.
    
    Returns:
    - X_train: Training input data.
    - X_test: Test input data.
    - y_train: Training labels.
    - y_test: Test labels.
    """
    assert 0 < test_size < 1, "test_size must be between 0 and 1"
    
    num_samples = X.shape[0]
    num_test = int(num_samples * test_size)
    num_train = num_samples - num_test
    
    if shuffle:
        indices = torch.randperm(num_samples)
        X = X[indices]
        y = y[indices]
    
    X_train = X[:num_train]
    X_test = X[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]
    
    return X_train, X_test, y_train, y_test

def get_batches(X, y, batch_size, shuffle=True):
    """
    Generate mini-batches for training.
    
    Parameters:
    - X: Input data.
    - y: Labels.
    - batch_size: Size of each mini-batch.
    - shuffle: Whether to shuffle the data before creating batches.
    
    Yields:
    - X_batch: Mini-batch of input data.
    - y_batch: Mini-batch of labels.
    """
    num_samples = X.shape[0]
    indices = torch.randperm(num_samples) if shuffle else torch.arange(num_samples)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]

def one_hot_encode(y, num_classes=None):
    """
    Convert class labels to one-hot encoding.
    
    Parameters:
    - y: Class labels.
    - num_classes: Number of classes. If None, use the maximum value in y + 1.
    
    Returns:
    - one_hot: One-hot encoded labels.
    """
    if num_classes is None:
        num_classes = int(torch.max(y)) + 1
    
    one_hot = torch.zeros(y.shape[0], num_classes, device=y.device)
    one_hot[torch.arange(y.shape[0]), y.long()] = 1
    return one_hot