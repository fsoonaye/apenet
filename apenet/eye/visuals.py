import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any, Union

def plot_history(history):
    """
    Plot all available metrics in history dict.
    Supports 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', etc.
    Will only plot curves found in the history.
    """

    # Attempt to find any train/val metrics
    train_keys = [k for k in history if k.startswith('train_')]
    val_keys = [k for k in history if k.startswith('val_')]

    # Group by metric name (e.g., 'loss', 'accuracy')
    def extract_metric(keys, prefix):
        return {k[len(prefix):].lstrip('_'): k for k in keys}

    train_metrics = extract_metric(train_keys, 'train_')
    val_metrics   = extract_metric(val_keys, 'val_')

    # Only plot metrics present in either train or validation
    all_metrics = sorted(set(train_metrics) | set(val_metrics))
    if not all_metrics:
        print("No train_*/val_* keys found in history!")
        return

    n_metrics = len(all_metrics)
    plt.figure(figsize=(6 * n_metrics, 4))
    for i, metric in enumerate(all_metrics, 1):
        plt.subplot(1, n_metrics, i)
        if metric in train_metrics:
            plt.plot(history[train_metrics[metric]], label=f'Train {metric.capitalize()}')
        if metric in val_metrics:
            plt.plot(history[val_metrics[metric]], label=f'Validation {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs. Epoch')
        plt.legend()
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    normalize: bool = False,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.
    class_names : Optional[List[str]], default=None
        The names of the classes. If None, will use the unique values from y_true.
    figsize : Tuple[int, int], default=(8, 6)
        The figure size.
    cmap : str, default="Blues"
        The colormap to use.
    normalize : bool, default=False
        Whether to normalize the confusion matrix.
    title : str, default="Confusion Matrix"
        The title of the plot.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    # Get unique classes
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_classes)

    # Map classes to indices
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}

    # Setup class names
    if class_names is None:
        class_names = [str(cls) for cls in unique_classes]
    elif len(class_names) != n_classes:
        class_names = [str(cls) for cls in unique_classes]

    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Fill confusion matrix
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1

    # Normalize if requested
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1)
        cm = np.divide(cm, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set ticks and labels
    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm[i, j] > thresh else "black"
            text = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            ax.text(j, i, text, ha="center", va="center", color=color)

    # Labels and title
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_feature_distributions(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    bins: int = 30,
    alpha: float = 0.7
) -> plt.Figure:
    """
    Plot feature distributions for each class.
    
    Parameters
    ----------
    X : np.ndarray
        The input data.
    y : np.ndarray
        The target labels.
    feature_names : Optional[List[str]], default=None
        The names of the features. If None, will use "Feature X" where X is the index.
    class_names : Optional[List[str]], default=None
        The names of the classes. If None, will use the unique values from y.
    n_cols : int, default=3
        The number of columns in the subplot grid.
    figsize : Optional[Tuple[int, int]], default=None
        The figure size. If None, will be determined automatically.
    bins : int, default=30
        The number of bins for the histograms.
    alpha : float, default=0.7
        The transparency of the histograms.
        
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    # Get dimensions
    n_samples, n_features = X.shape
    
    # Setup feature names
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]
    
    # Setup class names
    unique_classes = np.unique(y)
    if class_names is None:
        class_names = [f"Class {cls}" for cls in unique_classes]
    
    # Calculate number of rows
    n_rows = int(np.ceil(n_features / n_cols))
    
    # Set figsize if not provided
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)
    
    # Create plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easier iteration
    axes = np.array(axes).flatten()
    
    # Plot histograms for each feature
    for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        if i < n_features:
            for j, cls in enumerate(unique_classes):
                # Get data for this class
                class_data = X[y == cls, i]
                
                # Plot histogram
                ax.hist(class_data, bins=bins, alpha=alpha, label=class_names[j])
            
            # Set title and labels
            ax.set_title(feature_name)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            
            # Add legend
            if i == 0 or i % n_cols == 0:
                ax.legend()
        else:
            # Hide unused subplots
            ax.axis('off')
    
    plt.tight_layout()
    return fig