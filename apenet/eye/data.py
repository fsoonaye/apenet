# apenet/eye/data.py
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    figsize=(8, 6),
    cmap="Blues",
    normalize=False,
    title="Confusion Matrix"
):
    """Plot confusion matrix.

    Parameters
    ----------
    y_true : ndarray
        The true labels.
    y_pred : ndarray
        The predicted labels.
    class_names : list, default=None
        The names of the classes. If None, will use the unique values from y_true.
    figsize : tuple, default=(8, 6)
        The figure size.
    cmap : str, default="Blues"
        The colormap to use.
    normalize : bool, default=False
        Whether to normalize the confusion matrix.
    title : str, default="Confusion Matrix"
        The title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
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
