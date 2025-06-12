# apenet/eye/history.py
import numpy as np
import matplotlib.pyplot as plt

def plot_history(history):
    """Plot all available metrics in history dict.
    
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