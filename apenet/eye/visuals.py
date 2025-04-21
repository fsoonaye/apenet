import matplotlib.pyplot as plt

def plot_history(history, train_prefix="train", val_prefix="val"):
    """
    Plot training and validation loss/accuracy curves from history dict.
    Args:
      history (dict): like {
          'train_loss': [...], 'val_loss': [...],
          'train_accuracy': [...], 'val_accuracy': [...]
      }
    """
    if not (train_prefix + "_loss" in history and val_prefix + "_loss" in history):
        raise ValueError("History must contain 'train_loss' and 'val_loss'.")
    if not (train_prefix + "_accuracy" in history and val_prefix + "_accuracy" in history):
        raise ValueError("History must contain 'train_accuracy' and 'val_accuracy'.")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history[train_prefix + '_loss'], label='Train Loss')
    plt.plot(history[val_prefix + '_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history[train_prefix + '_accuracy'], label='Train Accuracy')
    plt.plot(history[val_prefix + '_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()
