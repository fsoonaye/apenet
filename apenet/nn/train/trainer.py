# apenet/train/trainer.py
import torch
from ..utils.metrics import accuracy

class Trainer:
    """
    Class for training neural network models.
    """
    def __init__(self, model, loss_fn, optimizer):
        """
        Initialize the trainer.

        Parameters:
        - model: Neural network model to train.
        - loss_fn: Loss function.
        - optimizer: Optimizer.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history = None

    def train(self, X_train, y_train, epochs=100, batch_size=32,
              X_val=None, y_val=None, verbose=1):
        """
        Train the model.

        Parameters:
        - X_train: Training input data.
        - y_train: Training labels.
        - epochs: Number of training epochs.
        - batch_size: Size of mini-batches.
        - X_val: Validation input data.
        - y_val: Validation labels.
        - verbose: Verbosity level.

        Returns:
        - history: Dictionary containing training and validation metrics.
        """
        self.history = self.model.train(
            X_train, y_train,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            epochs=epochs,
            batch_size=batch_size,
            X_val=X_val,
            y_val=y_val,
            verbose=verbose
        )

        return self.history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.

        Parameters:
        - X_test: Test input data.
        - y_test: Test labels.

        Returns:
        - metrics: Dictionary containing evaluation metrics.
        """
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, self.loss_fn)

        metrics = {
            'loss': test_loss,
            'accuracy': test_accuracy,
        }

        return metrics
