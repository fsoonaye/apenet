import numpy as np

from apenet.utils.metrics import accuracy
from apenet.utils.data import get_batches
from apenet.nn.activ import Activation
from apenet.nn.layer import Layer
from apenet.nn.utils import should_print_epoch, print_epoch_status


class Sequential:
    """Sequential container for stacking layers.

    Data flows through layers in sequence during forward and backward passes.
    """

    def __init__(self, layers=None):
        """Initialize the Sequential model.

        Parameters
        ----------
        layers : list, default=None
            List of initial layers (Layer or Activation).
        """
        self.layers = layers if layers is not None else []

    def add(self, layer):
        """Add a layer or activation to the model.

        Parameters
        ----------
        layer : Layer or Activation
            Layer or Activation instance to be added.

        Raises
        ------
        TypeError
            If the provided object is not a Layer or Activation.
        """
        if not (isinstance(layer, Layer) or isinstance(layer, Activation)):
            raise TypeError("Layer must be an instance of Layer or Activation class")
        self.layers.append(layer)

    def forward(self, x):
        """Forward pass through all layers.

        Parameters
        ----------
        x : ndarray
            Input array.

        Returns
        -------
        ndarray
            Output array.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        """Backward pass through all layers.

        Parameters
        ----------
        grad_output : ndarray
            Gradient with respect to the model output.

        Returns
        -------
        ndarray
            Gradient with respect to the model input.
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_parameters(self):
        """Get the list of parameters from each layer.

        Returns
        -------
        list
            List of parameters from each layer.
        """
        return self.layers

    def fit(
        self,
        X_train,
        y_train,
        loss_fn,
        optimizer,
        epochs=100,
        batch_size=32,
        X_val=None,
        y_val=None,
        verbose=1,
    ):
        """Train the model using mini-batch gradient descent.

        Parameters
        ----------
        X_train : ndarray
            Training input data.
        y_train : ndarray
            Training labels.
        loss_fn : Loss
            Loss function.
        optimizer : Optimizer
            Optimizer for parameter updates.
        epochs : int, default=100
            Number of training epochs.
        batch_size : int, default=32
            Number of samples per batch.
        X_val : ndarray, default=None
            Validation input data.
        y_val : ndarray, default=None
            Validation labels.
        verbose : int, default=1
            Verbosity level.

        Returns
        -------
        dict
            Training and validation history (loss and accuracy per epoch).
        """
        history = {'train_loss': [], 'train_accuracy': []}
        if X_val is not None and y_val is not None:
            history['val_loss'] = []
            history['val_accuracy'] = []

        for epoch in range(epochs):
            epoch_loss = 0
            correct_preds = 0
            num_samples = 0

            # Mini-batch training
            for X_batch, y_batch in get_batches(X_train, y_train, batch_size, shuffle=True):
                curr_batch_size = X_batch.shape[0]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                loss = loss_fn(y_pred, y_batch)
                epoch_loss += loss.item() * curr_batch_size

                # Compute accuracy
                correct_preds += accuracy(y_true=y_batch, y_pred=y_pred) * curr_batch_size
                num_samples += curr_batch_size

                # Backward pass
                grad_output = loss_fn.backward()
                self.backward(grad_output)

                # Update parameters
                optimizer.step()

            # Calculate epoch metrics
            epoch_loss /= num_samples
            epoch_accuracy = correct_preds / num_samples

            history['train_loss'].append(epoch_loss)
            history['train_accuracy'].append(epoch_accuracy)

            # Validation
            if X_val is not None and y_val is not None:
                val_loss, val_accuracy = self.evaluate(X_val, y_val, loss_fn)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

            # Verbose printing
            if verbose and should_print_epoch(epoch, epochs, verbose):
                print_epoch_status(
                    epoch, epochs, epoch_loss, epoch_accuracy,
                    val_loss if X_val is not None else None,
                    val_accuracy if X_val is not None else None
                )

        return history

    def evaluate(self, X, y, loss_fn):
        """Evaluate the model.

        Parameters
        ----------
        X : ndarray
            Input data.
        y : ndarray
            True labels.
        loss_fn : Loss
            Loss function.

        Returns
        -------
        tuple
            Tuple of (loss, accuracy).
        """
        y_pred = self.forward(X)
        loss = loss_fn(y_pred, y)
        accuracy_val = accuracy(y_true=y, y_pred=y_pred)
        return float(loss.item()), float(accuracy_val)

    def predict(self, X):
        """Make predictions.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray
            Predicted class indices.
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def save(self, filepath):
        """Save the model parameters to a file.

        Parameters
        ----------
        filepath : str
            Path of the file where parameters are saved.
        """
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'parameters'):
                for param_name, param_value in layer.parameters.items():
                    params[f"layer_{i}_{param_name}"] = param_value

        np.savez(filepath, **params)

    def load(self, filepath):
        """Load the model parameters from a file.

        Parameters
        ----------
        filepath : str
            Path of the file to load parameters from.
        """
        params = np.load(filepath)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'parameters'):
                for param_name in layer.parameters:
                    key = f"layer_{i}_{param_name}"
                    if key in params:
                        layer.parameters[param_name] = params[key]
