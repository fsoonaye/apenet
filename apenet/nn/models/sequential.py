# apenet/models/sequential.py
import numpy as np
from ..core.layers import Layer
from ..core.activations import Activation
from ..utils.data import get_batches
from ..utils.metrics import accuracy

class Sequential:
    """
    Sequential container for stacking layers.

    The Sequential model is a linear stack of layers where data flows through
    the layers in sequence during forward and backward passes.
    """
    def __init__(self, layers=None):
        """
        Initialize a Sequential model.

        Parameters:
        - layers: List of layers (optional).
        """
        self.layers = layers if layers is not None else []

    def add(self, layer):
        """
        Add a layer to the model.

        Parameters:
        - layer: Layer to add to the model.
        """
        if not (isinstance(layer, Layer) or isinstance(layer, Activation)):
            raise TypeError("Layer must be an instance of Layer or Activation class")
        self.layers.append(layer)

    def forward(self, x):
        """
        Forward pass through all layers.

        Parameters:
        - x: Input array.

        Returns:
        - output: Output array.
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output):
        """
        Backward pass through all layers.

        Parameters:
        - grad_output: Gradient of the loss with respect to the model output.

        Returns:
        - grad_input: Gradient of the loss with respect to the model input.
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_parameters(self):
        """
        Get all trainable parameters.

        Returns:
        - parameters: List of parameter dictionaries from each layer.
        """
        return self.layers

    def train(self, X_train, y_train, loss_fn, optimizer, epochs=100, batch_size=32,
            X_val=None, y_val=None, verbose=1):
        """
        Train the model.

        Parameters:
        - X_train: Training input data.
        - y_train: Training labels.
        - loss_fn: Loss function.
        - optimizer: Optimizer.
        - epochs: Number of training epochs.
        - batch_size: Size of mini-batches.
        - X_val: Validation input data.
        - y_val: Validation labels.
        - verbose: Verbosity level.

        Returns:
        - history: Dictionary containing training and validation metrics.
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
                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                loss = loss_fn(y_pred, y_batch)
                epoch_loss += loss.item() * X_batch.shape[0]

                # Compute accuracy
                correct_preds += accuracy(y_batch, np.argmax(y_pred, axis=1)) * X_batch.shape[0]
                num_samples += X_batch.shape[0]

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
            if verbose > 0 and (epoch % verbose == 0 or epoch == epochs - 1):
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch+1}/{epochs}: train_loss={epoch_loss:.4f}, train_accuracy={epoch_accuracy:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}: train_loss={epoch_loss:.4f}, train_accuracy={epoch_accuracy:.4f}")

        return history

    def evaluate(self, X, y, loss_fn):
        """
        Evaluate the model.

        Parameters:
        - X: Input data.
        - y: Labels.
        - loss_fn: Loss function.

        Returns:
        - loss: Loss value.
        - accuracy: Accuracy value.
        """
        # Forward pass
        y_pred = self.forward(X)

        # Compute loss
        loss = loss_fn(y_pred, y)

        # Compute accuracy
        accuracy_value = accuracy(y, np.argmax(y_pred, axis=1))

        return loss.item(), accuracy_value

    def predict(self, X):
        """
        Make predictions.

        Parameters:
        - X: Input data.

        Returns:
        - predictions: Predicted class indices.
        """
        # Forward pass
        y_pred = self.forward(X)

        # Get predicted class
        return np.argmax(y_pred, axis=1)

    def save(self, filepath):
        """
        Save the model parameters.

        Parameters:
        - filepath: Path to save the model.
        """
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'parameters'):
                for param_name, param_value in layer.parameters.items():
                    params[f"layer_{i}_{param_name}"] = param_value

        np.savez(filepath, **params)

    def load(self, filepath):
        """
        Load the model parameters.

        Parameters:
        - filepath: Path to load the model from.
        """
        params = np.load(filepath)

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'parameters'):
                for param_name in layer.parameters:
                    key = f"layer_{i}_{param_name}"
                    if key in params:
                        layer.parameters[param_name] = params[key]
