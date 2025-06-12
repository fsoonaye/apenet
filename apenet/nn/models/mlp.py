# apenet/nn/models/mlp.py
import numpy as np

from apenet.interfaces import BaseModel
from apenet.nn.activ import get_activation
from apenet.nn.init import get_initializer
from apenet.nn.layer import Linear
from apenet.nn.models.sequential import Sequential


class MLP(BaseModel):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        hidden_activation='relu',
        output_activation=None,
        initializer='xavier',
        seed=None,
        rng=None,
    ):
        """Multi-Layer Perceptron model.

        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_sizes : list
            List of hidden layer sizes.
        output_size : int
            Number of output features.
        hidden_activation : str, default='relu'
            Activation function for hidden layers.
        output_activation : str, default=None
            Activation function for output layer.
        initializer : str, default='xavier'
            Weight initializer for all layers.
        seed : int, default=None
            Seed for random number generator for reproducibility.
        rng : numpy.random.Generator, default=None
            NumPy random Generator instance. If None, will be created from seed.
        """
        super().__init__(seed=seed, rng=rng)
        self.sequential = Sequential()
        self._build(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            initializer=initializer,
        )

    def _build(self, input_size, hidden_sizes, output_size, hidden_activation, output_activation, initializer):
        """Build the MLP architecture.
        
        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_sizes : list
            List of hidden layer sizes.
        output_size : int
            Number of output features.
        hidden_activation : str
            Activation function for hidden layers.
        output_activation : str or None
            Activation function for output layer.
        initializer : str
            Weight initializer for all layers.
        """
        # Create layer dimensions
        layer_dims = [input_size] + hidden_sizes + [output_size]
        
        # Build network architecture
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i+1]
            linear_layer = Linear(in_dim, out_dim, get_initializer(initializer, rng = self.rng))
            self.sequential.add(linear_layer)
            
            # Add appropriate activation
            if i < len(layer_dims) - 2:  # Hidden layers
                if hidden_activation:
                    self.sequential.add(get_activation(hidden_activation))
            else:  # Output layer
                if output_activation:
                    self.sequential.add(get_activation(output_activation))

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
        """Train the model.
        
        Parameters
        ----------
        X_train : ndarray
            Training input data.
        y_train : ndarray
            Training target data.
        loss_fn : Loss
            Loss function.
        optimizer : Optimizer
            Optimizer instance.
        epochs : int, default=100
            Number of training epochs.
        batch_size : int, default=32
            Batch size for training.
        X_val : ndarray, default=None
            Validation input data.
        y_val : ndarray, default=None
            Validation target data.
        verbose : int, default=1
            Verbosity level.
            
        Returns
        -------
        dict
            Dictionary containing training history.
        """
        return self.sequential.fit(X_train,y_train, loss_fn, optimizer, epochs, batch_size, X_val, y_val, verbose)

    def predict(self, X):
        """Make predictions with the model.
        
        Parameters
        ----------
        X : ndarray
            Input data.
            
        Returns
        -------
        ndarray
            Model predictions.
        """
        return self.sequential.predict(X)

    def forward(self, x):
        """Perform forward pass through the model.
        
        Parameters
        ----------
        x : ndarray
            Input data.
            
        Returns
        -------
        ndarray
            Model output.
        """
        return self.sequential.forward(x)

    def evaluate(self, X, y, loss_fn):
        """Evaluate the model.
        
        Parameters
        ----------
        X : ndarray
            Input data.
        y : ndarray
            Target data.
        loss_fn : Loss
            Loss function.
            
        Returns
        -------
        float
            Evaluation loss.
        """
        return self.sequential.evaluate(X, y, loss_fn)

    def save(self, filepath):
        """Save model to file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        self.sequential.save(filepath)

    def load(self, filepath):
        """Load model from file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
        """
        self.sequential.load(filepath)

    def get_parameters(self):
        """Get model parameters.
        
        Returns
        -------
        list
            List of model parameters.
        """
        return self.sequential.get_parameters()
