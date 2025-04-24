# apenet/nn/models/mlp.py
from apenet.nn.models.sequential import Sequential
from apenet.nn.core.layers import Linear
from apenet.nn.core.activations import get_activation
from apenet.nn.initialize.initializers import get_initializer
from apenet.interfaces.base_model import BaseModel

import numpy as np

from apenet.nn.loss.losses import Loss
from apenet.nn.optimizers.optimizers import Optimizer
from typing import List, Optional

class MLP(BaseModel):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        hidden_activation: str = 'relu',
        output_activation: Optional[str] = None,
        initializer: str = 'xavier',
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Multi-Layer Perceptron model.

        Parameters
        ----------
        input_size : int
            Number of input features.
        hidden_sizes : List[int]
            List of hidden layer sizes.
        output_size : int
            Number of output features.
        hidden_activation : str, optional
            Activation function for hidden layers, by default 'relu'.
        output_activation : Optional[str], optional
            Activation function for output layer, by default None.
        initializer : str, optional
            Weight initializer for all layers, by default 'xavier'.
        seed : Optional[int], optional
            Seed for random number generator for reproducibility, by default None.
        rng : Optional[np.random.Generator], optional
            NumPy random Generator instance. If None, will be created from seed. By default None.
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
        """
        Build the MLP architecture.
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
        X_train: np.ndarray,
        y_train: np.ndarray,
        loss_fn: Loss,
        optimizer: Optimizer,
        epochs: int = 100,
        batch_size: int = 32,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 1,
    ) -> dict[str, List[float]]:
        return self.sequential.fit(X_train, y_train, loss_fn, optimizer, epochs, batch_size, X_val, y_val, verbose)

    def predict(self, X):
        return self.sequential.predict(X)

    def forward(self, x):
        return self.sequential.forward(x)

    def evaluate(self, X, y, loss_fn):
        return self.sequential.evaluate(X, y, loss_fn)

    def save(self, filepath):
        self.sequential.save(filepath)

    def load(self, filepath):
        self.sequential.load(filepath)

    def get_parameters(self):
        return self.sequential.get_parameters()
