# apenet/models/builder.py
from ..core.layers import Linear
from ..core.activations import Sigmoid, ReLU, Softmax, Tanh
from ..initialize.initializers import HeInitializer, XavierInitializer, ZerosInitializer, NormalInitializer

from .sequential import Sequential

class ModelBuilder:
    """
    Helper class to build neural network models with configurable parameters.
    """
    def __init__(self):
        self.model = Sequential()

    def add_layer(self, input_size, output_size, activation='relu', initializer='xavier'):
        """
        Add a layer to the model.

        Parameters:
        - input_size: Number of input features.
        - output_size: Number of output features.
        - activation: Activation function ('relu', 'sigmoid', 'softmax', 'tanh' or None).
        - initializer: Weight initializer ('he', 'xavier', 'zeros', 'normal', or None).

        Returns:
        - self: For method chaining.
        """
        # Handle initializer
        initializer_map = {
            'he': HeInitializer(),
            'xavier': XavierInitializer(),
            'zeros': ZerosInitializer(),
            'normal': NormalInitializer()
        }

        if initializer is not None:
            try:
                initializer = initializer_map[initializer.lower()]
            except KeyError:
                raise ValueError(f"Unsupported initializer: {initializer}")

        self.model.add(Linear(input_size, output_size, initializer=initializer))

        # Handle activation
        activation_map = {
            'relu': ReLU(),
            'sigmoid': Sigmoid(),
            'softmax': Softmax(),
            'tanh': Tanh()
        }

        if activation is not None:
            try:
                self.model.add(activation_map[activation.lower()])
            except KeyError:
                raise ValueError(f"Unsupported activation function: {activation}")

        return self

    def build_mlp(self, input_size, hidden_sizes, output_size, hidden_activation='relu', output_activation=None, initializer=None):
        """
        Build a multi-layer perceptron model.

        Parameters:
        - input_size: Number of input features.
        - hidden_sizes: List of hidden layer sizes.
        - output_size: Number of output features.
        - hidden_activation: Activation function for hidden layers.
        - output_activation: Activation function for output layer.
        - initializer: Weight initializer for all layers.

        Returns:
        - model: Sequential model.
        """
        self.model = Sequential()

        # Input layer to first hidden layer
        self.add_layer(input_size, hidden_sizes[0],
                       activation=hidden_activation,
                       initializer=initializer)

        # Additional hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.add_layer(hidden_sizes[i], hidden_sizes[i+1],
                           activation=hidden_activation,
                           initializer=initializer)

        # Final hidden layer to output layer
        self.add_layer(hidden_sizes[-1], output_size,
                       activation=output_activation,
                       initializer=initializer)

        return self.model

    def get_model(self):
        """
        Get the built model.

        Returns:
        - model: Sequential model.
        """
        return self.model
