# apenet/core/activations.py
from functools import lru_cache
import inspect
import sys
import numpy as np

class Activation:
    """Base class for all activation functions."""
    def forward(self, x):
        """
        Forward pass of the activation function.

        Parameters:
        - x: Input array.

        Returns:
        - Activation of the input array.
        """
        raise NotImplementedError

    def backward(self, dA):
        """
        Backward pass of the activation function.

        Parameters:
        - dA: Gradient of the cost with respect to the activation.

        Returns:
        - Gradient of the cost with respect to the pre-activation.
        """
        raise NotImplementedError

    def __call__(self, x):
        """
        Call method to perform the forward pass.

        Parameters:
        - x: Input array.

        Returns:
        - Activation of the input array.
        """
        return self.forward(x)

class Sigmoid(Activation):
    """
    Sigmoid activation function.

    Forward: f(x) = 1 / (1 + exp(-x))
    Backward: f'(x) = f(x) * (1 - f(x))
    """
    def forward(self, x):
        """
        Numerically stable sigmoid implementation.
        """

        # Initialize the output array
        result = np.empty_like(x)

        # Mask for positive and negative values
        positive = x >= 0
        negative = ~positive

        # Compute sigmoid for positive values
        result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))

        # Compute sigmoid for negative values
        exp_x = np.exp(x[negative])
        result[negative] = exp_x / (exp_x + 1.0)

        self.output = result
        return result

    def backward(self, dA):
        return dA * self.output * (1 - self.output)

class ReLU(Activation):
    """
    ReLU activation function.

    Forward: f(x) = max(0, x)
    Backward: f'(x) = 1 if x > 0 else 0
    """
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, dA):
        dZ = dA.copy()
        dZ[self.input <= 0] = 0
        return dZ

class Softmax(Activation):
    """
    Softmax activation function.

    Forward: f(x_i) = exp(x_i) / sum(exp(x_j))
    """
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, dA):
        pass  # Not used directly; combined with CrossEntropyLoss

class Tanh(Activation):
    """
    Tanh activation function.

    Forward: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Backward: f'(x) = 1 - f(x)^2
    """
    def forward(self, x):
        exp_x = np.exp(x)
        exp_neg_x = np.exp(-x)
        self.output = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
        return self.output

    def backward(self, dA):
        return dA * (1 - self.output ** 2)


@lru_cache(maxsize=1)
def get_activation_registry():
    """
    Auto-discover all activation classes and create a registry.
    Cache the result for performance.
    """
    registry = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and issubclass(obj, Activation) and obj != Activation:
            registry[name.lower()] = obj
    return registry

def get_activation(name):
    if name is None:
        return None
        
    registry = get_activation_registry()
    name = name.lower()
    activation_class = registry.get(name)
    
    if activation_class is None:
        available = list(registry.keys())
        raise ValueError(f"Unsupported activation function: '{name}'. Available options: {available}")
    
    return activation_class()