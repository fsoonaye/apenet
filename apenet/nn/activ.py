# apenet/nn/activ.py
import numpy as np
import inspect
import sys
from functools import lru_cache

class Activation:
    """Base class for all activation functions."""
    def forward(self, x):
        """Forward pass of the activation function.

        Parameters
        ----------
        x : ndarray
            Input array.

        Returns
        -------
        ndarray
            Activation of the input array.
        """
        raise NotImplementedError

    def backward(self, dA):
        """Backward pass of the activation function.

        Parameters
        ----------
        dA : ndarray
            Gradient of the cost with respect to the activation.

        Returns
        -------
        ndarray
            Gradient of the cost with respect to the pre-activation.
        """
        raise NotImplementedError

    def __call__(self, x):
        """Call method to perform the forward pass.

        Parameters
        ----------
        x : ndarray
            Input array.

        Returns
        -------
        ndarray
            Activation of the input array.
        """
        return self.forward(x)

class Sigmoid(Activation):
    """Sigmoid activation function.

    Forward: f(x) = 1 / (1 + exp(-x))
    Backward: f'(x) = f(x) * (1 - f(x))
    """
    def forward(self, x):
        """Numerically stable sigmoid implementation.
        
        Parameters
        ----------
        x : ndarray
            Input array.
            
        Returns
        -------
        ndarray
            Sigmoid activation applied to input.
        """
        # Clip input to prevent overflow
        x = np.clip(x, -50.0, 50.0)
        
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

        # Ensure result is between 0 and 1
        result = np.clip(result, 1e-7, 1.0 - 1e-7)
        
        self.output = result
        return result

    def backward(self, dA):
        """Compute gradient of sigmoid activation.
        
        Parameters
        ----------
        dA : ndarray
            Gradient of the cost with respect to the activation.
            
        Returns
        -------
        ndarray
            Gradient of the cost with respect to the pre-activation.
        """
        # Clip gradient to prevent overflow
        dA = np.clip(dA, -1e10, 1e10)
        
        # Ensure output values are in a safe range
        output_safe = np.clip(self.output, 1e-7, 1.0 - 1e-7)
        
        # Compute gradient
        gradient = dA * output_safe * (1 - output_safe)
        
        # Clip gradient to prevent extreme values
        return np.clip(gradient, -1e10, 1e10)

class ReLU(Activation):
    """ReLU activation function.

    Forward: f(x) = max(0, x)
    Backward: f'(x) = 1 if x > 0 else 0
    """
    def forward(self, x):
        """Apply ReLU activation.
        
        Parameters
        ----------
        x : ndarray
            Input array.
            
        Returns
        -------
        ndarray
            ReLU activation applied to input.
        """
        # Clip input to prevent overflow
        x = np.clip(x, -1e10, 1e10)
        self.input = x
        return np.maximum(0, x)

    def backward(self, dA):
        """Compute gradient of ReLU activation.
        
        Parameters
        ----------
        dA : ndarray
            Gradient of the cost with respect to the activation.
            
        Returns
        -------
        ndarray
            Gradient of the cost with respect to the pre-activation.
        """
        # Clip gradient to prevent overflow
        dA = np.clip(dA, -1e10, 1e10)
        
        dZ = dA.copy()
        dZ[self.input <= 0] = 0
        return dZ

class Softmax(Activation):
    """Softmax activation function.

    Forward: f(x_i) = exp(x_i) / sum(exp(x_j))
    """
    def forward(self, x):
        """Apply softmax activation with numerical stability.
        
        Parameters
        ----------
        x : ndarray
            Input array.
            
        Returns
        -------
        ndarray
            Softmax activation applied to input.
        """
        # Clip input to prevent overflow
        x = np.clip(x, -50.0, 50.0)
        
        # Subtract max for numerical stability
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        
        # Compute exp and sum
        exp_x = np.exp(shifted_x)
        sum_exp = np.sum(exp_x, axis=1, keepdims=True)
        
        # Handle potential division by zero
        sum_exp = np.maximum(sum_exp, 1e-7)
        
        # Compute softmax
        self.output = exp_x / sum_exp
        
        # Ensure probabilities sum to 1 and are not 0 or 1
        self.output = np.clip(self.output, 1e-7, 1.0 - 1e-7)
        row_sums = np.sum(self.output, axis=1, keepdims=True)
        self.output = self.output / row_sums
        
        return self.output

    def backward(self, dA):
        """Not used directly; combined with CrossEntropyLoss."""
        pass

class Tanh(Activation):
    """Tanh activation function.

    Forward: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Backward: f'(x) = 1 - f(x)^2
    """
    def forward(self, x):
        """Apply tanh activation.
        
        Parameters
        ----------
        x : ndarray
            Input array.
            
        Returns
        -------
        ndarray
            Tanh activation applied to input.
        """
        # Clip input to prevent overflow
        x = np.clip(x, -50.0, 50.0)
        
        # Use numpy's tanh for numerical stability
        self.output = np.tanh(x)
        return self.output

    def backward(self, dA):
        """Compute gradient of tanh activation.
        
        Parameters
        ----------
        dA : ndarray
            Gradient of the cost with respect to the activation.
            
        Returns
        -------
        ndarray
            Gradient of the cost with respect to the pre-activation.
        """
        # Clip gradient to prevent overflow
        dA = np.clip(dA, -1e10, 1e10)
        
        # Ensure output values are in a safe range
        output_safe = np.clip(self.output, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Compute gradient
        gradient = dA * (1 - output_safe ** 2)
        
        # Clip gradient to prevent extreme values
        return np.clip(gradient, -1e10, 1e10)


@lru_cache(maxsize=1)
def get_activation_registry():
    """Auto-discover all activation classes and create a registry.
    
    Cache the result for performance.
    
    Returns
    -------
    dict
        Dictionary mapping activation names to classes.
    """
    registry = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and issubclass(obj, Activation) and obj != Activation:
            registry[name.lower()] = obj
    return registry

def get_activation(name):
    """Get activation function by name.
    
    Parameters
    ----------
    name : str
        Name of the activation function
        
    Returns
    -------
    Activation
        Instance of the activation function
        
    Raises
    ------
    ValueError
        If activation name is not recognized
    """
    if name is None:
        return None
        
    registry = get_activation_registry()
    name = name.lower()
    activation_class = registry.get(name)
    
    if activation_class is None:
        available = list(registry.keys())
        raise ValueError(f"Unsupported activation function: '{name}'. Available options: {available}")
    
    return activation_class()