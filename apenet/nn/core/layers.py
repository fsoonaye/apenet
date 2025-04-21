# mydeeplib/core/layers.py
import torch
from ..initialize.initializers import HeInitializer

class Layer:
    """Base class for all layers."""
    def __init__(self):
        self.parameters = {}
        self.gradients = {}
        self.cache = {}
    
    def forward(self, x):
        """Forward pass computation."""
        raise NotImplementedError
    
    def backward(self, dA):
        """Backward pass computation."""
        raise NotImplementedError
    
    def update_parameters(self, learning_rate):
        """Update parameters using gradients."""
        for key in self.parameters:
            self.parameters[key] -= learning_rate * self.gradients.get(f"d{key}", 0)
    
    def get_parameters(self):
        """Get layer parameters."""
        return self.parameters
    
    def get_gradients(self):
        """Get layer gradients."""
        return self.gradients

class Linear(Layer):
    """
    Fully connected layer.
    
    y = x @ W + b
    """
    def __init__(self, in_features, out_features, initializer=None):
        """
        Initialize a fully connected layer.
        
        Parameters:
        - in_features: Number of input features.
        - out_features: Number of output features.
        - initializer: Weight initialization strategy (default: HeInitializer).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and biases
        if initializer is None:
            initializer = HeInitializer()
            
        self.parameters = {
            'W': initializer.initialize((in_features, out_features)),
            'b': torch.zeros(1, out_features, device=initializer.device)
        }
        
    def forward(self, x):
        """
        Forward pass for the linear layer.
        
        Parameters:
        - x: Input tensor of shape (batch_size, in_features).
        
        Returns:
        - output: Output tensor of shape (batch_size, out_features).
        """
        self.cache['A_prev'] = x
        return x @ self.parameters['W'] + self.parameters['b']
    
    def backward(self, dZ):
        """
        Backward pass for the linear layer.
        
        Parameters:
        - dZ: Gradient of the cost with respect to the output of this layer.
        
        Returns:
        - dA_prev: Gradient of the cost with respect to the input of this layer.
        """
        A_prev = self.cache['A_prev']
        m = A_prev.shape[0]
        
        self.gradients['dW'] = (A_prev.T @ dZ) / m
        self.gradients['db'] = torch.sum(dZ, dim=0, keepdim=True) / m
        
        return dZ @ self.parameters['W'].T