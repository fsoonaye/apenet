# apenet/core/activations.py
import torch

class Activation:
    """Base class for all activation functions."""
    def forward(self, x):
        raise NotImplementedError
        
    def backward(self, dA):
        raise NotImplementedError
        
    def __call__(self, x):
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
        # Initialize the output tensor
        result = torch.empty_like(x)
        
        # Mask for positive and negative values
        positive = x >= 0
        negative = ~positive
        
        # Compute sigmoid for positive values
        result[positive] = 1.0 / (1.0 + torch.exp(-x[positive]))
        
        # Compute sigmoid for negative values
        exp_x = torch.exp(x[negative])
        result[negative] = exp_x / (exp_x + 1.0)
        
        self.output = result
        return result
    
    def backward(self, dA):
        """
        Compute the gradient of the sigmoid activation.
        
        Parameters:
        - dA: Gradient of the cost with respect to the activation.
        
        Returns:
        - dZ: Gradient of the cost with respect to the pre-activation.
        """
        return dA * self.output * (1 - self.output)

class ReLU(Activation):
    """
    ReLU activation function.
    
    Forward: f(x) = max(0, x)
    Backward: f'(x) = 1 if x > 0 else 0
    """
    def forward(self, x):
        """
        ReLU activation function.
        
        Parameters:
        - x: Input tensor.
        
        Returns:
        - ReLU of the input tensor.
        """
        self.input = x
        return torch.maximum(torch.tensor(0, device=x.device), x)
    
    def backward(self, dA):
        """
        Compute the gradient of the ReLU activation.
        
        Parameters:
        - dA: Gradient of the cost with respect to the activation.
        
        Returns:
        - dZ: Gradient of the cost with respect to the pre-activation.
        """
        dZ = dA.clone()
        dZ[self.input <= 0] = 0
        return dZ

class Softmax(Activation):
    """
    Softmax activation function.
    
    Forward: f(x_i) = exp(x_i) / sum(exp(x_j))
    """
    def forward(self, x):
        """
        Compute the softmax of the input tensor.
        
        Parameters:
        - x: Input tensor.
        
        Returns:
        - softmax: Softmax of the input tensor.
        """
        exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])  # Subtract max for numerical stability
        self.output = exp_x / torch.sum(exp_x, dim=1, keepdim=True)
        return self.output
    
    def backward(self, dA):
        """
        For softmax, this is typically combined with cross-entropy loss,
        so we don't implement a separate backward pass here.
        """
        pass  # Not used directly; combined with CrossEntropyLoss

class Tanh(Activation):
    """
    Tanh activation function.

    Forward: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Backward: f'(x) = 1 - f(x)^2
    """
    def forward(self, x):
        """
        Tanh activation function.

        Parameters:
        - x: Input tensor.

        Returns:
        - Tanh of the input tensor.
        """
        exp_x = torch.exp(x)
        exp_neg_x = torch.exp(-x)
        self.output = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
        return self.output

    def backward(self, dA):
        """
        Compute the gradient of the Tanh activation.

        Parameters:
        - dA: Gradient of the cost with respect to the activation.

        Returns:
        - dZ: Gradient of the cost with respect to the pre-activation.
        """
        return dA * (1 - self.output ** 2)