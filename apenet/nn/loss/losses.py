# apenet/loss/losses.py
import torch
from ..core.activations import Softmax

class Loss:
    """Base class for all loss functions."""
    def forward(self, y_pred, y_true):
        """Compute the loss."""
        raise NotImplementedError
    
    def backward(self):
        """Compute the gradient of the loss."""
        raise NotImplementedError
    
    def __call__(self, y_pred, y_true):
        """Alias for forward method."""
        return self.forward(y_pred, y_true)

class CrossEntropyLoss(Loss):
    """
    Cross-entropy loss function with softmax activation.
    
    L = -sum(y_true * log(softmax(y_pred)))
    """
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
    
    def forward(self, logits, y_true):
        """
        Compute the multi-class cross-entropy loss.
        
        Parameters:
        - logits: Predicted logits (before softmax), shape (batch_size, num_classes).
        - y_true: Ground-truth labels, shape (batch_size).
        
        Returns:
        - loss: Cross-entropy loss.
        """
        self.batch_size = logits.shape[0]
        self.logits = logits
        self.y_true = y_true
        
        # Compute softmax
        self.probs = self.softmax(logits)
        
        # Compute cross-entropy loss
        if y_true.ndim == 1:  # Class indices
            log_probs = torch.log(self.probs + 1e-10)  # Add small constant for numerical stability
            self.loss = -torch.mean(log_probs[range(self.batch_size), y_true])
        else:  # One-hot encoded labels
            log_probs = torch.log(self.probs + 1e-10)
            self.loss = -torch.mean(torch.sum(y_true * log_probs, dim=1))
            
        return self.loss
    
    def backward(self):
        """
        Compute the gradient of the cross-entropy loss.
        
        Returns:
        - dlogits: Gradient of the loss with respect to the logits.
        """
        if self.y_true.ndim == 1:  # Class indices
            dlogits = self.probs.clone()
            dlogits[range(self.batch_size), self.y_true] -= 1
        else:  # One-hot encoded labels
            dlogits = self.probs - self.y_true
            
        return dlogits / self.batch_size

class MSELoss(Loss):
    """
    Mean squared error loss function.
    
    L = mean((y_pred - y_true)^2)
    """
    def forward(self, y_pred, y_true):
        """
        Compute the mean squared error loss.
        
        Parameters:
        - y_pred: Predicted values.
        - y_true: Ground-truth values.
        
        Returns:
        - loss: Mean squared error loss.
        """
        self.y_pred = y_pred
        self.y_true = y_true
        self.batch_size = y_pred.shape[0]
        
        self.loss = torch.mean((y_pred - y_true) ** 2)
        return self.loss
    
    def backward(self):
        """
        Compute the gradient of the mean squared error loss.
        
        Returns:
        - doutput: Gradient of the loss with respect to the model output.
        """
        return 2 * (self.y_pred - self.y_true) / self.batch_size