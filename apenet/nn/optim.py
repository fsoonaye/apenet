# apenet/nn/optim.py
import numpy as np

class Optimizer:
    """Base class for all optimizers.
    
    This class defines the interface that all optimizer classes must implement.
    """
    def __init__(self, parameters):
        """Initialize the optimizer.
        
        Parameters
        ----------
        parameters : list
            List of parameter dictionaries from each layer.
        """
        self.parameters = parameters

    def step(self):
        """Update parameters using gradients."""
        raise NotImplementedError

    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for layer in self.parameters:
            if hasattr(layer, 'gradients'):
                for grad in layer.gradients.values():
                    grad.fill(0)

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.
    
    Parameters are updated as:
    velocity = momentum * velocity - learning_rate * gradient
    θ = θ + velocity
    """
    def __init__(self, parameters, learning_rate=0.01, momentum=0.9):
        """
        Parameters
        ----------
        parameters : list
            List of parameter dictionaries from each layer.
        learning_rate : float, default=0.01
            Learning rate for gradient descent.
        momentum : float, default=0.9
            Momentum factor.
        """
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def step(self):
        """Update parameters using gradients with momentum.
        
        For each parameter, computes the momentum update and applies it.
        """
        for layer in self.parameters:
            # skip activation layers
            if not hasattr(layer, "get_parameters"):
                continue

            layer_params = layer.get_parameters()
            layer_grads = layer.get_gradients()

            for param_name in layer_params:
                gradient_name = f"d{param_name}"
                if gradient_name in layer_grads:
                    # Initialize velocity if not already
                    key = (id(layer), param_name)
                    if key not in self.velocities:
                        self.velocities[key] = np.zeros_like(layer_params[param_name])

                    # Update velocity
                    v = self.velocities[key]
                    grad = layer_grads[gradient_name]
                    v = self.momentum * v - self.learning_rate * grad
                    self.velocities[key] = v

                    # Update parameter
                    layer_params[param_name] += v

class Adam(Optimizer):
    """Adam optimizer.
    
    Parameters are updated as:
    m = β₁ * m + (1 - β₁) * g
    v = β₂ * v + (1 - β₂) * g²
    m̂ = m / (1 - β₁ᵗ)
    v̂ = v / (1 - β₂ᵗ)
    θ = θ - learning_rate * m̂ / (√v̂ + ε)
    """
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Parameters
        ----------
        parameters : list
            List of layer modules with get_parameters and get_gradients methods.
        learning_rate : float, default=0.001
            Initial step size.
        beta1 : float, default=0.9
            Decay rate for the first moment estimates.
        beta2 : float, default=0.999
            Decay rate for the second moment estimates.
        epsilon : float, default=1e-8
            Small constant for numerical stability.
        """
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # timestep

        # Store first and second moment estimates per parameter
        self.m = {}
        self.v = {}

    def step(self):
        """Update parameters using Adam algorithm.
        
        Computes adaptive learning rates for each parameter based on 
        first and second moment estimates of the gradients.
        """
        self.t += 1
        for layer in self.parameters:
            if not hasattr(layer, 'get_parameters'):
                continue
            layer_params = layer.get_parameters()
            layer_grads = layer.get_gradients()

            for param_name in layer_params:
                gradient_name = f"d{param_name}"
                if gradient_name in layer_grads:
                    p = layer_params[param_name]
                    g = layer_grads[gradient_name]

                    pid = id(p)
                    # Initialize moments if not yet
                    if pid not in self.m:
                        self.m[pid] = np.zeros_like(p)
                        self.v[pid] = np.zeros_like(p)

                    # Update moments
                    self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * g
                    self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (g * g)

                    # Bias correction
                    m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[pid] / (1 - self.beta2 ** self.t)

                    # Update parameter
                    layer_params[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
