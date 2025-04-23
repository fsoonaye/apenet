# apenet/optimizers/optimizers.py
import numpy as np

class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, parameters):
        """
        Initialize the optimizer.

        Parameters:
        - parameters: List of parameter dictionaries from each layer.
        """
        self.parameters = parameters

    def step(self):
        """Update parameters using gradients."""
        raise NotImplementedError

    def zero_grad(self):
        """Zero out gradients."""
        for layer in self.parameters:
            if hasattr(layer, 'gradients'):
                for grad in layer.gradients.values():
                    grad.fill(0)

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Parameters are updated as:
    θ = θ - learning_rate * gradient
    """
    def __init__(self, parameters, learning_rate=0.01, momentum=0.9):
        """
        Initialize the SGD optimizer.

        Parameters:
        - parameters: List of parameter dictionaries from each layer.
        - learning_rate: Learning rate for gradient descent.
        - momentum: Momentum factor.
        """
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def step(self):
        """
        Update parameters using gradients.

        For each layer, update its parameters using its gradients.
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
    """
    Adam optimizer.

    Parameters are updated as per the Adam update rules, maintaining running averages
    of both the gradients and their squares (first and second moments).
    """
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Adam optimizer.

        Parameters:
        - parameters: List of layer modules (with .get_parameters and .get_gradients)
        - learning_rate: Initial step size (default 0.001)
        - beta1: Decay rate for the first moment estimates (default 0.9)
        - beta2: Decay rate for the second moment estimates (default 0.999)
        - epsilon: Small constant for numerical stability (default 1e-8)
        """
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # timestep

        # Store first and second moment estimates per parameter
        # {id(param): tensor same shape as param}
        self.m = {}
        self.v = {}

    def step(self):
        """Update parameters using Adam algorithm."""
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
