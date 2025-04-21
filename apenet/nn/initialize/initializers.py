# apenet/initialize/initializers.py
import torch

class Initializer:
    """Base class for all weight initializers."""
    def __init__(self, device="cpu"):
        self.device = device

    def initialize(self, shape):
        """Initialize weights."""
        raise NotImplementedError

class HeInitializer(Initializer):
    """
    He initialization for weights.

    W ~ N(0, sqrt(2 / in_dim))
    """
    def initialize(self, shape):
        """
        Initialize weights using He initialization.

        Parameters:
        - shape: Tuple of (in_features, out_features).

        Returns:
        - W: Initialized weights.
        """
        in_dim, out_dim = shape
        std = torch.sqrt(torch.tensor(2.0, device=self.device) / in_dim)
        return torch.randn(in_dim, out_dim, device=self.device) * std

class ZerosInitializer(Initializer):
    """Initialize weights with zeros."""
    def initialize(self, shape):
        """
        Initialize weights with zeros.

        Parameters:
        - shape: Shape of the weight tensor.

        Returns:
        - W: Initialized weights.
        """
        return torch.zeros(shape, device=self.device)

class NormalInitializer(Initializer):
    """Initialize weights with normal distribution."""
    def __init__(self, mean=0.0, std=0.01, device="cpu"):
        super().__init__(device)
        self.mean = mean
        self.std = std

    def initialize(self, shape):
        """
        Initialize weights with normal distribution.

        Parameters:
        - shape: Shape of the weight tensor.

        Returns:
        - W: Initialized weights.
        """
        return torch.randn(shape, device=self.device) * self.std + self.mean

class XavierInitializer(Initializer):
    """
    Xavier initialization for weights.

    W ~ N(0, sqrt(2 / (in_dim + out_dim)))
    """
    def initialize(self, shape):
        """
        Initialize weights using Xavier initialization.

        Parameters:
        - shape: Tuple of (in_features, out_features).

        Returns:
        - W: Initialized weights.
        """
        in_dim, out_dim = shape
        std = torch.sqrt(torch.tensor(2.0, device=self.device) / (in_dim + out_dim))
        return torch.randn(in_dim, out_dim, device=self.device) * std
