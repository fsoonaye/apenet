# apenet/initialize/initializers.py
import numpy as np

class Initializer:
    """Base class for all weight initializers."""
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

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
        std = np.sqrt(2.0 / in_dim)
        return np.random.randn(in_dim, out_dim) * std

class ZerosInitializer(Initializer):
    """Initialize weights with zeros."""
    def initialize(self, shape):
        """
        Initialize weights with zeros.

        Parameters:
        - shape: Shape of the weight array.

        Returns:
        - W: Initialized weights.
        """
        return np.zeros(shape)

class NormalInitializer(Initializer):
    """Initialize weights with normal distribution."""
    def __init__(self, mean=0.0, std=0.01, seed=None):
        super().__init__(seed)
        self.mean = mean
        self.std = std

    def initialize(self, shape):
        """
        Initialize weights with normal distribution.

        Parameters:
        - shape: Shape of the weight array.

        Returns:
        - W: Initialized weights.
        """
        return np.random.randn(*shape) * self.std + self.mean

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
        std = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.randn(in_dim, out_dim) * std
