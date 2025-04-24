# apenet/initialize/initializers.py
from functools import lru_cache
import inspect
import sys
import numpy as np

class Initializer:
    """Base class for all weight initializers."""
    def __init__(self, rng=None, seed=None):
        self.rng = rng if rng is not None else np.random.default_rng(seed)


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

@lru_cache(maxsize=1)
def get_initializer_registry():
    """
    Auto-discover all initializer classes and create a registry.
    Cache the result for performance.
    """
    registry = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and issubclass(obj, Initializer) and obj != Initializer:
            # Remove 'Initializer' suffix and convert to lowercase
            key = name.lower().replace('initializer', '')
            registry[key] = obj
    return registry

def get_initializer(name, **kwargs):
    """
    Get initializer by name.
    
    Args:
        name (str): Name of the initializer
        **kwargs: Additional arguments to pass to the initializer
        
    Returns:
        Initializer: Instance of the initializer
        
    Raises:
        ValueError: If initializer name is not recognized
    """
    if name is None:
        return None
        
    registry = get_initializer_registry()
    name = name.lower()
    initializer_class = registry.get(name)
    
    if initializer_class is None:
        available = list(registry.keys())
        raise ValueError(f"Unsupported initializer: '{name}'. Available options: {available}")
    
    return initializer_class(**kwargs)