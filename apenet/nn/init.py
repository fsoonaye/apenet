# apenet/nn/init.py
import numpy as np
import inspect
import sys
from functools import lru_cache

class Initializer:
    """Base class for all weight initializers."""
    
    def __init__(self, rng=None, seed=None):
        self.rng = rng if rng is not None else np.random.default_rng(seed)
    
    def init(self, shape):
        """Initialize weights."""
        raise NotImplementedError


class HeInitializer(Initializer):
    """He initialization for weights.
    W ~ N(0, sqrt(2 / in_dim))
    """
    
    def __init__(self, scale=0.1, rng=None, seed=None):
        super().__init__(rng=rng, seed=seed)
        self.scale = scale
    
    def init(self, shape):
        """Initialize weights using scaled He initialization.
        
        Parameters
        ----------
        shape : tuple
            Tuple of (in_features, out_features).
            
        Returns
        -------
        ndarray
            Initialized weights.
        """
        in_dim, out_dim = shape
        std = np.sqrt(2.0 / in_dim) * self.scale
        return self.rng.normal(0, std, (in_dim, out_dim))


class ZerosInitializer(Initializer):
    """Initialize weights with zeros."""
    
    def init(self, shape):
        """Initialize weights with zeros.
        
        Parameters
        ----------
        shape : tuple
            Shape of the weight array.
            
        Returns
        -------
        ndarray
            Initialized weights.
        """
        return np.zeros(shape)


class NormalInitializer(Initializer):
    """Initialize weights with normal distribution."""
    
    def __init__(self, mean=0.0, std=0.01, rng=None, seed=None):
        super().__init__(rng=rng, seed=seed)
        self.mean = mean
        self.std = std
    
    def init(self, shape):
        """Initialize weights with normal distribution.
        
        Parameters
        ----------
        shape : tuple
            Shape of the weight array.
            
        Returns
        -------
        ndarray
            Initialized weights.
        """
        return self.rng.normal(self.mean, self.std, shape)


class XavierInitializer(Initializer):
    """Xavier initialization for weights.
    W ~ N(0, sqrt(2 / (in_dim + out_dim)))
    """
    
    def init(self, shape):
        """Initialize weights using Xavier initialization.
        
        Parameters
        ----------
        shape : tuple
            Tuple of (in_features, out_features).
            
        Returns
        -------
        ndarray
            Initialized weights.
        """
        in_dim, out_dim = shape
        std = np.sqrt(2.0 / (in_dim + out_dim))
        return self.rng.normal(0, std, (in_dim, out_dim))


@lru_cache(maxsize=1)
def get_initializer_registry():
    """Auto-discover all initializer classes and create a registry.
    Cache the result for performance.
    
    Returns
    -------
    dict
        Dictionary mapping initializer names to classes.
    """
    registry = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and issubclass(obj, Initializer) and obj != Initializer:
            # Remove 'Initializer' suffix and convert to lowercase
            key = name.lower().replace('initializer', '')
            registry[key] = obj
    return registry


def get_initializer(name, **kwargs):
    """Get initializer by name.
    
    Parameters
    ----------
    name : str
        Name of the initializer
    **kwargs
        Additional arguments to pass to the initializer
        
    Returns
    -------
    Initializer
        Instance of the initializer
        
    Raises
    ------
    ValueError
        If initializer name is not recognized
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