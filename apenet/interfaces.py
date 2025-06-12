# apenet/interfaces.py
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all models in apenet."""

    def __init__(self, seed=None, rng=None):
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        super().__init__()


    @abstractmethod
    def fit(self, X, y, *args, **kwargs): ...
    @abstractmethod
    def predict(self, X, *args, **kwargs): ...