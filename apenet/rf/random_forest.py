# apenet/rf/rf.py
import numpy as np
from collections import Counter
from apenet.interfaces.base_model import BaseModel
from apenet.utils.data import bootstrap_sample
from apenet.rf.decision_tree import DecisionTree

from typing import Optional

class RandomForest(BaseModel):
    """
    A random forest classifier.

    Parameters
    ----------
    n_trees : int, default=100
        The number of trees in the forest.
    max_depth : int, default=None
        The maximum depth of the trees.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    max_features : int, default=None
        The number of features to consider when looking for the best split.
    seed : int, default=None
        Seed for the random number generator.
    rng : np.random.Generator, default=None
        Numpy random generator.
    """
    def __init__(self,
                 n_trees: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 max_features: Optional[int] = None,
                 seed: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None):
        super().__init__(seed=seed, rng=rng)
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            rng: Optional[np.random.Generator] = None,
            seed: Optional[int] = None) -> None:
        """
        Fit the random forest to the data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target labels.
        rng : np.random.Generator, default=None
            Numpy random generator.
        seed : int, default=None
            Seed for the random number generator.
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )

            X_samp, y_samp = bootstrap_sample(X, y, rng=rng, seed=seed)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        np.ndarray
            The predicted labels.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], 0, tree_preds)
        return tree_preds
