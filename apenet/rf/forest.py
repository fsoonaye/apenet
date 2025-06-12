# apenet/rf/forest.py
import numpy as np
from collections import Counter

from apenet.interfaces import BaseModel
from apenet.utils.data import bootstrap_sample
from apenet.rf.tree import DecisionTreeClassifier, DecisionTreeRegressor

class RandomForest(BaseModel):
    """Base random forest implementation.
    
    This class implements the common functionality for random forests.
    Use the specialized RandomForestClassifier or RandomForestRegressor classes.

    Parameters
    ----------
    n_trees : int, default=100
        The number of trees in the forest.
    max_depth : int, default=None
        The maximum depth of the trees.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be a leaf node.
    max_features : int or float, default=None
        The number of features to consider when looking for the best split.
    seed : int, default=None
        Seed for the random number generator.
    rng : numpy.random.Generator, default=None
        Numpy random generator.
    """
    def __init__(self,
                 n_trees=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 seed=None,
                 rng=None):
        super().__init__(seed=seed, rng=rng)
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []
        
    def _get_tree_class(self):
        """Get the tree class to use. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_tree_class")
        
    def fit(self, X, y):
        """Fit the random forest to the data.

        Parameters
        ----------
        X : ndarray
            The input data of shape (n_samples, n_features).
        y : ndarray
            The target values of shape (n_samples,).

        Returns
        -------
        self
            The fitted estimator.
        """
        self.trees = []
        tree_class = self._get_tree_class()
        
        for _ in range(self.n_trees):
            tree = tree_class(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                rng=self.rng
            )

            X_samp, y_samp = bootstrap_sample(X, y, rng=self.rng)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)
            
        return self
        
    def predict(self, X):
        """Predict values for X. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict")
        
    def feature_importances(self):
        """Get feature importances from the random forest.

        Returns
        -------
        dict
            Dictionary mapping feature indices to their importance scores.
        """
        # Combine feature importances from all trees
        importances = {}
        for tree in self.trees:
            tree_importances = tree.feature_importances()
            for feature, imp in tree_importances.items():
                importances[feature] = importances.get(feature, 0) + imp / self.n_trees
                
        return importances


class RandomForestClassifier(RandomForest):
    """Random forest classifier implementation.
    
    A random forest is an ensemble of decision trees, where each tree is trained 
    on a bootstrap sample of the data with a random subset of features.
    
    Parameters
    ----------
    n_trees : int, default=100
        The number of trees in the forest.
    max_depth : int, default=None
        The maximum depth of the trees.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be a leaf node.
    max_features : int or float, default=None
        The number of features to consider when looking for the best split:
        - If int, consider `max_features` features at each split.
        - If float between 0 and 1, consider `max_features * n_features` features at each split.
        - If None, consider sqrt(n_features) features at each split.
    seed : int, default=None
        Seed for the random number generator.
    rng : numpy.random.Generator, default=None
        Numpy random generator.
    """
    def _get_tree_class(self):
        """Get the tree class to use (DecisionTreeClassifier)."""
        return DecisionTreeClassifier
    
    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : ndarray
            The input data of shape (n_samples, n_features).

        Returns
        -------
        ndarray
            The predicted class labels of shape (n_samples,).
        """
        # Get all tree predictions
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Do a majority vote across all trees
        predictions = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], 0, tree_preds)
        return predictions
        

class RandomForestRegressor(RandomForest):
    """Random forest regressor implementation.
    
    A random forest is an ensemble of decision trees, where each tree is trained 
    on a bootstrap sample of the data with a random subset of features.
    
    Parameters
    ----------
    n_trees : int, default=100
        The number of trees in the forest.
    max_depth : int, default=None
        The maximum depth of the trees.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be a leaf node.
    max_features : int or float, default=None
        The number of features to consider when looking for the best split:
        - If int, consider `max_features` features at each split.
        - If float between 0 and 1, consider `max_features * n_features` features at each split.
        - If None, consider sqrt(n_features) features at each split.
    seed : int, default=None
        Seed for the random number generator.
    rng : numpy.random.Generator, default=None
        Numpy random generator.
    """
    def _get_tree_class(self):
        return DecisionTreeRegressor
    
    def predict(self, X):
        """Predict regression values for X.

        Parameters
        ----------
        X : ndarray
            The input data of shape (n_samples, n_features).

        Returns
        -------
        ndarray
            The predicted values of shape (n_samples,).
        """
        # Get all tree predictions
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Average the predictions from all trees
        return np.mean(tree_preds, axis=0)
