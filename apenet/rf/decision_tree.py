# apenet/rf/decision_tree.py
import numpy as np
from collections import Counter
from typing import Optional, Tuple, List, Dict, Union, Any

from apenet.interfaces.base_model import BaseModel

class Node:
    """
    A node in the decision tree.

    Parameters
    ----------
    feature_idx : int, optional
        The index of the feature to split on.
    threshold : float, optional
        The threshold value to split the feature.
    left : Node, optional
        The left child node.
    right : Node, optional
        The right child node.
    value : Any, optional
        The predicted value if this is a leaf node.
    """
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None, *,
        value: Optional[Any] = None
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf node.

        Returns
        -------
        bool
            True if the node is a leaf node, False otherwise.
        """
        return self.value is not None


class DecisionTree(BaseModel):
    """
    Base decision tree implementation providing common functionality.
    
    This class should not be used directly. Use derived classes for
    specific tasks (classification or regression).

    Parameters
    ----------
    max_depth : int, optional
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure
        or contain less than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be a leaf node.
    max_features : int or float, optional
        The number of features to consider when looking for the best split:
        - If int, consider `max_features` features at each split.
        - If float between 0 and 1, consider `max_features * n_features` features at each split.
        - If None, consider all features.
    seed : int, optional
        Controls the randomness of the estimator. Passed to BaseModel.
    rng : np.random.Generator, optional
        Pre-initialized random number generator. If provided, seed is ignored.
    """
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None
    ):
        super().__init__(seed=seed, rng=rng)
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree_ = None
        self.n_features_ = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Abstract method to be implemented by derived classes.
        """
        raise NotImplementedError("DecisionTree is an abstract class. Use DecisionTreeClassifier or DecisionTreeRegressor.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted values.
        """
        self._check_is_fitted()
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample: np.ndarray) -> Any:
        """
        Predict the value for a single sample.

        Parameters
        ----------
        sample : np.ndarray of shape (n_features,)
            The input sample.

        Returns
        -------
        Any
            The predicted value.
        """
        node = self.tree_
        
        while not node.is_leaf():
            if node.feature_idx >= len(sample):  # Safety check
                break
                
            if sample[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
                
        return node.value

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively grow the tree. To be implemented by derived classes.
        """
        raise NotImplementedError("Method _grow_tree should be implemented by derived classes.")

    def _get_potential_thresholds(self, X_column: np.ndarray) -> List[float]:
        """
        Get potential threshold values for a feature.

        Parameters
        ----------
        X_column : np.ndarray of shape (n_samples,)
            The feature values.

        Returns
        -------
        List[float]
            List of potential threshold values.
        """
        # Get unique values and sort them
        sorted_values = np.sort(np.unique(X_column))
        
        # Calculate midpoints between adjacent values
        if len(sorted_values) <= 1:
            return []
            
        return [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(len(sorted_values) - 1)]

    def _split_data(self, X_column: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the data based on a feature and threshold.

        Parameters
        ----------
        X_column : np.ndarray of shape (n_samples,)
            The feature values.
        threshold : float
            The threshold value.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The indices of samples in the left and right nodes.
        """
        left_indices = np.where(X_column <= threshold)[0]
        right_indices = np.where(X_column > threshold)[0]
        return left_indices, right_indices

    def feature_importances(self) -> Dict[int, float]:
        """
        Calculate the feature importances.

        Returns
        -------
        Dict[int, float]
            A dictionary mapping feature indices to their normalized importance.
        """
        self._check_is_fitted()
        importances = self._calculate_feature_importances(self.tree_)
        
        # Normalize importances
        total = sum(importances.values())
        if total > 0:
            return {feat: importance / total for feat, importance in importances.items()}
        
        return importances

    def _calculate_feature_importances(self, node: Node, depth: int = 0) -> Dict[int, float]:
        """
        Recursively calculate feature importances in the tree.

        Parameters
        ----------
        node : Node
            The current node.
        depth : int, default=0
            The current depth of the tree.

        Returns
        -------
        Dict[int, float]
            A dictionary mapping feature indices to their importance.
        """
        importances = Counter()
        
        if not node.is_leaf():
            # The importance of a feature is weighted by the depth
            importance = 1.0 / (depth + 1.0)
            importances[node.feature_idx] += importance
            
            # Recursively calculate importance for children
            left_importances = self._calculate_feature_importances(node.left, depth + 1)
            right_importances = self._calculate_feature_importances(node.right, depth + 1)
            
            # Combine importances
            for feat, importance in left_importances.items():
                importances[feat] += importance
                
            for feat, importance in right_importances.items():
                importances[feat] += importance
                
        return importances
    
    def _check_is_fitted(self) -> None:
        """
        Check if the model is fitted.
        
        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        if not self._is_fitted:
            raise ValueError("This DecisionTree instance is not fitted yet. Call 'fit' before using this estimator.")


class DecisionTreeClassifier(DecisionTree):
    """
    A decision tree classifier.

    Parameters
    ----------
    max_depth : int, optional
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure
        or contain less than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be a leaf node.
    max_features : int or float, optional
        The number of features to consider when looking for the best split:
        - If int, consider `max_features` features at each split.
        - If float between 0 and 1, consider `max_features * n_features` features at each split.
        - If None, consider all features.
    seed : int, optional
        Controls the randomness of the estimator. Passed to BaseModel.
    rng : np.random.Generator, optional
        Pre-initialized random number generator. If provided, seed is ignored.
    
    Attributes
    ----------
    tree_ : Node
        The root node of the decision tree.
    n_features_ : int
        The number of features when `fit` is performed.
    n_classes_ : int
        The number of classes.
    """
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None
    ):
        super().__init__(
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            seed=seed,
            rng=rng
        )
        self.n_classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Build a decision tree classifier from the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : DecisionTreeClassifier
            The fitted decision tree classifier.
        """
        # Get dataset information
        n_samples, self.n_features_ = X.shape
        self.n_classes_ = len(np.unique(y))
        
        # Determine max_features
        if self.max_features is None:
            self.max_features_ = self.n_features_
        elif isinstance(self.max_features, float) and self.max_features <= 1.0:
            self.max_features_ = max(int(self.max_features * self.n_features_), 1)
        else:
            self.max_features_ = min(self.max_features, self.n_features_)
        
        # Build the tree
        self.tree_ = self._grow_tree(X, y)
        self._is_fitted = True
        
        return self

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively grow the classification tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values (class labels).
        depth : int, default=0
            The current depth of the tree.

        Returns
        -------
        Node
            The root node of the grown tree.
        """
        n_samples, _ = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            
            leaf_value = self._most_common_class(y)
            return Node(value=leaf_value)
        
        # Find best split
        feature_idx, threshold = self._find_best_split(X, y)
        
        # If no split improves the criterion, create a leaf node
        if feature_idx is None:
            leaf_value = self._most_common_class(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_indices, right_indices = self._split_data(X[:, feature_idx], threshold)
        
        # Check if either of the splits has fewer samples than min_samples_leaf
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            leaf_value = self._most_common_class(y)
            return Node(value=leaf_value)
        
        # Recursively grow left and right branches
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        # Create and return node
        return Node(feature_idx, threshold, left, right)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best feature and threshold to split the data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        Tuple[Optional[int], Optional[float]]
            The index of the feature to split on and the threshold value.
            Returns (None, None) if no split is found.
        """
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None
        
        # Calculate parent impurity
        parent_impurity = self._calculate_gini_impurity(y)
        
        # If the node is already pure, no need to split
        if parent_impurity == 0:
            return None, None
            
        # Track best split
        best_gain = 0.0
        best_feature, best_threshold = None, None
        
        # Randomly select features to consider
        feature_indices = self.rng.choice(
            n_features, 
            size=min(self.max_features_, n_features), 
            replace=False
        )
        
        # Find the best split among the selected features
        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            thresholds = self._get_potential_thresholds(X_column)
            
            for threshold in thresholds:
                # Split the data
                left_indices, right_indices = self._split_data(X_column, threshold)
                
                # Skip if the split doesn't produce two valid groups
                if (len(left_indices) < self.min_samples_leaf or 
                    len(right_indices) < self.min_samples_leaf):
                    continue
                
                # Calculate the impurity for each child
                left_impurity = self._calculate_gini_impurity(y[left_indices])
                right_impurity = self._calculate_gini_impurity(y[right_indices])
                
                # Calculate the weighted impurity
                weighted_impurity = (len(left_indices) / n_samples) * left_impurity + (len(right_indices) / n_samples) * right_impurity
                
                # Calculate information gain
                gain = parent_impurity - weighted_impurity
                
                # Update the best split if we found a better one
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def _calculate_gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate the Gini impurity of a node.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        float
            The Gini impurity.
        """
        if len(y) == 0:
            return 0.0
            
        # Count occurrences of each class
        class_counts = np.bincount(y, minlength=self.n_classes_)
        
        # Calculate class probabilities
        probabilities = class_counts / len(y)
        
        # Calculate Gini impurity
        return 1.0 - np.sum(probabilities ** 2)

    def _most_common_class(self, y: np.ndarray) -> int:
        """
        Find the most common class in a node.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        int
            The most common class.
        """
        if len(y) == 0:
            return 0
            
        # Use bincount for efficiency with integer labels
        counts = np.bincount(y, minlength=self.n_classes_)
        return np.argmax(counts)


class DecisionTreeRegressor(DecisionTree):
    """
    A decision tree regressor.

    Parameters
    ----------
    max_depth : int, optional
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure
        or contain less than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be a leaf node.
    max_features : int or float, optional
        The number of features to consider when looking for the best split:
        - If int, consider `max_features` features at each split.
        - If float between 0 and 1, consider `max_features * n_features` features at each split.
        - If None, consider all features.
    seed : int, optional
        Controls the randomness of the estimator. Passed to BaseModel.
    rng : np.random.Generator, optional
        Pre-initialized random number generator. If provided, seed is ignored.
    
    Attributes
    ----------
    tree_ : Node
        The root node of the decision tree.
    n_features_ : int
        The number of features when `fit` is performed.
    """
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None
    ):
        super().__init__(
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            seed=seed,
            rng=rng
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        """
        Build a decision tree regressor from the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values (real numbers).

        Returns
        -------
        self : DecisionTreeRegressor
            The fitted decision tree regressor.
        """
        # Get dataset information
        n_samples, self.n_features_ = X.shape
        
        # Determine max_features
        if self.max_features is None:
            self.max_features_ = self.n_features_
        elif isinstance(self.max_features, float) and self.max_features <= 1.0:
            self.max_features_ = max(int(self.max_features * self.n_features_), 1)
        else:
            self.max_features_ = min(self.max_features, self.n_features_)
        
        # Build the tree
        self.tree_ = self._grow_tree(X, y)
        self._is_fitted = True
        
        return self

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively grow the regression tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values (real numbers).
        depth : int, default=0
            The current depth of the tree.

        Returns
        -------
        Node
            The root node of the grown tree.
        """
        n_samples, _ = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or 
            n_samples < self.min_samples_split):
            
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Find best split
        feature_idx, threshold = self._find_best_split(X, y)
        
        # If no split improves the criterion, create a leaf node
        if feature_idx is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_indices, right_indices = self._split_data(X[:, feature_idx], threshold)
        
        # Check if either of the splits has fewer samples than min_samples_leaf
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Recursively grow left and right branches
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        # Create and return node
        return Node(feature_idx, threshold, left, right)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best feature and threshold to split the data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values (real numbers).

        Returns
        -------
        Tuple[Optional[int], Optional[float]]
            The index of the feature to split on and the threshold value.
            Returns (None, None) if no split is found.
        """
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None
        
        # Calculate total MSE (before split)
        total_mse = self._calculate_mse(y)
        
        # Track best split
        best_mse_reduction = 0.0
        best_feature, best_threshold = None, None
        
        # Randomly select features to consider
        feature_indices = self.rng.choice(
            n_features, 
            size=min(self.max_features_, n_features), 
            replace=False
        )
        
        # Find the best split among the selected features
        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            thresholds = self._get_potential_thresholds(X_column)
            
            for threshold in thresholds:
                # Split the data
                left_indices, right_indices = self._split_data(X_column, threshold)
                
                # Skip if the split doesn't produce two valid groups
                if (len(left_indices) < self.min_samples_leaf or 
                    len(right_indices) < self.min_samples_leaf):
                    continue
                
                # Calculate MSE for each child
                left_mse = self._calculate_mse(y[left_indices])
                right_mse = self._calculate_mse(y[right_indices])
                
                # Calculate the weighted MSE
                weighted_mse = (len(left_indices) / n_samples) * left_mse + (len(right_indices) / n_samples) * right_mse
                
                # Calculate MSE reduction
                mse_reduction = total_mse - weighted_mse
                
                # Update the best split if we found a better one
                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def _calculate_mse(self, y: np.ndarray) -> float:
        """
        Calculate the mean squared error of a node.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            The target values (real numbers).

        Returns
        -------
        float
            The mean squared error.
        """
        if len(y) == 0:
            return 0.0
            
        # Calculate mean squared error
        return np.mean((y - np.mean(y)) ** 2)