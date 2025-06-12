# apenet/rf/decision_tree.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from typing import Optional, Tuple, List, Dict, Union, Any
import inspect

from apenet.interfaces import BaseModel

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
    impurity_reduction : float, optional
        The impurity reduction achieved by this split.
    n_samples : int, optional
        The number of samples in this node.
    """
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None, *,
        value: Optional[Any] = None,
        impurity_reduction: float = 0.0,
        n_samples: int = 0
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.impurity_reduction = impurity_reduction
        self.n_samples = n_samples

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
    nodesize : int, default=2
        The minimum number of samples required to be a valid node.
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
        nodesize: int = 2,
        max_features: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None
    ):
        super().__init__(seed=seed, rng=rng)
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.nodesize = nodesize
        self.max_features = max_features
        self.tree_ = None
        self.n_features_ = None
        self.n_samples_ = None
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

    def _traverse_tree(
        self, 
        node: Node, 
        operation,
        condition = None,
        depth: int = 0,
        X = None,
        sample_indices = None
    ):
        """
        Enhanced tree traversal that optionally supports sample tracking.
        
        Parameters
        ----------
        node : Node
            Current node to process
        operation : callable
            Function to apply to each node
        condition : callable, optional
            Optional condition to filter nodes
        depth : int, default=0
            Current depth in the tree
        X : np.ndarray, optional
            Input data for sample-aware operations
        sample_indices : np.ndarray, optional
            Indices of samples reaching this node
            
        Returns
        -------
        Any
            Result of the traversal operation
        """
        if node is None:
            return 0
        
        if X is not None and (sample_indices is None or len(sample_indices) == 0):
            return 0
            
        # Apply condition if provided
        if condition is None or condition(node):
            # Smart operation calling based on parameter count
            sig = inspect.signature(operation)
            param_count = len(sig.parameters)
            
            if X is not None and sample_indices is not None and param_count >= 3:
                result = operation(node, sample_indices, depth)  # Sample-aware
            elif param_count >= 2:
                result = operation(node, depth)  # Depth-aware
            else:
                result = operation(node)  # Simple
        else:
            result = 0
            
        # Process children
        if not node.is_leaf():
            left_result = right_result = 0
            
            if X is not None and sample_indices is not None:
                # Split samples for children
                feature_values = X[sample_indices, node.feature_idx]
                left_mask = feature_values <= node.threshold
                right_mask = ~left_mask
                left_indices = sample_indices[left_mask]
                right_indices = sample_indices[right_mask]
                
                if node.left and len(left_indices) > 0:
                    left_result = self._traverse_tree(
                        node.left, operation, condition, depth + 1, X, left_indices
                    )
                if node.right and len(right_indices) > 0:
                    right_result = self._traverse_tree(
                        node.right, operation, condition, depth + 1, X, right_indices
                    )
            else:
                # Standard traversal
                if node.left:
                    left_result = self._traverse_tree(
                        node.left, operation, condition, depth + 1, X, sample_indices
                    )
                if node.right:
                    right_result = self._traverse_tree(
                        node.right, operation, condition, depth + 1, X, sample_indices
                    )
            
            # Combine results
            if isinstance(result, (int, float)):
                return result + left_result + right_result
            elif isinstance(result, dict):
                for key, value in left_result.items():
                    result[key] = result.get(key, 0) + value
                for key, value in right_result.items():
                    result[key] = result.get(key, 0) + value
                return result
            else:
                return max(result, left_result, right_result)
        
        return result

    def get_tree_stats(self, n_strong: Optional[int] = None) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive tree statistics in a single traversal.
        
        Parameters
        ----------
        n_strong : int, optional
            Number of strong (signal) features. If provided, will calculate effective metrics.
        
        Returns
        -------
        Dict[str, Union[int, float]]
            Dictionary containing various tree statistics
        """
        self._check_is_fitted()
        
        if self.tree_ is None:
            base_stats = {
                'n_nodes': 0,
                'n_split_nodes': 0,
                'n_leaf_nodes': 0,
                'max_depth': -1,
                'log_depth': 0.0,
                'n_cuts': 0,
            }
            if n_strong is not None:
                base_stats.update({
                    'n_strong': n_strong,
                    'n_noise': max(0, self.n_features_ - n_strong) if self.n_features_ else 0,
                    'n_cuts_effective': 0,
                    'log_depth_effective': 0.0
                })
            return base_stats
        
        stats = {
            'n_nodes': 0, 
            'n_split_nodes': 0, 
            'n_leaf_nodes': 0, 
            'max_depth': 0,
            'n_cuts_effective': 0
        }
        
        def collect_stats(node: Node, depth: int) -> int:
            stats['n_nodes'] += 1
            stats['max_depth'] = max(stats['max_depth'], depth)
            
            if node.is_leaf():
                stats['n_leaf_nodes'] += 1
            else:
                stats['n_split_nodes'] += 1
                # Count effective cuts if n_strong is provided
                if n_strong is not None and node.feature_idx < n_strong:
                    stats['n_cuts_effective'] += 1
            
            return 1
        
        self._traverse_tree(self.tree_, collect_stats)
        
        # Calculate derived metrics
        stats['n_cuts'] = stats['n_split_nodes']
        stats['log_depth'] = np.log2(stats['n_split_nodes'] + 1)
        
        # Add strong/noise feature counts and effective log depth if n_strong provided
        if n_strong is not None:
            stats['n_strong'] = n_strong
            stats['n_noise'] = self.n_features_ - n_strong
            stats['log_depth_effective'] = np.log2(stats['n_cuts_effective'] + 1)
        
        return stats

    def get_effective_cuts(self, signal_feature_indices: List[int]) -> int:
        """
        Count the number of cuts/splits on signal features only.
        
        Parameters
        ----------
        signal_feature_indices : List[int]
            List of indices corresponding to signal features (non-noise features)
            
        Returns
        -------
        int
            The number of splits on signal features.
        """
        self._check_is_fitted()
        
        if self.tree_ is None:
            return 0
        
        signal_features = set(signal_feature_indices)
        
        def count_signal_splits(node: Node) -> int:
            return 1 if node.feature_idx in signal_features else 0
        
        def is_split_node(node: Node) -> bool:
            return not node.is_leaf()
        
        return self._traverse_tree(
            self.tree_, 
            count_signal_splits, 
            condition=is_split_node
        )

    def get_max_depth(self) -> int:
        """Get the maximum depth of the tree."""
        return self.get_tree_stats()['max_depth']

    def get_n_nodes(self) -> int:
        """Get the total number of nodes in the tree."""
        return self.get_tree_stats()['n_nodes']

    def get_n_split_nodes(self) -> int:
        """Get the number of split (internal) nodes in the tree."""
        return self.get_tree_stats()['n_split_nodes']

    def count_split_nodes_by_depth(self) -> Dict[int, int]:
        """
        Count the number of split (non-leaf) nodes at each depth.
        
        Returns
        -------
        Dict[int, int]
            Dictionary mapping depth to number of split nodes at that depth
        """
        self._check_is_fitted()
        
        if self.tree_ is None:
            return {}
        
        # Initialize dictionary to store counts
        depth_counts = defaultdict(int)
        
        def count_at_depth(node: Node, depth: int) -> int:
            if not node.is_leaf():
                depth_counts[depth] += 1
            return 0
        
        self._traverse_tree(self.tree_, count_at_depth)
        return dict(depth_counts)

    def get_log_depth(self) -> float:
        """Calculate the depth as log_2(N+1) where N is the number of split nodes."""
        return self.get_tree_stats()['log_depth']

    def get_effective_log_depth(self, signal_feature_indices: List[int]) -> float:
        """
        Calculate the effective depth as log_2(N*+1) where N* is the number of splits
        on signal features only.
        
        Parameters
        ----------
        signal_feature_indices : List[int]
            List of indices corresponding to signal features (non-noise features)
            
        Returns
        -------
        float
            The effective log-based depth of the tree.
        """
        n_effective_cuts = self.get_effective_cuts(signal_feature_indices)
        return np.log2(n_effective_cuts + 1)
    
    def get_weighted_variance_by_depth(self, X: np.ndarray, feature_idx: int = 0, 
                                    alpha: float = 1.0) -> dict:
        """Compute sum of p_{n,A} * Var(alpha * X_feature) by depth."""
        self._check_is_fitted()
        
        if self.tree_ is None:
            return {}
        
        variances_by_depth = defaultdict(float)
        total_samples = len(X)
        
        def variance_operation(node, sample_indices: np.ndarray, depth: int) -> float:
            if len(sample_indices) > 1:
                # Calculate fraction of samples in this node
                p_n_A = len(sample_indices) / total_samples
                
                # Calculate variance of the feature in this node
                feature_values = X[sample_indices, feature_idx]
                feature_variance = np.var(feature_values, ddof=1)
                
                # Scale by alpha^2 and weight by fraction of samples
                scaled_weighted_variance = (alpha ** 2) * feature_variance * p_n_A
            else:
                scaled_weighted_variance = 0.0
            
            variances_by_depth[depth] += scaled_weighted_variance
            return scaled_weighted_variance

        all_indices = np.arange(len(X))
        self._traverse_tree(self.tree_, variance_operation, X=X, sample_indices=all_indices)
    
        return dict(variances_by_depth)

    def feature_importances(self) -> Dict[int, float]:
        """
        Calculate the feature importances using simple depth-based weighting.

        Returns
        -------
        Dict[int, float]
            A dictionary mapping feature indices to their normalized importance.
        """
        self._check_is_fitted()
        
        if self.tree_ is None:
            return {}
        
        def calculate_importance(node: Node, depth: int) -> Dict[int, float]:
            if node.is_leaf():
                return {}
            
            importance = 1.0 / (depth + 1.0)
            return {node.feature_idx: importance}
        
        importances = self._traverse_tree(self.tree_, calculate_importance)
        
        # Normalize importances
        total = sum(importances.values()) if importances else 0
        if total > 0:
            return {feat: importance / total for feat, importance in importances.items()}
        
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
    nodesize : int, default=2
        The minimum number of samples required to be a valid node.
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
        nodesize: int = 2,
        max_features: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None
    ):
        super().__init__(
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            nodesize=nodesize,
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
        self.n_samples_, self.n_features_ = X.shape
        self.n_classes_ = len(np.unique(y))
        
        # Determine max_features
        self.max_features_ = self._calculate_max_features()
        
        # Build the tree
        self.tree_ = self._grow_tree(X, y)
        self._is_fitted = True
        
        return self

    def _calculate_max_features(self) -> int:
        """Calculate the actual number of features to consider at each split."""
        if self.max_features is None:
            return self.n_features_
        elif isinstance(self.max_features, float) and 0 < self.max_features <= 1.0:
            return max(1, int(self.max_features * self.n_features_))
        elif isinstance(self.max_features, int):
            return max(1, min(self.max_features, self.n_features_))
        else:
            return self.n_features_

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
        if self._should_stop_splitting(n_samples, depth, y):
            leaf_value = self._most_common_class(y)
            return Node(value=leaf_value, n_samples=n_samples)
        
        # Find best split
        best_split = self._find_best_split(X, y)
        
        # If no split improves the criterion, create a leaf node
        if best_split is None:
            leaf_value = self._most_common_class(y)
            return Node(value=leaf_value, n_samples=n_samples)
        
        feature_idx, threshold, gini_reduction = best_split
        left_indices, right_indices = self._split_data(X[:, feature_idx], threshold)
        
        # Check if either of the splits has fewer samples than min_samples_leaf
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            leaf_value = self._most_common_class(y)
            return Node(value=leaf_value, n_samples=n_samples)
        
        # Recursively grow left and right branches
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        # Create and return node
        return Node(
            feature_idx=feature_idx, 
            threshold=threshold, 
            left=left, 
            right=right,
            impurity_reduction=gini_reduction,
            n_samples=n_samples
        )

    def _should_stop_splitting(self, n_samples: int, depth: int, y: np.ndarray) -> bool:
        """Check if we should stop splitting based on stopping criteria."""
        return (
            (self.max_depth is not None and depth >= self.max_depth) or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1 or
            n_samples < self.nodesize
        )

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[Tuple[int, float, float]]:
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
        Tuple[Optional[int], Optional[float], Optional[float]]
            The index of the feature to split on, the threshold value, and the gini reduction.
            Returns None if no split is found.
        """
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None
        
        # Calculate parent impurity
        parent_impurity = self._calculate_gini_impurity(y)
        
        # If the node is already pure, no need to split
        if parent_impurity == 0:
            return None
            
        # Track best split
        best_gain = 0.0
        best_split = None
        
        # Randomly select features to consider
        feature_indices = self.rng.choice(
            n_features, 
            size=min(self.max_features_, n_features), 
            replace=False
        )
        
        # Find the best split among the selected features
        for feature_idx in feature_indices:
            split_result = self._evaluate_feature_splits(
                X[:, feature_idx], y, feature_idx, parent_impurity, n_samples
            )
            
            if split_result and split_result[2] > best_gain:
                best_gain = split_result[2]
                best_split = split_result
        
        return best_split

    def _evaluate_feature_splits(
        self, 
        X_column: np.ndarray, 
        y: np.ndarray, 
        feature_idx: int, 
        parent_impurity: float, 
        n_samples: int
    ) -> Optional[Tuple[int, float, float]]:
        """Evaluate all potential splits for a given feature."""
        thresholds = self._get_potential_thresholds(X_column)
        
        if not thresholds:
            return None

        best_split = None
        best_gain = 0.0
        
        for threshold in thresholds:
            left_indices, right_indices = self._split_data(X_column, threshold)
            
            # Skip if the split doesn't produce two valid groups
            if (len(left_indices) < self.min_samples_leaf or 
                len(right_indices) < self.min_samples_leaf):
                continue
            
            # Calculate the impurity for each child
            left_impurity = self._calculate_gini_impurity(y[left_indices])
            right_impurity = self._calculate_gini_impurity(y[right_indices])
            
            # Calculate the weighted impurity
            weighted_impurity = (len(left_indices) / n_samples) * left_impurity + \
                               (len(right_indices) / n_samples) * right_impurity
            
            # Calculate information gain
            gain = parent_impurity - weighted_impurity
            
            # Update the best split if we found a better one
            if gain > best_gain:
                best_gain = gain
                best_split = (feature_idx, threshold, gain)
        
        return best_split

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

    def calculate_mdi(self) -> Dict[int, float]:
        """
        Calculate feature importances using Mean Decrease in Impurity (MDI).
        
        Returns
        -------
        Dict[int, float]
            A dictionary mapping feature indices to their MDI importance.
        """
        self._check_is_fitted()
        
        if self.tree_ is None:
            return {}
        
        # Initialize importance dictionary
        importances = {i: 0.0 for i in range(self.n_features_)}
        
        def calculate_mdi_contribution(node: Node) -> Dict[int, float]:
            if node.is_leaf():
                return {}
            
            # Calculate p_{n,A}: fraction of observations falling into this node
            p_n_A = node.n_samples / self.n_samples_
            
            # L_{n,A}: impurity reduction for this split
            L_n_A = node.impurity_reduction
            
            # Add weighted importance: p_{n,A} * L_{n,A}
            return {node.feature_idx: p_n_A * L_n_A}
        
        mdi_contributions = self._traverse_tree(self.tree_, calculate_mdi_contribution)
        
        # Update importances with contributions
        for feature_idx, contribution in mdi_contributions.items():
            importances[feature_idx] += contribution
        
        return importances


class DecisionTreeRegressor(DecisionTree):
    """
    A decision tree regressor.

    Parameters
    ----------
    max_depth : int, optional
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be a leaf node.
    nodesize : int, default=5
        The minimum number of samples required to be a valid node.
    max_features : int or float, optional
        The number of features to consider when looking for the best split.
    seed : int, optional
        Controls the randomness of the estimator.
    rng : np.random.Generator, optional
        Pre-initialized random number generator.
    
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
        nodesize: int = 5,
        max_features: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None
    ):
        super().__init__(
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            nodesize=nodesize,
            max_features=max_features,
            seed=seed,
            rng=rng
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        """Build a decision tree regressor from the training data."""
        # Get dataset information
        self.n_samples_, self.n_features_ = X.shape
        
        # Set max_features_
        self.max_features_ = self._calculate_max_features()
        
        # Build the tree
        self.tree_ = self._grow_tree(X, y)
        self._is_fitted = True
        
        return self

    def _calculate_max_features(self) -> int:
        """Calculate the actual number of features to consider at each split."""
        if self.max_features is None:
            return self.n_features_
        elif isinstance(self.max_features, float) and 0 < self.max_features <= 1.0:
            return max(1, int(self.max_features * self.n_features_))
        elif isinstance(self.max_features, int):
            return max(1, min(self.max_features, self.n_features_))
        else:
            return self.n_features_

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively grow the regression tree."""
        n_samples = len(X)
        
        # Check stopping criteria
        if self._should_stop_splitting(n_samples, depth):
            return Node(value=np.mean(y), n_samples=n_samples)
        
        # Find best split
        best_split = self._find_best_split(X, y)
        
        if best_split is None:  # No valid split found
            return Node(value=np.mean(y), n_samples=n_samples)
        
        feature_idx, threshold, mse_reduction = best_split
        left_indices, right_indices = self._split_data(X[:, feature_idx], threshold)
        
        # Check if split produces valid leaf sizes
        if (len(left_indices) < self.min_samples_leaf or 
            len(right_indices) < self.min_samples_leaf):
            return Node(value=np.mean(y), n_samples=n_samples)
        
        # Create child nodes
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(
            feature_idx=feature_idx, 
            threshold=threshold, 
            left=left_child, 
            right=right_child,
            impurity_reduction=mse_reduction,
            n_samples=n_samples
        )

    def _should_stop_splitting(self, n_samples: int, depth: int) -> bool:
        """Check if we should stop splitting based on stopping criteria."""
        return (
            (self.max_depth is not None and depth >= self.max_depth) or 
            n_samples < self.min_samples_split or
            n_samples < self.nodesize
        )

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[Tuple[int, float, float]]:
        """Find the best feature and threshold to split the data."""
        n_samples, n_features = X.shape

        if n_samples <= 1:
            return None
        
        # Calculate total MSE (before split)
        total_mse = self._calculate_mse(y)
        
        # Track best split
        best_mse_reduction = 0.0
        best_split = None
        
        # Randomly select features to consider
        feature_indices = self.rng.choice(
            n_features, 
            size=min(self.max_features_, n_features),
            replace=False
        )
        
        # Find the best split among the selected features
        for feature_idx in feature_indices:
            split_result = self._evaluate_feature_splits(X[:, feature_idx], y, feature_idx, total_mse, n_samples)
            
            if split_result and split_result[2] > best_mse_reduction:
                best_mse_reduction = split_result[2]
                best_split = split_result
        
        return best_split

    def _evaluate_feature_splits(
        self, 
        X_column: np.ndarray, 
        y: np.ndarray, 
        feature_idx: int, 
        total_mse: float, 
        n_samples: int
    ) -> Optional[Tuple[int, float, float]]:
        """Evaluate all potential splits for a given feature."""
        thresholds = self._get_potential_thresholds(X_column)
        
        if not thresholds:
            return None

        best_split = None
        best_reduction = 0.0
        
        for threshold in thresholds:
            left_indices, right_indices = self._split_data(X_column, threshold)
            
            # Skip if split doesn't produce valid groups
            if (len(left_indices) < self.min_samples_leaf or 
                len(right_indices) < self.min_samples_leaf):
                continue
            
            # Calculate MSE reduction
            mse_reduction = self._calculate_mse_reduction(y, left_indices, right_indices, total_mse, n_samples)
            
            if mse_reduction > best_reduction:
                best_reduction = mse_reduction
                best_split = (feature_idx, threshold, mse_reduction)
        
        return best_split

    def _calculate_mse_reduction(
        self, 
        y: np.ndarray, 
        left_indices: np.ndarray, 
        right_indices: np.ndarray, 
        total_mse: float, 
        n_samples: int
    ) -> float:
        """Calculate the MSE reduction for a split."""
        left_mse = self._calculate_mse(y[left_indices])
        right_mse = self._calculate_mse(y[right_indices])
        
        # Calculate weighted MSE
        left_weight = len(left_indices) / n_samples
        right_weight = len(right_indices) / n_samples
        weighted_mse = left_weight * left_mse + right_weight * right_mse
        
        return total_mse - weighted_mse

    def _calculate_mse(self, y: np.ndarray) -> float:
        """Calculate the mean squared error of a node."""
        if len(y) == 0:
            return 0.0
        
        return np.mean((y - np.mean(y)) ** 2)

    def calculate_mdi(self) -> Dict[int, float]:
        """
        Calculate feature importances using Mean Decrease in Impurity (MDI).
        
        Returns
        -------
        Dict[int, float]
            A dictionary mapping feature indices to their MDI importance.
        """
        self._check_is_fitted()
        
        if self.tree_ is None:
            return {}
        
        # Initialize importance dictionary
        importances = {i: 0.0 for i in range(self.n_features_)}
        
        def calculate_mdi_contribution(node: Node) -> Dict[int, float]:
            if node.is_leaf():
                return {}
            
            # Calculate p_{n,A}: fraction of observations falling into this node
            p_n_A = node.n_samples / self.n_samples_
            
            # L_{n,A}: impurity reduction for this split
            L_n_A = node.impurity_reduction
            
            # Add weighted importance: p_{n,A} * L_{n,A}
            return {node.feature_idx: p_n_A * L_n_A}
        
        mdi_contributions = self._traverse_tree(self.tree_, calculate_mdi_contribution)
        
        # Update importances with contributions
        for feature_idx, contribution in mdi_contributions.items():
            importances[feature_idx] += contribution
        
        return importances