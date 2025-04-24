# apenet/rf/decision_tree.py
import numpy as np
from collections import Counter

from typing import Optional, Tuple

class Node:
    """
    A node in the decision tree.

    Parameters
    ----------
    feature : int, default=None
        The index of the feature to split on.
    threshold : float, default=None
        The threshold value to split the feature.
    left : Node, default=None
        The left child node.
    right : Node, default=None
        The right child node.
    value : int, default=None
        The value of the node if it is a leaf node.
    """
    def __init__(self,
                 feature: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: Optional['Node'] = None,
                 right: Optional['Node'] = None, *,
                 value: Optional[int] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        """
        Check if the node is a leaf node.

        Returns
        -------
        bool
            True if the node is a leaf node, False otherwise.
        """
        return self.value is not None

class DecisionTree:
    """
    A decision tree classifier.

    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    """
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the decision tree to the data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target labels.
        """
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> 'Node':
        """
        Recursively grow the tree.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target labels.
        depth : int, default=0
            The current depth of the tree.

        Returns
        -------
        Node
            The root node of the grown tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check for a maximum depth
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or \
                n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idx, threshold = self._best_split(X, y, n_samples, n_features)
        if feat_idx is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, feat_idx], threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feat_idx, threshold, left, right)

    def _best_split(self, X: np.ndarray, y: np.ndarray, n_samples: int, n_features: int) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best split for the data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray
            The target labels.
        n_samples : int
            The number of samples.
        n_features : int
            The number of features.

        Returns
        -------
        Tuple[Optional[int], Optional[float]]
            The index of the feature to split on and the threshold value.
        """
        best_gini = 1.0
        split_idx, split_threshold = None, None
        unique_classes = np.unique(y).tolist()
        for idx in range(n_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx].tolist(), y.tolist())))
            num_left = Counter()
            num_right = Counter(classes)
            for i in range(1, n_samples):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                left_count = i
                right_count = n_samples - i
                gini_left = self._gini([num_left[cls] for cls in unique_classes], left_count)
                gini_right = self._gini([num_right[cls] for cls in unique_classes], right_count)
                gini = (left_count / n_samples) * gini_left + (right_count / n_samples) * gini_right
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    split_idx = idx
                    split_threshold = (thresholds[i] + thresholds[i - 1]) / 2
        return split_idx, split_threshold

    def _gini(self, counts: list, n: int) -> float:
        """
        Calculate the Gini impurity.

        Parameters
        ----------
        counts : list
            The counts of each class.
        n : int
            The total number of samples.

        Returns
        -------
        float
            The Gini impurity.
        """
        if n == 0:
            return 0
        return 1.0 - sum((num / n) ** 2 for num in counts)

    def _split(self, X_column: np.ndarray, split_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the data based on a threshold.

        Parameters
        ----------
        X_column : np.ndarray
            The column of data to split.
        split_threshold : float
            The threshold value to split the data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The indices of the left and right splits.
        """
        left_idxs = np.where(X_column <= split_threshold)[0]
        right_idxs = np.where(X_column > split_threshold)[0]
        return left_idxs, right_idxs

    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Find the most common label in the data.

        Parameters
        ----------
        y : np.ndarray
            The target labels.

        Returns
        -------
        int
            The most common label.
        """
        counter = Counter(y.tolist())
        most_common = counter.most_common(1)[0][0]
        return most_common

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
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: 'Node') -> int:
        """
        Traverse the tree to make a prediction.

        Parameters
        ----------
        x : np.ndarray
            The input sample.
        node : Node
            The current node in the tree.

        Returns
        -------
        int
            The predicted label.
        """
        if node.is_leaf_node():
            return node.value
        
        if node.feature >= len(x):
            return self._most_common_label(x)

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def feature_importance(self) -> dict:
        """
        Calculate the feature importance.

        Returns
        -------
        dict
            A dictionary mapping feature indices to their importance.
        """
        importance = Counter()

        def recurse(node):
            if not node.is_leaf_node():
                importance[node.feature] += 1
                recurse(node.left)
                recurse(node.right)

        recurse(self.tree)
        return importance