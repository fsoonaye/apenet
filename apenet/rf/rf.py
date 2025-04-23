# apenet/apenet/rf/random_forest.py

import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check for a maximum depth
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < self.min_samples_split:
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

    def _best_split(self, X, y, n_samples, n_features):
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
                gini = (left_count/n_samples)*gini_left + (right_count/n_samples)*gini_right
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    split_idx = idx
                    split_threshold = (thresholds[i] + thresholds[i - 1]) / 2
        return split_idx, split_threshold

    def _gini(self, counts, n):
        if n == 0:
            return 0
        return 1.0 - sum((num / n) ** 2 for num in counts)

    def _split(self, X_column, split_threshold):
        left_idxs = np.where(X_column <= split_threshold)[0]
        right_idxs = np.where(X_column > split_threshold)[0]
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = Counter(y.tolist())
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            X_samp, y_samp = self._bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.randint(0, n_samples, (n_samples,))
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], 0, tree_preds)
        return tree_preds
