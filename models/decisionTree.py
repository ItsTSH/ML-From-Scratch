import numpy as np
from utils.losses import mse
from utils.metrics import gini_impurity

# Represents a node in the decision tree
class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Index of the feature to split on
        self.threshold = threshold      # Value of the feature to split at
        self.left = left                # Left subtree (<= threshold)
        self.right = right              # Right subtree (> threshold)
        self.value = value              # Value if it's a leaf node (prediction)

# Base class for Decision Tree (abstract class not to be used directly)
class DecisionTreeBase:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.root = None
        self.max_depth = max_depth                      # Maximum tree depth
        self.min_samples_split = min_samples_split      # Minimum samples to allow split

    def _is_pure(self, y):
        """Check if all labels are the same (node is pure)."""
        return len(np.unique(y)) == 1

    def _split(self, X_column, threshold):
        """Splits data based on a feature column and a threshold."""
        left = np.argwhere(X_column <= threshold).flatten()
        right = np.argwhere(X_column > threshold).flatten()
        return left, right

    def _best_split(self, X, y):
        """Finds the best feature and threshold that gives the best split score."""
        best_score = float('inf')
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])  # Only unique values matter
            for threshold in thresholds:
                left_idx, right_idx = self._split(X[:, feature], threshold)
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue  # Skip if no split happens

                score = self._calculate_score(y[left_idx], y[right_idx])
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        """
        Recursively builds the decision tree.
        Stops if the node is pure, max depth is reached, or not enough samples.
        """
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            self._is_pure(y)):
            return DecisionNode(value=self._leaf_value(y))  # Leaf node

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return DecisionNode(value=self._leaf_value(y))

        left_idx, right_idx = self._split(X[:, feature], threshold)
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return DecisionNode(feature, threshold, left_subtree, right_subtree)

    def _predict_sample(self, x, node):
        """Traverse the tree to make a prediction for one sample."""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def fit(self, X, y):
        """Fits the decision tree to the data."""
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """Predicts values for multiple samples."""
        return np.array([self._predict_sample(x, self.root) for x in X])

    # ============================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # ============================
    
    def _calculate_score(self, left_y, right_y):
        """
        Abstract method to calculate split score.
        - In classification, it's Gini impurity.
        - In regression, it's Mean Squared Error.
        """
        raise NotImplementedError

    def _leaf_value(self, y):
        """
        Abstract method to get prediction at a leaf.
        - In classification: most frequent class.
        - In regression: mean value.
        """
        raise NotImplementedError

# ===============================
# Subclass for Classification
# ===============================
class DecisionTreeClassifier(DecisionTreeBase):
    def _calculate_score(self, left_y, right_y):
        total = len(left_y) + len(right_y)
        weighted_gini = (len(left_y) * gini_impurity(left_y) + 
                         len(right_y) * gini_impurity(right_y)) / total
        return weighted_gini

    def _leaf_value(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]  # Return the most common class

# ===============================
# Subclass for Regression
# ===============================
class DecisionTreeRegressor(DecisionTreeBase):
    def _calculate_score(self, left_y, right_y):
        # Reuse imported mse to compute weighted MSE
        total = len(left_y) + len(right_y)
        weighted_mse = (len(left_y) * mse(left_y, np.full_like(left_y, np.mean(left_y))) +
                        len(right_y) * mse(right_y, np.full_like(right_y, np.mean(right_y)))) / total
        return weighted_mse

    def _leaf_value(self, y):
        return np.mean(y)  # Return the average value for regression