# Import the decision_trees
# I asked Claude.ai to add extra information.

import numpy as np
from collections import Counter

# Node class (represents a single node in the decision tree)
class Node:
    """
    Represents a node in the decision tree.
    A node can be either an internal node with a splitting rule, or a leaf node with a prediction value.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # For internal nodes
        self.feature = feature      # Index of the feature used for splitting
        self.threshold = threshold  # Threshold value for the split decision
        self.left = left            # Left child node (samples where feature <= threshold)
        self.right = right          # Right child node (samples where feature > threshold)
        
        # For leaf nodes
        self.value = value          # Prediction value at this leaf node

    def is_leaf_node(self):
        """Check if the current node is a leaf node (has a prediction value)."""
        return self.value is not None


# Decision Tree Classifier
class DecisionTree:
    """
    Decision Tree Classifier implementation using CART (Classification and Regression Trees) algorithm.
    Uses information gain with entropy as the splitting criterion.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        """
        Initialize a Decision Tree Classifier.
        
        Parameters:
        -----------
        min_samples_split : int
            The minimum number of samples required to split an internal node
        max_depth : int
            The maximum depth of the tree
        n_features : int or None
            Number of features to consider when looking for the best split.
            If None, all features are considered.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        Build the decision tree by recursively finding the best splits.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            The training input samples
        y : numpy array, shape (n_samples,)
            The target values
            
        Returns:
        --------
        self : object
        """
        # If n_features not specified, use all features, otherwise use min(n_features, total features)
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
        return self

    def predict(self, X):
        """
        Predict class for X.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            The input samples
            
        Returns:
        --------
        y : numpy array, shape (n_samples,)
            The predicted classes
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])


    # Helper Methods for Tree Building and Prediction
    # ---------------------------------------------

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree by selecting the best splits.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            The training input samples
        y : numpy array, shape (n_samples,)
            The target values
        depth : int
            Current depth of the tree
            
        Returns:
        --------
        node : Node
            The root node of the built tree
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or               # Max depth reached
            n_labels == 1 or                         # Pure node (only one class)
            n_samples < self.min_samples_split):     # Not enough samples to split
            leaf_value = self._most_common_label(y)  # Assign most common label
            return Node(value=leaf_value)

        # Randomly select a subset of features to consider for splitting
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split among the selected features
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        
        # Handle edge case: if split doesn't actually split the data
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
            
        # Recursively build left and right subtrees
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_thresh, left=left, right=right)

    def _most_common_label(self, y):
        """
        Find the most common label in a set of target values.
        
        Parameters:
        -----------
        y : numpy array
            Target values
            
        Returns:
        --------
        label : 
            Most common label
        """
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _best_split(self, X, y, feat_idxs):
        """
        Find the best feature and threshold for splitting based on information gain.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target values
        feat_idxs : numpy array
            Indices of features to consider
            
        Returns:
        --------
        best_feat_idx, best_threshold : tuple
            The best feature index and threshold value for splitting
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        # Try each feature
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            # Try each threshold for the current feature
            for threshold in thresholds:
                # Calculate information gain for this split
                gain = self._information_gain(y, X_column, threshold)

                # Update best split if this one is better
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        """
        Calculate information gain for a split.
        Information Gain = entropy(parent) - weighted_average(entropy(children))
        
        Parameters:
        -----------
        y : numpy array
            Target values
        X_column : numpy array
            Feature values for a single feature
        threshold : float
            Threshold value for splitting
            
        Returns:
        --------
        info_gain : float
            Information gain from the split
        """
        # Calculate entropy of parent node
        parent_entropy = self._entropy(y)

        # Split the data
        left_idxs, right_idxs = self._split(X_column, threshold)

        # If split doesn't actually split the data, return 0 gain
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate weighted average entropy of children
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # Information gain is the difference between parent entropy and weighted child entropy
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        """
        Calculate entropy of a distribution.
        Entropy = -sum(p_i * log(p_i)) where p_i is the probability of class i.
        
        Parameters:
        -----------
        y : numpy array
            Target values
            
        Returns:
        --------
        entropy : float
            Entropy of the distribution
        """
        hist = np.bincount(y)
        ps = hist / len(y)  # Probabilities of each class
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _split(self, X_column, split_threshold):
        """
        Split the data based on a feature and threshold.
        
        Parameters:
        -----------
        X_column : numpy array
            Feature values for a single feature
        split_threshold : float
            Threshold value for splitting
            
        Returns:
        --------
        left_idxs, right_idxs : tuple of numpy arrays
            Indices for left (<=threshold) and right (>threshold) splits
        """
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction for a single sample.
        
        Parameters:
        -----------
        x : numpy array
            Single sample features
        node : Node
            Current node in the tree
            
        Returns:
        --------
        prediction : 
            Predicted class value
        """
        # If this is a leaf node, return its value
        if node.is_leaf_node():
            return node.value

        # Otherwise, decide whether to go left or right based on the split rule
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)