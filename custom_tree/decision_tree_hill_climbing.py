from copy import deepcopy

import numpy as np


class HillClimbingDecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, n_splits=10, n_hill_iter=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_splits = n_splits
        self.n_hill_iter = n_hill_iter
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        node_value = np.mean(y)
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'value': node_value}

        best_node = {'value': node_value}
        best_impurity = float('inf')

        for _ in range(self.n_hill_iter):
            feature = np.random.randint(0, X.shape[1])
            thresholds = np.percentile(X[:, feature], np.linspace(10, 90, self.n_splits))
            threshold = np.random.choice(thresholds)

            left_idx = X[:, feature] <= threshold
            right_idx = ~left_idx

            if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                continue

            y_left, y_right = y[left_idx], y[right_idx]
            left_mean, right_mean = np.mean(y_left), np.mean(y_right)

            impurity = np.sum((y_left - left_mean) ** 2) + np.sum((y_right - right_mean) ** 2)

            if impurity < best_impurity:
                best_impurity = impurity
                best_node = {
                    'feature': feature,
                    'threshold': threshold,
                    'left': deepcopy(self._build_tree(X[left_idx], y_left, depth + 1)),
                    'right': deepcopy(self._build_tree(X[right_idx], y_right, depth + 1))
                }

        return best_node

    def predict(self, X):
        return np.array([self._predict_vectorized(sample) for sample in X])

    def _predict_vectorized(self, sample):
        node = self.tree
        while 'value' not in node:
            if sample[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['value']
