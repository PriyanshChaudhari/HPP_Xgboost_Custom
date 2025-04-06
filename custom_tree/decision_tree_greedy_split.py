
import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, n_splits=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_splits = n_splits
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        node_value = np.mean(y)

        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'value': node_value}

        best_feature, best_threshold, best_gain = None, None, float('inf')
        best_left_idx, best_right_idx = None, None

        for feature in range(X.shape[1]):
            percentiles = np.percentile(X[:, feature], np.linspace(10, 90, self.n_splits))
            for threshold in percentiles:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx

                if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                    continue

                y_left, y_right = y[left_idx], y[right_idx]
                left_mean, right_mean = np.mean(y_left), np.mean(y_right)

                impurity = np.sum((y_left - left_mean) ** 2) + np.sum((y_right - right_mean) ** 2)

                if impurity < best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, impurity
                    best_left_idx, best_right_idx = left_idx, right_idx

        if best_feature is None:
            return {'value': node_value}

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[best_left_idx], y[best_left_idx], depth + 1),
            'right': self._build_tree(X[best_right_idx], y[best_right_idx], depth + 1)
        }

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

