import numpy as np


class SADecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, n_splits=10, initial_temp=1.0, cooling_rate=0.95, max_iter=50):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_splits = n_splits
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _impurity(self, y_left, y_right):
        left_mean, right_mean = np.mean(y_left), np.mean(y_right)
        return np.sum((y_left - left_mean) ** 2) + np.sum((y_right - right_mean) ** 2)

    def _build_tree(self, X, y, depth):
        node_value = np.mean(y)

        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'value': node_value}

        best_feature = None
        best_threshold = None
        best_impurity = float('inf')
        best_left_idx = None
        best_right_idx = None

        temp = self.initial_temp

        for _ in range(self.max_iter):
            feature = np.random.randint(0, X.shape[1])
            percentiles = np.percentile(X[:, feature], np.linspace(10, 90, self.n_splits))
            threshold = np.random.choice(percentiles)

            left_idx = X[:, feature] <= threshold
            right_idx = ~left_idx

            if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                continue

            y_left, y_right = y[left_idx], y[right_idx]
            impurity = self._impurity(y_left, y_right)

            if impurity < best_impurity or np.random.rand() < np.exp(-(impurity - best_impurity) / temp):
                best_feature, best_threshold, best_impurity = feature, threshold, impurity
                best_left_idx, best_right_idx = left_idx, right_idx

            temp *= self.cooling_rate

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
