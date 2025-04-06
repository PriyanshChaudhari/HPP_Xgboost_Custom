import numpy as np

from scipy.optimize import differential_evolution


class GeneticDecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=10, population_size=20, generations=50):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.population_size = population_size
        self.generations = generations
        self.tree = None

    def fit(self, X, y):
        print("Inside GeneticDecisionTreeRegressor.fit()")
        X = np.nan_to_num(X, nan=np.nanmean(X))
        # X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        # y = np.where(np.isnan(y), np.nanmean(y, axis=0), y)
        y = np.nan_to_num(y, nan=np.nanmean(y))
        # y = np.nan_to_num(y, nan=np.nanmean(y))
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples == 0:
            return {'value': 0}
        node_value = np.mean(y) if len(y) > 0 else 0
        impurity = np.mean((y - node_value) ** 2) if len(y) > 0 else 0
        if depth >= self.max_depth or n_samples < self.min_samples_split or impurity == 0:
            return {'value': node_value}

        def objective(params):
            feature, threshold = int(params[0]), params[1]
            left_idx = np.where(X[:, feature] < threshold)[0]
            right_idx = np.where(X[:, feature] >= threshold)[0]
            if len(left_idx) == 0 or len(right_idx) == 0:
                return np.inf
            y_left, y_right = y[left_idx], y[right_idx]
            impurity_left = np.mean((y_left - np.mean(y_left)) ** 2) if len(y_left) > 0 else 0
            impurity_right = np.mean((y_right - np.mean(y_right)) ** 2) if len(y_right) > 0 else 0
            return (len(y_left) * impurity_left + len(y_right) * impurity_right) / max(n_samples, 1)

        bounds = [(0, n_features - 1), (np.min(X), np.max(X))]
        result = differential_evolution(objective, bounds, strategy='best1bin', popsize=self.population_size,
                                        maxiter=self.generations)
        best_feature, best_threshold = int(result.x[0]), result.x[1]

        left_idx = np.where(X[:, best_feature] < best_threshold)[0]
        right_idx = np.where(X[:, best_feature] >= best_threshold)[0]
        left_tree = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold, 'left': left_tree, 'right': right_tree}

    def _predict_one(self, x, tree):
        if 'value' in tree:
            return tree['value']
        if x[tree['feature']] < tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
