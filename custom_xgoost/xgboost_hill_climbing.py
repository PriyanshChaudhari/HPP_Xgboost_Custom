import numpy as np

from custom_tree.decision_tree_hill_climbing import HillClimbingDecisionTreeRegressor


class CustomXGBoostRegressor:

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, n_splits=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_splits = n_splits
        self.trees = []
        self.init_val = 0

    def fit(self, X, y):
        self.init_val = np.mean(y)
        y_pred = np.full(y.shape, self.init_val)

        for _ in range(self.n_estimators):
            residual = y - y_pred
            tree = HillClimbingDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_splits=self.n_splits,
                n_hill_iter=5
            )
            tree.fit(X, residual)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full((X.shape[0],), self.init_val)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

