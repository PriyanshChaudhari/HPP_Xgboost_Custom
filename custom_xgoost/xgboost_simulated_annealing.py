import numpy as np

from custom_tree.decision_tree_simulated_annealing import SADecisionTreeRegressor


class CustomXGBoostRegressor:
    def __init__(
            self,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            n_splits=10,
            initial_temp=1.0,
            cooling_rate=0.95,
            max_iter=50
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_splits = n_splits
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.trees = []
        self.init_val = 0

    def fit(self, X, y):
        self.init_val = np.mean(y)
        y_pred = np.full(y.shape, self.init_val)

        for _ in range(self.n_estimators):
            residual = y - y_pred
            tree = SADecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_splits=self.n_splits,
                initial_temp=self.initial_temp,
                cooling_rate=self.cooling_rate,
                max_iter=self.max_iter
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
