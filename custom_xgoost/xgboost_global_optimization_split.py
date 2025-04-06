import numpy as np

from custom_tree.decision_tree_global_optimization import GlobalOptimizationDecisionTreeRegressor


class CustomXGBoostWithGlobalTree:

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=10,
                 min_impurity_decrease=1e-7):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.trees = []
        self.init_val = 0

    def fit(self, X, y):
        X = np.nan_to_num(X, nan=np.nanmean(X))

        y = np.nan_to_num(y, nan=np.nanmean(y))

        self.init_val = np.mean(y)
        # y_pred = np.full_like(y, self.init_val)
        y_pred = np.full_like(y, self.init_val, dtype=np.float64)


        for _ in range(self.n_estimators):
            residual = y - y_pred
            tree = GlobalOptimizationDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease
            )
            tree.fit(X, residual)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        X = np.nan_to_num(X, nan=np.nanmean(X))
        y_pred = np.full((X.shape[0],), self.init_val)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

