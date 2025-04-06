import numpy as np

from custom_tree.decision_tree_genetic_algorithm import GeneticDecisionTreeRegressor


class CustomXGBoost:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_split=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        y_pred = np.full_like(y, np.mean(y), dtype=np.float64)
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = GeneticDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        predictions += np.mean([tree.tree['value'] if 'value' in tree.tree else 0 for tree in self.trees])
        return predictions

