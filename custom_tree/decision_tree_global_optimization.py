import numpy as np

class GlobalOptimizationDecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=10, min_impurity_decrease=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None
    
    def fit(self, X, y):
        X = np.nan_to_num(X, nan=np.nanmean(X))
        y = np.nan_to_num(y, nan=np.nanmean(y))
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples == 0:
            return {'value': 0}
        node_value = np.mean(y) if len(y) > 0 else 0
        impurity = np.mean((y - node_value)**2) if len(y) > 0 else 0
        if depth >= self.max_depth or n_samples < self.min_samples_split or impurity == 0:
            return {'value': node_value}
        
        best_feature, best_threshold, best_gain = None, None, 0
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = np.where(X[:, feature] < threshold)[0]
                right_idx = np.where(X[:, feature] >= threshold)[0]
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                
                y_left, y_right = y[left_idx], y[right_idx]
                impurity_left = np.mean((y_left - np.mean(y_left))**2) if len(y_left) > 0 else 0
                impurity_right = np.mean((y_right - np.mean(y_right))**2) if len(y_right) > 0 else 0
                weighted_impurity = (len(y_left) * impurity_left + len(y_right) * impurity_right) / max(n_samples, 1)
                gain = impurity - weighted_impurity
                
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain
        
        if best_gain < self.min_impurity_decrease:
            return {'value': node_value}
        
        left_idx = np.where(X[:, best_feature] < best_threshold)[0]
        right_idx = np.where(X[:, best_feature] >= best_threshold)[0]
        left_tree = self._build_tree(X[left_idx, :], y[left_idx], depth+1)
        right_tree = self._build_tree(X[right_idx, :], y[right_idx], depth+1)
        
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

