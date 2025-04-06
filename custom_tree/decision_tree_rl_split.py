
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, 64)
        self.out = nn.Linear(64, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc(x))
        return torch.softmax(self.out(x), dim=-1)

class RLDecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=10, min_impurity_decrease=1e-7, n_features=None, policy_lr=1e-2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_features = n_features
        self.policy_net = PolicyNetwork(input_dim=n_features+1, num_actions=n_features)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.tree = None
    
    def fit(self, X, y):
        X = np.nan_to_num(X, nan=np.nanmean(X))
        y = np.nan_to_num(y, nan=np.nanmean(y))
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples == 0:
            return {'value': 0}
        node_value = np.mean(y)
        impurity = np.mean((y - node_value) ** 2)
        if depth >= self.max_depth or n_samples < self.min_samples_split or impurity == 0:
            return {'value': node_value}
        
        state = np.concatenate([np.nan_to_num(np.mean(X, axis=0)), [impurity]])
        state_tensor = torch.FloatTensor(state)
        probs = self.policy_net(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        chosen_prob = probs[action]
        
        threshold = np.median(X[:, action]) if np.any(X[:, action]) else 0
        left_idx = np.where(X[:, action] < threshold)[0]
        right_idx = np.where(X[:, action] >= threshold)[0]
        
        if len(left_idx) == 0 or len(right_idx) == 0:
            return {'value': node_value}
        
        y_left, y_right = y[left_idx], y[right_idx]
        impurity_left = np.mean((y_left - np.mean(y_left)) ** 2) if len(y_left) > 0 else 0
        impurity_right = np.mean((y_right - np.mean(y_right)) ** 2) if len(y_right) > 0 else 0
        weighted_impurity = (len(y_left) * impurity_left + len(y_right) * impurity_right) / max(n_samples, 1)
        gain = impurity - weighted_impurity
        reward = gain  
        
        loss = -torch.log(chosen_prob + 1e-8) * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if gain < self.min_impurity_decrease:
            return {'value': node_value}
        
        left_tree = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        
        return {'feature': action, 'threshold': threshold, 'left': left_tree, 'right': right_tree}
    
    def _predict_one(self, x, tree):
        if 'value' in tree:
            return tree['value']
        if x[tree['feature']] < tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

