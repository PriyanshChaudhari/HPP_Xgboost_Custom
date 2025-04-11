import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.out(x), dim=-1)


class RLDecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=10, min_impurity_decrease=1e-6, n_features=None, policy_lr=1e-3,
                 gamma=0.99):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.gamma = gamma
        self.n_features = n_features
        self.policy_net = PolicyNetwork(input_dim=n_features + 5, num_actions=n_features)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.tree = None
        self.episode_log_probs = []
        self.episode_rewards = []

    def _extract_state(self, X, y):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        impurity = np.mean((y - np.mean(y)) ** 2)
        return np.concatenate([mean, [np.mean(std), np.mean(q25), np.mean(q75), np.mean(X), impurity]])

    def fit(self, X, y, episodes=10):
        X = np.nan_to_num(X, nan=np.nanmean(X))
        y = np.nan_to_num(y, nan=np.nanmean(y))

        for _ in range(episodes):
            self.episode_log_probs = []
            self.episode_rewards = []
            self.tree = self._build_tree(X, y, depth=0)
            self._update_policy()

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if n_samples == 0:
            return {'value': 0}
        node_value = np.mean(y)
        impurity = np.mean((y - node_value) ** 2)
        if depth >= self.max_depth or n_samples < self.min_samples_split or impurity == 0:
            return {'value': node_value}

        state = self._extract_state(X, y)
        state_tensor = torch.FloatTensor(state)
        probs = self.policy_net(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        chosen_prob = probs[action]

        self.episode_log_probs.append(torch.log(chosen_prob + 1e-8))

        feature_idx = action.item()
        threshold = np.median(X[:, feature_idx])
        left_idx = X[:, feature_idx] < threshold
        right_idx = ~left_idx

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            self.episode_rewards.append(0)
            return {'value': node_value}

        y_left, y_right = y[left_idx], y[right_idx]
        impurity_left = np.var(y_left) if len(y_left) > 0 else 0
        impurity_right = np.var(y_right) if len(y_right) > 0 else 0
        weighted_impurity = (len(y_left) * impurity_left + len(y_right) * impurity_right) / n_samples
        gain = impurity - weighted_impurity

        reward = max(gain, 0)
        self.episode_rewards.append(reward)

        if gain < self.min_impurity_decrease:
            return {'value': node_value}

        left_tree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return {'feature': feature_idx, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def _update_policy(self):
        G = 0
        returns = []
        for r in reversed(self.episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        loss = 0
        for log_prob, G in zip(self.episode_log_probs, returns):
            loss -= log_prob * G
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _predict_one(self, x, tree):
        if 'value' in tree:
            return tree['value']
        if x[tree['feature']] < tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
