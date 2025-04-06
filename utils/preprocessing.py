import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True):
    if shuffle:
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def min_max_scale(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))