import numpy as np

def one_hot_encode(y, num_classes=None):
    y = np.array(y)
    if num_classes is None:
        num_classes = np.max(y) + 1
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def shuffle_data(X, y):
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]