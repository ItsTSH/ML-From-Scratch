import numpy as np

def zeros_init(shape):
    return np.zeros(shape)

def random_init(shape):
    return np.random.randn(*shape) * 0.01

def he_init(shape):
    return np.random.randn(*shape) * np.sqrt(2. / shape[0])

def xavier_init(shape):
    return np.random.randn(*shape) * np.sqrt(1. / shape[0])