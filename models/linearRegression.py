import numpy as np
from utils.losses import mse, mse_derivative
from utils.metrics import r2_score


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, verbose=False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and i % 100 == 0:
                loss = mse(y, y_predicted)
                print(f"Iteration {i}: MSE = {loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            "mse": mse(y, y_pred),
            "r2_score": r2_score(y, y_pred)
        }