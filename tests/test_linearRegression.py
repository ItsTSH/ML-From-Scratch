import numpy as np
from models.linearRegression import LinearRegression

def test_linear_regression_fit_predict():
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(100)

    model = LinearRegression(learning_rate=0.01, iterations=500)
    model.fit(X, y)
    preds = model.predict(X)

    mse = np.mean((preds - y) ** 2)
    assert mse < 1.0, f"MSE too high: {mse}"