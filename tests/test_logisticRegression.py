import numpy as np
from models.logisticRegression import LogisticRegression

def test_logistic_regression_fit_predict():
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(learning_rate=0.1, iterations=1000, tol=1e-6)
    model.fit(X, y)
    preds = model.predict(X)

    accuracy = np.mean(preds == y)
    assert accuracy > 0.9, f"Accuracy too low: {accuracy}"