import numpy as np
from models.decisionTree import DecisionTreeClassifier, DecisionTreeRegressor

def test_decision_tree_classifier_fit_predict():
    np.random.seed(42)
    X = np.random.randn(150, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)

    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X, y)
    preds = clf.predict(X)

    accuracy = np.mean(preds == y)
    assert accuracy > 0.9, f"Classifier accuracy too low: {accuracy}"


def test_decision_tree_regressor_fit_predict():
    np.random.seed(42)
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * np.random.randn(100)

    reg = DecisionTreeRegressor(max_depth=5)
    reg.fit(X, y)
    preds = reg.predict(X)

    mse = np.mean((preds - y) ** 2)
    assert mse < 0.05, f"Regressor MSE too high: {mse}"