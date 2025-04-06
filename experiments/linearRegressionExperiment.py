import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from utils.preprocessing import train_test_split, normalize
from models.linearRegression import LinearRegression

# Dataset
X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=1)
#X = normalize(X)  # Optional normalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression(learning_rate=0.05, n_iters=1000, verbose=True)
model.fit(X_train, y_train)

# Evaluation
results = model.evaluate(X_test, y_test)
print("Evaluation:", results)

# Visualize predictions vs actual data
y_pred_line = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Actual Data", alpha=0.6)
plt.plot(X, y_pred_line, color="red", label="Regression Line", linewidth=2)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression - Fit Visualization")
plt.legend()
plt.grid(True)
plt.show()