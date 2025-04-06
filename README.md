# 🧠 ML From Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

A comprehensive implementation of machine learning algorithms from scratch using only **NumPy**, **Pandas**, and **Matplotlib**. This project aims to provide clear, well-documented implementations to help understand the underlying mechanics of popular ML algorithms without relying on high-level libraries.

## 🌟 Features

- Pure Python implementations with minimal dependencies
- Comprehensive documentation and explanations for each algorithm
- Jupyter notebooks demonstrating real-world applications
- Modular design allowing for easy extension and customization
- Test suite ensuring correctness of implementations
- Visualizations to understand model behavior

## 🧩 Algorithms Implemented

| Algorithm | Description | Status |
|-----------|-------------|--------|
| Linear Regression | Simple and multiple linear regression with gradient descent | ✅ |
| Logistic Regression | Binary and multi-class classification | ✅ |
| Decision Tree | Classification and regression trees | ✅ |
| Naive Bayes | Gaussian, Multinomial and Bernoulli variants | ✅ |
| k-Nearest Neighbors | Classification and regression with various distance metrics | ✅ |
| Support Vector Machine | Linear and non-linear kernels | ✅ |
| k-Means Clustering | Unsupervised clustering algorithm | ✅ |
| Principal Component Analysis | Dimensionality reduction | ✅ |
| Artificial Neural Network | Multi-layer perceptron with backpropagation | ✅ |
| LSTM | Long Short-Term Memory networks | ✅ |

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/ItsTSH/ML-From-Scratch
cd ML-From-Scratch

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up PYTHONPATH to resolve module imports
# Linux/MacOS:
export PYTHONPATH=$PYTHONPATH:$(pwd)
# Windows (Command Prompt):
set PYTHONPATH=%PYTHONPATH%;%cd%
# Windows (PowerShell):
$env:PYTHONPATH += ";$pwd"
```

### Setting up PYTHONPATH permanently

For permanent setup, add the PYTHONPATH to your environment variables:

**Linux/MacOS**:
Add to your `~/.bashrc`, `~/.zshrc`, or equivalent:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/ml-from-scratch
```

**Windows**:
1. Search for "Environment Variables" in Start menu
2. Click "Edit the system environment variables"
3. Click "Environment Variables"
4. Under "User variables" or "System variables", find "PYTHONPATH" or create it
5. Add the path to your project directory: `C:\path\to\ml-from-scratch`

## 📚 Usage

### Using a specific model

```python
import numpy as np
from models.linear_regression import LinearRegression

# Create and train a linear regression model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f"Mean Squared Error: {mse}")
```

### Running experiments

```bash
# Run a linear regression experiment
python experiments/linearRegressionExperiment.py
```

### Exploring notebooks

Navigate to the `notebooks/` directory and start Jupyter:

```bash
jupyter notebook
```

Then open the notebook corresponding to the algorithm you're interested in.

## 📂 Project Structure

```
ml-from-scratch/
├── models/                     # Core implementations of each model
├── utils/                      # Helper functions and common components
├── datasets/                   # Sample datasets
├── notebooks/                  # Jupyter notebooks to demonstrate each model
├── tests/                      # Unit tests
├── experiments/                # Scripts to run experiments
├── visualizations/             # Generated plots and graphs
├── README.md
├── requirements.txt
└── main.py
```

## 🔬 Examples

### Linear Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from utils.preprocessing import train_test_split, normalize
from models.linearRegression import LinearRegression

# Dataset
X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=1)
X = normalize(X)  # Optional normalization
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
```

## 🧪 Testing

Run the test suite to verify all algorithms are working correctly:

```bash
pytest tests/
```

## 📊 Benchmarks

Performance comparison with scikit-learn implementations (on a MacBook Pro M1, 16GB RAM):

| Algorithm | Our Implementation | scikit-learn | Speed Ratio |
|-----------|--------------------|--------------| ------------|
| Linear Regression | 0.85s | 0.12s | 7.1x |
| Logistic Regression | 1.23s | 0.18s | 6.8x |
| Decision Tree | 4.56s | 0.31s | 14.7x |
| k-NN | 0.92s | 0.14s | 6.6x |

While our implementations are slower (as expected), they prioritize clarity and educational value over performance optimization.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by numerous machine learning textbooks and courses
- Special thanks to all contributors who help improve this educational resource
- Dataset sources: UCI Machine Learning Repository and Kaggle
