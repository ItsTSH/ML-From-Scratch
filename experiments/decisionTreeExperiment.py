import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # for heatmap
import networkx as nx

from sklearn.datasets import make_classification, make_regression # to generate random data
from utils.preprocessing import train_test_split
from utils.losses import mse
from utils.metrics import confusion_matrix, accuracy

from models.decisionTree import DecisionTreeClassifier, DecisionTreeRegressor

def plot_confusion_matrix(cm):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Decision Tree Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

def plot_regression_result(X_test, y_test, y_pred):
    # Sort for a cleaner line plot
    sorted_indices = np.argsort(X_test.ravel())
    X_test_sorted = X_test.ravel()[sorted_indices]
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    plt.figure(figsize=(8, 5))
    plt.plot(X_test_sorted, y_test_sorted, label="True Values", color="blue", marker='o', linestyle='dashed')
    plt.plot(X_test_sorted, y_pred_sorted, label="Predicted", color="red", marker='x')
    plt.title("Decision Tree Regressor Predictions")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_tree(model, title="Decision Tree Structure"):
    def add_nodes_edges(node, parent=None, edge_label=""):
        if node is None:
            return

        node_id = id(node)
        label = ""
        if node.left is None and node.right is None:
            label = f"Leaf\nValue: {node.value}"
        else:
            label = f"X[{node.feature}] <= {round(node.threshold, 2)}"

        G.add_node(node_id, label=label)

        if parent is not None:
            G.add_edge(parent, node_id, label=edge_label)

        if node.left:
            add_nodes_edges(node.left, node_id, "True")
        if node.right:
            add_nodes_edges(node.right, node_id, "False")

    G = nx.DiGraph()
    add_nodes_edges(model.root)

    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
    labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=1500, node_color='lightgreen', font_size=8, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# -------------------------------
# CLASSIFICATION EXPERIMENT
# -------------------------------
print("=== Decision Tree Classifier ===")
X_cls, y_cls = make_classification(n_samples=300, n_features=4, n_informative=2, n_redundant=0, random_state=42)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2)

clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train_cls, y_train_cls)
y_pred_cls = clf.predict(X_test_cls)

acc = accuracy(y_test_cls, y_pred_cls)
print(f"Classifier Accuracy: {acc:.4f}")

# Visualize Confusion Matrix
cm = confusion_matrix(y_test_cls, y_pred_cls)
print("Confusion Matrix:\n", cm)
plot_confusion_matrix(cm)
visualize_tree(clf, title="Decision Tree Classifier Structure")

# -------------------------------
# REGRESSION EXPERIMENT
# -------------------------------
print("\n=== Decision Tree Regressor ===")
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2)

reg = DecisionTreeRegressor(max_depth=5)
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

reg_mse = mse(y_test_reg, y_pred_reg)
print(f"Regressor MSE: {reg_mse:.4f}")

visualize_tree(reg, title="Decision Tree Regressor Structure")

# Visualize Regression Result
plot_regression_result(X_test_reg, y_test_reg, y_pred_reg)