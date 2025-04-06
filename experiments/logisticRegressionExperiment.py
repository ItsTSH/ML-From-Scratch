import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # optional but makes the plots prettier

from models.logisticRegression import LogisticRegression
from utils.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from utils.preprocessing import train_test_split
from sklearn.datasets import make_classification

def plot_confusion_matrix(cm):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_proba):
    thresholds = np.linspace(0, 1, 100)
    tpr = []
    fpr = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))

        tpr.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        fpr.append(FP / (FP + TN) if (FP + TN) > 0 else 0)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label='ROC Curve', color='blue')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_logistic_regression_experiment():
    # Generate data
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5, 
        n_redundant=2, 
        n_classes=2,
        random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(
        learning_rate=0.05, 
        iterations=1000, 
        l2_penalty=0.01,
        verbose=True
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Metrics
    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Results
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"ROC AUC Score  : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Visualizations
    plot_confusion_matrix(cm)
    plot_roc_curve(y_test, y_proba)

if __name__ == "__main__":
    run_logistic_regression_experiment()