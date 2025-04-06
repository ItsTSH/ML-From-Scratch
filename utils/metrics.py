import numpy as np

def accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth target values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        Proportion of correct predictions
    """
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    """
    Calculate precision: TP / (TP + FP)
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth binary target values (0 or 1)
    y_pred : array-like
        Predicted binary values (0 or 1)
        
    Returns:
    --------
    float
        Precision score
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def recall(y_true, y_pred):
    """
    Calculate recall (sensitivity): TP / (TP + FN)
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth binary target values (0 or 1)
    y_pred : array-like
        Predicted binary values (0 or 1)
        
    Returns:
    --------
    float
        Recall score
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0

def f1_score(y_true, y_pred):
    """
    Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth binary target values (0 or 1)
    y_pred : array-like
        Predicted binary values (0 or 1)
        
    Returns:
    --------
    float
        F1 score
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def r2_score(y_true, y_pred):
    """
    Calculate R² coefficient of determination.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth target values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        R² score
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total > 0 else 0

def confusion_matrix(y_true, y_pred, classes=None):
    """
    Calculate confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth target values
    y_pred : array-like
        Predicted values
    classes : array-like, default=None
        List of class labels
        
    Returns:
    --------
    array
        Confusion matrix where rows represent true labels and columns represent predicted labels
    """
    if classes is None:
        classes = np.unique(np.concatenate((y_true, y_pred)))
    
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return cm

def gini_impurity(y):
    """
    Calculates the Gini Impurity for a list of class labels.
    """
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def roc_auc_score(y_true, y_scores, num_thresholds=100):
    """
    Compute the ROC AUC score from scratch.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1)
    y_scores : array-like of shape (n_samples,)
        Predicted probabilities or confidence scores
    num_thresholds : int, default=100
        Number of thresholds to use for calculating the ROC curve

    Returns:
    --------
    float
        Area under the ROC curve
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    thresholds = np.linspace(0, 1, num_thresholds)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        TPR = TP / (TP + FN + 1e-10)  # Sensitivity / Recall
        FPR = FP / (FP + TN + 1e-10)  # 1 - Specificity

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    # Sort by FPR for integration
    fpr_sorted_idx = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[fpr_sorted_idx]
    tpr_sorted = np.array(tpr_list)[fpr_sorted_idx]

    # Trapezoidal rule to compute AUC
    auc = np.trapezoid(tpr_sorted, fpr_sorted)
    return auc