import numpy as np

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
import numpy as np
from utils.activations import sigmoid

def mse(y_true, y_pred):
    """
    Mean squared error loss function: L = (1/n) * sum((y_true - y_pred)^2)
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth target values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        Mean squared error value
    """
    return np.mean(np.square(y_true - y_pred))

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary cross-entropy loss function: 
    L = -(1/n) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth binary target values (0 or 1)
    y_pred : array-like
        Predicted probabilities (between 0 and 1)
    epsilon : float, default=1e-15
        Small value to avoid log(0)
        
    Returns:
    --------
    float
        Binary cross-entropy value
    """
    # Clip predictions to avoid log(0) or log(1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Categorical cross-entropy loss function: 
    L = -(1/n) * sum(sum(y_true_i * log(y_pred_i)))
    
    Parameters:
    -----------
    y_true : array-like
        One-hot encoded ground truth values
    y_pred : array-like
        Predicted class probabilities (must sum to 1 for each sample)
    epsilon : float, default=1e-15
        Small value to avoid log(0)
        
    Returns:
    --------
    float
        Categorical cross-entropy value
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def logistic_loss(y_true, logits):
    """
    Logistic loss function for binary classification:
    L = (1/n) * sum(log(1 + exp(-y_true * logits)))
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth binary target values (-1 or 1)
    logits : array-like
        Raw logit values before sigmoid activation
        
    Returns:
    --------
    float
        Logistic loss value
    """
    # Convert 0/1 labels to -1/1 if needed
    if np.any((y_true == 0) | (y_true == 1)):
        y_true = 2 * y_true - 1
        
    # Compute logistic loss with numeric stability
    z = y_true * logits
    # Use log(1 + exp(-z)) when z is positive, and -z + log(1 + exp(z)) when z is negative
    return np.mean(np.log(1 + np.exp(-np.abs(z))) + np.maximum(-z, 0))