import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function: f(z) = 1 / (1 + exp(-z))
    Clips values to avoid overflow.
    
    Parameters:
    -----------
    z : array-like
        Input values
        
    Returns:
    --------
    array-like
        Sigmoid output, same shape as input
    """
    # Clip values to avoid overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def relu(z):
    """
    ReLU activation function: f(z) = max(0, z)
    
    Parameters:
    -----------
    z : array-like
        Input values
        
    Returns:
    --------
    array-like
        ReLU output, same shape as input
    """
    return np.maximum(0, z)

def tanh(z):
    """
    Hyperbolic tangent activation function: f(z) = tanh(z)
    
    Parameters:
    -----------
    z : array-like
        Input values
        
    Returns:
    --------
    array-like
        Tanh output, same shape as input
    """
    return np.tanh(z)

def softmax(z):
    """
    Softmax activation function: f(z_i) = exp(z_i) / sum(exp(z_j))
    Handles numeric stability by subtracting max value.
    
    Parameters:
    -----------
    z : array-like
        Input values, typically logits
        
    Returns:
    --------
    array-like
        Softmax probabilities, same shape as input
    """
    # Shift values for numerical stability (prevent overflow) 
    shifted_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def leaky_relu(z, alpha=0.01):
    """
    Leaky ReLU activation: f(z) = max(alpha*z, z)
    
    Parameters:
    -----------
    z : array-like
        Input values
    alpha : float, default=0.01
        Slope for negative values
        
    Returns:
    --------
    array-like
        Leaky ReLU output, same shape as input
    """
    return np.maximum(alpha * z, z)