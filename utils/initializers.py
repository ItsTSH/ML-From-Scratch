import numpy as np

def zeros(shape):
    """
    Initialize weights or biases with zeros.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the weight matrix or bias vector
        
    Returns:
    --------
    array
        Zero initialized weights
    """
    return np.zeros(shape)

def ones(shape):
    """
    Initialize weights or biases with ones.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the weight matrix or bias vector
        
    Returns:
    --------
    array
        One initialized weights
    """
    return np.ones(shape)

def random_uniform(shape, low=-0.1, high=0.1):
    """
    Initialize weights with random values from uniform distribution.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the weight matrix
    low : float, default=-0.1
        Lower bound of uniform distribution
    high : float, default=0.1
        Upper bound of uniform distribution
        
    Returns:
    --------
    array
        Uniformly initialized weights
    """
    return np.random.uniform(low=low, high=high, size=shape)

def random_normal(shape, mean=0.0, std=0.1):
    """
    Initialize weights with random values from normal distribution.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the weight matrix
    mean : float, default=0.0
        Mean of normal distribution
    std : float, default=0.1
        Standard deviation of normal distribution
        
    Returns:
    --------
    array
        Normally initialized weights
    """
    return np.random.normal(loc=mean, scale=std, size=shape)

def xavier_uniform(shape):
    """
    Xavier/Glorot uniform initialization.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the weight matrix (fan_in, fan_out)
        
    Returns:
    --------
    array
        Xavier uniform initialized weights
    """
    fan_in, fan_out = shape
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(low=-limit, high=limit, size=shape)

def xavier_normal(shape):
    """
    Xavier/Glorot normal initialization.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the weight matrix (fan_in, fan_out)
        
    Returns:
    --------
    array
        Xavier normal initialized weights
    """
    fan_in, fan_out = shape
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(loc=0.0, scale=std, size=shape)

def he_uniform(shape):
    """
    He uniform initialization for ReLU activations.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the weight matrix (fan_in, fan_out)
        
    Returns:
    --------
    array
        He uniform initialized weights
    """
    fan_in, fan_out = shape
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(low=-limit, high=limit, size=shape)

def he_normal(shape):
    """
    He normal initialization for ReLU activations.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the weight matrix (fan_in, fan_out)
        
    Returns:
    --------
    array
        He normal initialized weights
    """
    fan_in, fan_out = shape
    std = np.sqrt(2 / fan_in)
    return np.random.normal(loc=0.0, scale=std, size=shape)