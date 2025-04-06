import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets.
    
    Parameters:
    -----------
    X : array-like
        Features data
    y : array-like
        Target data
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=None
        Controls the shuffling applied to the data
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Train and test splits of the input data
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_samples = int(n_samples * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def standardize(X, mean=None, std=None):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Parameters:
    -----------
    X : array-like
        Features data
    mean : array-like, default=None
        Mean values to use for normalization
    std : array-like, default=None
        Standard deviation values to use for normalization
        
    Returns:
    --------
    X_scaled : array
        Standardized features
    mean : array (if mean was None)
        Computed mean values
    std : array (if std was None)
        Computed standard deviation values
    """
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8), mean, std
    else:
        return (X - mean) / (std + 1e-8)

def normalize(X, min_val=None, max_val=None):
    """
    Scale features to the range [0, 1].
    
    Parameters:
    -----------
    X : array-like
        Features data
    min_val : array-like, default=None
        Minimum values to use for normalization
    max_val : array-like, default=None
        Maximum values to use for normalization
        
    Returns:
    --------
    X_scaled : array
        Normalized features
    min_val : array (if min_val was None)
        Computed minimum values
    max_val : array (if max_val was None)
        Computed maximum values
    """
    if min_val is None or max_val is None:
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8), min_val, max_val
    else:
        return (X - min_val) / (max_val - min_val + 1e-8)

def one_hot_encode(y, n_classes=None):
    """
    Convert a class vector to one-hot encoded matrix.
    
    Parameters:
    -----------
    y : array-like
        Class values
    n_classes : int, default=None
        Number of classes
        
    Returns:
    --------
    array
        One-hot encoded matrix
    """
    if n_classes is None:
        n_classes = len(np.unique(y))
    
    one_hot = np.zeros((len(y), n_classes))
    for i, val in enumerate(y):
        one_hot[i, int(val)] = 1
    
    return one_hot

def polynomial_features(X, degree=2):
    """
    Generate polynomial features up to specified degree.
    
    Parameters:
    -----------
    X : array-like
        Features data of shape (n_samples, n_features)
    degree : int, default=2
        Degree of polynomial features
        
    Returns:
    --------
    array
        Polynomial features
    """
    n_samples, n_features = X.shape
    n_output_features = 1
    
    # Calculate number of output features
    for i in range(1, degree + 1):
        n_output_features += np.math.comb(n_features + i - 1, i)
    
    X_poly = np.ones((n_samples, n_output_features))
    col_idx = 1
    
    # Generate polynomial features
    for d in range(1, degree + 1):
        if d == 1:
            X_poly[:, col_idx:col_idx + n_features] = X
            col_idx += n_features
        else:
            for i in range(n_features):
                X_poly_prev = X_poly[:, 1:col_idx].copy()
                X_poly_i = X[:, i].reshape(-1, 1)
                
                # Only multiply by features with index >= i to avoid duplicates
                valid_cols = np.arange(1, col_idx)
                for j in range(n_features):
                    if j < i:
                        # Skip columns associated with features with index < i
                        valid_cols = valid_cols[valid_cols != j+1]
                
                X_poly_new = X_poly_prev[:, valid_cols-1] * X_poly_i
                n_new_cols = X_poly_new.shape[1]
                X_poly[:, col_idx:col_idx + n_new_cols] = X_poly_new
                col_idx += n_new_cols
    
    return X_poly

def min_max_scale(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))