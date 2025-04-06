import numpy as np
from utils.activations import sigmoid
from utils.losses import binary_cross_entropy

class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch.
    Uses gradient descent to minimize binary cross-entropy loss.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for gradient descent
    iterations : int, default=1000
        Maximum number of iterations for gradient descent
    batch_size : int, default=None
        Size of mini-batches for stochastic gradient descent.
        If None, use batch gradient descent
    l1_penalty : float, default=0.0
        L1 regularization strength (LASSO)
    l2_penalty : float, default=0.0
        L2 regularization strength (Ridge)
    tol : float, default=1e-4
        Tolerance for stopping criterion
    verbose : bool, default=False
        If True, print loss during training
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000, batch_size=None,
                 l1_penalty=0.0, l2_penalty=0.0, tol=1e-4, verbose=False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.tol = tol
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.history = {'loss': []}

    def _initialize_parameters(self, n_features):
        """Initialize weights and bias"""
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def _forward(self, X):
        """Compute the linear combination and apply sigmoid activation"""
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def _compute_gradients(self, X, y, y_pred):
        """Compute gradients of loss with respect to weights and bias"""
        m = X.shape[0]
        error = y_pred - y  # Derivative of binary cross-entropy w.r.t sigmoid output
        
        # Compute gradients
        dw = (1/m) * np.dot(X.T, error)
        db = (1/m) * np.sum(error)
        
        # Add L1 regularization term (with subgradient for L1)
        if self.l1_penalty > 0:
            dw += self.l1_penalty * np.sign(self.weights)
        
        # Add L2 regularization term
        if self.l2_penalty > 0:
            dw += self.l2_penalty * self.weights
            
        return dw, db

    def _update_parameters(self, dw, db):
        """Update weights and bias using gradient descent"""
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X, y, verbose=None):
        """
        Fit the logistic regression model to training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (0 or 1)
        verbose : bool, default=None
            If provided, overrides the instance's verbose attribute
            
        Returns:
        --------
        self : LogisticRegression
            Fitted model
        """
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y).reshape(-1)
        
        n_samples, n_features = X.shape
        verbose = self.verbose if verbose is None else verbose
        
        # Initialize parameters
        self._initialize_parameters(n_features)
        
        # Initialize history
        self.history = {'loss': []}
        
        # Training loop with early stopping
        prev_loss = float('inf')
        
        for i in range(self.iterations):
            # Mini-batch or batch gradient descent
            if self.batch_size is None:
                indices = np.arange(n_samples)
                batch_indices = [indices]
            else:
                indices = np.random.permutation(n_samples)
                batch_indices = [indices[k:k+self.batch_size] 
                                for k in range(0, n_samples, self.batch_size)]
            
            epoch_loss = 0
            
            for batch_idx in batch_indices:
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward pass
                y_pred = self._forward(X_batch)
                
                # Compute loss with regularization
                batch_loss = binary_cross_entropy(y_batch, y_pred)
                
                # Add L1 regularization to loss
                if self.l1_penalty > 0:
                    batch_loss += self.l1_penalty * np.sum(np.abs(self.weights))
                
                # Add L2 regularization to loss
                if self.l2_penalty > 0:
                    batch_loss += 0.5 * self.l2_penalty * np.sum(np.square(self.weights))
                
                epoch_loss += batch_loss * len(batch_idx) / n_samples
                
                # Compute gradients
                dw, db = self._compute_gradients(X_batch, y_batch, y_pred)
                
                # Update parameters
                self._update_parameters(dw, db)
            
            # Track loss history
            self.history['loss'].append(epoch_loss)
            
            if verbose and (i % max(1, self.iterations // 10) == 0):
                print(f"Iteration {i}: Loss = {epoch_loss:.6f}")
            
            # Check for convergence
            if abs(prev_loss - epoch_loss) < self.tol:
                if verbose:
                    print(f"Converged at iteration {i} with loss {epoch_loss:.6f}")
                break
                
            prev_loss = epoch_loss
            
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        array of shape (n_samples,)
            Probability of class 1 for each sample
        """
        X = np.array(X)
        return self._forward(X)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
        threshold : float, default=0.5
            Decision threshold for class prediction
            
        Returns:
        --------
        array of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def score(self, X, y):
        """
        Return the accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels for X
            
        Returns:
        --------
        float
            Accuracy score
        """
        X = np.array(X)
        y = np.array(y).reshape(-1)
        return np.mean(self.predict(X) == y)