import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    w = np.zeros(X.shape[1])
    b = 0
    for step in range(steps):
        y_pred = _sigmoid(X@w + b)
        error = y_pred - y
        grad_w = (X.T @ error) / X.shape[0]
        grad_b = np.mean(error)
        w -= lr*grad_w
        b -= lr*grad_b
    return w, b