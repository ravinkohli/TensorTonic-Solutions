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
    y = np.array(y).reshape(-1, 1)
    W = np.random.rand(X.shape[1], 1)
    b = np.random.rand(1).item()
    for i in range(steps):
        pred = _sigmoid(np.matmul(X, W) + b)
        error = pred-y
        grad_w = 1/X.shape[0] * np.matmul(X.T, error)
        grad_b = np.mean(error)
        W -= lr * grad_w
        b -= lr * grad_b
    return W.reshape(X.shape[1]), b