import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x)
    max_x = np.max(x, axis=1, keepdims=True) if len(x.shape)>1 else np.max(x)
    denom = np.sum(np.exp(x - max_x), axis=1, keepdims=True) if len(x.shape)>1 else np.sum(np.exp(x - max_x))
    return np.exp(x - max_x)/denom