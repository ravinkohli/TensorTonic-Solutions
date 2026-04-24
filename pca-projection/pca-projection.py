import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.array(X, dtype=float)
    X -= np.mean(X, axis=0, keepdims=True)
    C = 1/(X.shape[0]-1) * (X.T @ X)
    values, vectors = np.linalg.eigh(C)
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:, idx]
    return X @ vectors[:, :k]
