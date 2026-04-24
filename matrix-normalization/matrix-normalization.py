import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    matrix = np.array(matrix)
    if len(matrix.shape) > 2:
        return None
    if axis and axis>len(matrix.shape)-1:
        return None
    if norm_type not in ("l1", "l2", "max"):
        return None
    match norm_type:
        case "l1":
            denom = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        case "l2":
            denom = np.sqrt(np.sum(np.pow(matrix, 2), axis=axis, keepdims=True))
        case "max":
            denom = np.max(np.abs(matrix), axis=axis, keepdims=True)
    return matrix/(denom + 1e-10)