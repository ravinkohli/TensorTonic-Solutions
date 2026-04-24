import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    if not matrix: return None
    if not isinstance(matrix[0], list): return None
    if len(matrix)>1 and len(matrix[0]) != len(matrix[1]):
        return None
    matrix = np.array(matrix)
    if matrix.ndim !=2 or matrix.shape[0] != matrix.shape[1]:
        return None
    eigenvals = np.linalg.eigvals(matrix)
    return eigenvals[np.lexsort((eigenvals.real, eigenvals.imag))]