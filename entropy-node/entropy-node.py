import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.asarray(y)
    values, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    p = (counts/total) * np.log2(counts/total)
    return -np.sum(p).item()