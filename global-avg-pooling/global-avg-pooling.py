import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.array(x).copy()
    if x.ndim not in [3, 4]:
        raise ValueError
    return np.mean(x, axis=(x.ndim-1, x.ndim-2))