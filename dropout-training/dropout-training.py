import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    rand = rng.random(x.shape) if rng else np.random.random(x.shape)
    keep = rand < (1 - p)
    pattern = np.zeros_like(x, dtype=float)
    pattern[keep] = 1 / (1 - p)
    output = x * pattern
    return output, pattern