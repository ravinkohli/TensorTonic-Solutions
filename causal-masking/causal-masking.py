import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    scores = np.asarray(scores)
    mask = np.ones(scores.shape[-2:], dtype=bool)
    mask = np.triu(mask, 1)

    scores = scores.copy()
    scores[..., mask] = mask_value
    return scores