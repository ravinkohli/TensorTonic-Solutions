import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    y_pred = np.clip(y_pred, eps, 1-eps)
    losses = []
    for i, y in enumerate(y_true):
        losses.append(-1*(y*np.log(y_pred[i] + (1-y)*np.log(1-y_pred[i]))))
    return losses
        