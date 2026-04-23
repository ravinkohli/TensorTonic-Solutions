import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_pred = np.array(y_pred)
    class_probs = y_pred[np.arange(len(y_true)), y_true]
    return -np.mean(np.log(class_probs))