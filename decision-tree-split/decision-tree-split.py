import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)
    def gini(y: np.ndarray) -> float:
        values, counts = np.unique(y, return_counts=True)
        p=counts/counts.sum()
        return 1 - np.sum(p**2).item()
    parent_gini = gini(y)
    gains = []
    f_thres = []
    total = y.shape[0]
    for i in range(X.shape[1]):
        features = np.unique(sorted(X[:, i]))
        if len(features) < 2:
            continue
        mid = [(features[i]+features[i+1])/2 for i in range(len(features)-1)]
        for threshold in mid:
            f_thres.append((i, threshold))
            left = X[:, i] <= threshold
            right = X[:, i] > threshold
            weighted_gini = (left.sum()/total)*gini(y[left]) + (right.sum()/total)*gini(y[right])
            gains.append(parent_gini-weighted_gini)
    return np.array(f_thres)[np.argsort(-np.array(gains))][0].tolist()
            
    