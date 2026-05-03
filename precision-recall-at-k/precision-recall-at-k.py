def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    recommended_k = recommended[:k]
    num = len(set(recommended_k).intersection(set(relevant)))
    return [num/k, num/len(relevant)]