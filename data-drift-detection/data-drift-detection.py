import numpy as np

def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    reference_counts = np.asarray(reference_counts)
    production_counts = np.asarray(production_counts)
    tvd = 1/2*np.sum(np.abs(reference_counts/reference_counts.sum() - production_counts/production_counts.sum())).item()
    return {"score": tvd, "drift_detected": tvd>threshold}