import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here
    x = np.array(x)
    rng = rng if rng else np.random.default_rng() 
    indices = rng.integers(0, x.shape[0], size=(n_bootstrap, x.shape[0]))
    boot_samples = x[indices]
    boot_means = boot_samples.mean(1)
    alpha = 1 - ci

    lower = np.quantile(boot_means, alpha / 2)

    upper = np.quantile(boot_means, 1 - alpha / 2)

    return boot_means, float(lower), float(upper)