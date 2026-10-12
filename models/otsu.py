import numpy as np
import torch

def compute_otsu_threshold(A):
    """Otsu threshold for the attention weights of one bag (Otsu 1979).

    Searches a 256-bin histogram on [0, 1] for the threshold that maximises the
    between-class variance. Used as the default counting threshold for the
    sigmoid and min-max attention activations (``count_positive_instances``).
    Returns 0.5 if no split separates the weights.
    """
    A_flat = A.flatten().cpu().numpy()

    hist, bin_edges = np.histogram(A_flat, bins=256, range=(0,1))
    hist = hist.astype(float) / hist.sum()

    best_threshold = 0.5
    best_variance = 0

    for t_idx in range(len(hist)):
        t = bin_edges[t_idx]
        w0 = hist[:t_idx].sum()
        w1 = hist[t_idx:].sum()

        if w0 == 0 or w1 == 0:
            continue

        mu0 = (A_flat[A_flat <= t]).mean() if (A_flat <= t).sum() > 0 else 0
        mu1 = (A_flat[A_flat > t]).mean() if (A_flat > t).sum() > 0 else 0

        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = t

    return best_threshold