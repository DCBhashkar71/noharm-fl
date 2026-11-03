import numpy as np


def clip_loss(p, y, B=5.0):
    """Log loss clipped to [0, B]."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return np.minimum(l, B)


def eb_upper(d, lo, hi, alpha=0.05):
    """Maurer-Pontil empirical-Bernstein upper bound on E[d], d in [lo, hi], one-sided level alpha."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return hi
    z = (d - lo) / (hi - lo)
    L = np.log(2 / alpha)
    ub = z.mean() + np.sqrt(2 * z.var(ddof=1) * L / n) + 7 * L / (3 * (n - 1))
    return lo + (hi - lo) * min(ub, 1.0)


def eb_pvalue(d, eps, lo, hi, alpha=0.05):
    """Certifies (returns 0.0) if the EB upper bound is <= eps, else 1.0. A level-alpha test, not a graded p-value."""
    return 0.0 if eb_upper(d, lo, hi, alpha) <= eps else 1.0


def betting_pvalue(d, eps, lo, hi, alpha=0.05, c=0.5):
    """Betting p-value for H0: E[d] >= eps (harm), d in [lo, hi]. Small p => certify E[d] < eps."""
    d = np.asarray(d, float)
    n = len(d)
    z = (d - lo) / (hi - lo)
    m = (eps - lo) / (hi - lo)
    if m >= 1:
        return 0.0
    if m <= 0:
        return 1.0
    idx = np.arange(1, n + 1)
    mu_hat = (0.5 + np.cumsum(z)) / (idx + 1)
    var_hat = (0.25 + np.cumsum((z - mu_hat) ** 2)) / (idx + 1)
    var_prev = np.concatenate([[0.25], var_hat[:-1]])
    lam = np.sqrt(2 * np.log(1 / alpha) / (var_prev * idx * np.log(1 + idx)))
    lam = np.minimum(lam, c / (1 - m))
    logK = np.cumsum(np.log1p(lam * (m - z)))
    return float(min(1.0, np.exp(-logK.max())))


def fixed_sequence(pvals, alpha=0.05):
    """Candidates ordered most local to most federated. Returns index of the last candidate that passed
    before the first failure, or -1 (stay local)."""
    sel = -1
    for j, p in enumerate(pvals):
        if p <= alpha:
            sel = j
        else:
            break
    return sel