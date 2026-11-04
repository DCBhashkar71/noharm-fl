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


def eb_lower(d, lo, hi, alpha=0.05):
    """Empirical-Bernstein lower bound on E[d] (one-sided level alpha)."""
    return -eb_upper(-np.asarray(d, float), -hi, -lo, alpha)


def worst_case_cal_gap(l0, u0, lb, ub):
    """max of |a+b| - |a| over the box a in [l0,u0], b in [lb,ub]. The function is piecewise linear,
    so the maximum is at a box corner or where a=0 or a+b=0 meets a box edge."""
    cands = [(a, b) for a in (l0, u0) for b in (lb, ub)]
    if l0 <= 0 <= u0:
        cands += [(0.0, lb), (0.0, ub)]
    for b in (lb, ub):
        if l0 <= -b <= u0:
            cands.append((-b, b))
    return max(abs(a + b) - abs(a) for a, b in cands)


def calibration_gap_bound(p_m, p_0, y, alpha=0.05):
    """Upper bound on |e_m| - |e_0| for one group (e = mean prediction minus outcome), with probability >= 1 - alpha.
    Four one-sided EB bounds at alpha/4 each: two for e_0, two for the paired difference D = mean(p_m - p_0)."""
    a = alpha / 4
    e0 = np.asarray(p_0, float) - np.asarray(y, float)
    dd = np.asarray(p_m, float) - np.asarray(p_0, float)
    return worst_case_cal_gap(eb_lower(e0, -1, 1, a), eb_upper(e0, -1, 1, a),
                              eb_lower(dd, -1, 1, a), eb_upper(dd, -1, 1, a))


def calibration_gap_bound_clipped(p_m, p_0, y, c, alpha=0.05):
    """Same as calibration_gap_bound, for the candidate clipped to p_0 +/- c, so the paired difference lies in [-c, c]."""
    a = alpha / 4
    p_m, p_0, y = np.asarray(p_m, float), np.asarray(p_0, float), np.asarray(y, float)
    dd = np.clip(p_m, p_0 - c, p_0 + c) - p_0
    e0 = p_0 - y
    return worst_case_cal_gap(eb_lower(e0, -1, 1, a), eb_upper(e0, -1, 1, a),
                              eb_lower(dd, -c, c, a), eb_upper(dd, -c, c, a))