import numpy as np
from noharm_fl.certify.bounds import eb_pvalue, betting_pvalue

def rate(fn, mean_shift, reps=1000, n=2000, eps=0.001, lo=-1.0, hi=1.0, seed=0):
    rng = np.random.default_rng(seed)
    out = 0
    for _ in range(reps):
        x = (rng.random(n) < 0.03) * 0.5          # skewed: rare large values, true mean 0.015
        d = x - 0.015 + eps + mean_shift           # true mean of d = eps + mean_shift
        out += fn(d, eps, lo, hi) <= 0.05
    return out / reps

def test_validity_at_boundary():
    for fn in (eb_pvalue, betting_pvalue):
        r = rate(fn, 0.0)
        print(fn.__name__, "false certification at the boundary:", r)
        assert r <= 0.05

def test_power_report():
    for fn in (eb_pvalue, betting_pvalue):
        for s in (-0.005, -0.01, -0.02):
            print(fn.__name__, "true mean below eps by", -s, "-> certified", rate(fn, s))