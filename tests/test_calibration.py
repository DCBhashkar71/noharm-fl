import numpy as np
from noharm_fl.certify.bounds import worst_case_cal_gap, calibration_gap_bound, calibration_gap_bound_clipped

def test_worst_case_known_values():
    assert abs(worst_case_cal_gap(0.01, 0.01, 0.02, 0.02) - 0.02) < 1e-12      # moves away from zero
    assert abs(worst_case_cal_gap(0.05, 0.05, -0.03, -0.03) - (-0.03)) < 1e-12  # improves calibration
    assert worst_case_cal_gap(-0.01, 0.01, 0.0, 0.0) <= 1e-12                   # no change

def test_floor_for_equal_models():
    rng = np.random.default_rng(0)
    p = np.clip(rng.normal(0.05, 0.02, 3000), 0.001, 0.9)
    y = (rng.random(3000) < p).astype(float)
    full = calibration_gap_bound(p, p, y)
    clipped = calibration_gap_bound_clipped(p, p, y, 0.05)
    print("floor, range [-1,1]:", full, " floor, clipped c=0.05:", clipped)
    assert 0 <= clipped < full < 0.02

def test_clipped_bound_is_valid_when_truth_is_exactly_delta():
    # candidate shifts every prediction by +delta on top of a calibrated local model: true gap is exactly delta
    rng = np.random.default_rng(1)
    delta, fails, reps = 0.01, 0, 200
    for _ in range(reps):
        p0 = np.clip(rng.normal(0.05, 0.02, 3000), 0.01, 0.9)
        y = (rng.random(3000) < p0).astype(float)
        fails += calibration_gap_bound_clipped(p0 + delta, p0, y, 0.05) <= delta
    assert fails / reps <= 0.05