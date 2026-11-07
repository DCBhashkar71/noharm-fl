import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from noharm_fl.data.features import load, ROOT

seed = int(sys.argv[1])
X, y, h, split, keep = load(seed)
tr = split == "train"
pooled = LogisticRegression(C=1.0, max_iter=300).fit(X[tr], y[tr])
wp = np.append(pooled.coef_[0], pooled.intercept_[0])
d, N, Nd = X.shape[1] + 1, int(tr.sum()), int(y[tr].sum())
rows = []
for k in keep:
    a_ = (h == k) & tr
    loc = LogisticRegression(C=1.0, max_iter=300).fit(X[a_], y[a_])
    wl = np.append(loc.coef_[0], loc.intercept_[0])
    rows.append((k, seed, int(a_.sum()), int(y[a_].sum()), d, N, Nd, float(np.linalg.norm(wl - wp))))
pd.DataFrame(rows, columns=["hospital", "seed", "n_train", "deaths_train", "d", "N", "N_deaths", "dist"]).to_csv(
    ROOT / f"outputs/tables/participation_seed{seed}.csv", index=False)
print(f"seed {seed}: {len(rows)} hospitals, d = {d}, N = {N}, median distance {np.median([r[-1] for r in rows]):.3f}")