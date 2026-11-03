import sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests
from noharm_fl.data.features import load, ROOT
from noharm_fl.certify.bounds import clip_loss, betting_pvalue, fixed_sequence

seed = int(sys.argv[1])
with_apr = len(sys.argv) > 2 and sys.argv[2] == "apr"
B, ALPHA = 5.0, 0.05
BLENDS = [0.0, 0.25, 0.5, 0.75, 1.0]      # 0 = local
EPS = [0.005, 0.01, 0.02]

X, y, h, split, keep = load(seed, with_apr)
tr, ce, te = split == "train", split == "cert", split == "test"
fed = LogisticRegression(C=1.0, max_iter=300).fit(X[tr], y[tr])

def mse(d):
    return d.mean(), (d.std(ddof=1) / np.sqrt(len(d)) if d.std() > 0 else 0.0)

rows = []
for k in keep:
    mk = h == k
    a_, c_, t_ = mk & tr, mk & ce, mk & te
    loc = LogisticRegression(C=1.0, max_iter=300).fit(X[a_], y[a_])
    P = {}
    for name, m in (("cert", c_), ("test", t_)):
        pl = loc.predict_proba(X[m])[:, 1]; pf = fed.predict_proba(X[m])[:, 1]
        l0 = clip_loss(pl, y[m], B)
        P[name] = [clip_loss((1 - a) * pl + a * pf, y[m], B) - l0 for a in BLENDS]
    r = {"hospital": k, "n_train": int(a_.sum()), "deaths_train": int(y[a_].sum())}
    for j, a in enumerate(BLENDS):
        r[f"diff_b{a}"], r[f"se_b{a}"] = mse(P["test"][j])
    for eps in EPS:
        pv = [betting_pvalue(P["cert"][j], eps, -B, B, ALPHA) for j in range(1, len(BLENDS))]
        j = fixed_sequence(pv, ALPHA) + 1
        r[f"sel_eps{eps}"] = BLENDS[j]
        r[f"seldiff_eps{eps}"], r[f"selse_eps{eps}"] = mse(P["test"][j])
    rows.append(r)

res = pd.DataFrame(rows)
tag = "apr" if with_apr else "noapr"
res.to_csv(ROOT / f"outputs/tables/harm_seed{seed}_{tag}.csv", index=False)

def summarize(diff, se):
    z = np.where(se > 0, diff / np.where(se > 0, se, 1), 0.0)
    worse = multipletests(1 - stats.norm.cdf(z), alpha=0.05, method="holm")[0]
    better = multipletests(stats.norm.cdf(z), alpha=0.05, method="holm")[0]
    return {"mean_diff": diff.mean(), "raw_worse": (diff > 0).mean(), "sig_worse": int(worse.sum()), "sig_better": int(better.sum())}

out = {}
for a in BLENDS[1:]:
    out[f"always blend {a}"] = summarize(res[f"diff_b{a}"].values, res[f"se_b{a}"].values)
for eps in EPS:
    s = summarize(res[f"seldiff_eps{eps}"].values, res[f"selse_eps{eps}"].values)
    s["not_local"] = int((res[f"sel_eps{eps}"] > 0).sum())
    out[f"certified eps={eps}"] = s
print(f"seed {seed} {tag}: {len(res)} hospitals")
print(pd.DataFrame(out).T.round(5))