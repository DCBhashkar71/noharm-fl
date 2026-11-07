import sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from noharm_fl.data.features import load, ROOT
from noharm_fl.certify.bounds import clip_loss, betting_pvalue, calibration_gap_bound_clipped

seed = int(sys.argv[1])
B, C = 5.0, 0.05
BLENDS = [0.25, 0.5, 0.75, 1.0]
EPS, DELTA = 0.01, 0.015
GROUPS = ["gender", "age_band", "race_grp"]

X, y, h, split, keep = load(seed)
gdf = pd.read_parquet(ROOT / "data/interim/sparcs_2022_adults.parquet", columns=GROUPS).reset_index(drop=True)
gcols = {g: gdf[g].values for g in GROUPS}
tr, ce, te = split == "train", split == "cert", split == "test"
pooled = LogisticRegression(C=1.0, max_iter=300).fit(X[tr], y[tr])


def cand(pl, ps, a):
    return np.clip(np.clip((1 - a) * pl + a * ps, pl - C, pl + C), 0.0, 1.0)


def eo_gap(p, yy, sex):
    if len(p) < 50:
        return np.nan
    flag = p >= np.quantile(p, 0.90)
    r = []
    for lab in (1, 0):
        t = []
        for sx in ("F", "M"):
            m = (sex == sx) & (yy == lab)
            t.append(flag[m].mean() if m.sum() >= 5 else np.nan)
        r.append(abs(t[0] - t[1]))
    return np.nanmax(r) if not np.all(np.isnan(r)) else np.nan


def select(pl_c, pp_c, yc, ic, alpha):
    base = clip_loss(pl_c, yc, B)
    sel = -1
    for j, a in enumerate(BLENDS):
        pc = cand(pl_c, pp_c, a)
        p_loss = betting_pvalue(clip_loss(pc, yc, B) - base, EPS, -B, B, alpha)
        cal = max(calibration_gap_bound_clipped(pc[i], pl_c[i], yc[i], C, alpha) for i in ic)
        if p_loss <= alpha and cal <= DELTA:
            sel = j
        else:
            break
    return sel


rows = []
nh = len(keep)
for k in keep:
    mk = h == k
    a_, c_, t_ = mk & tr, mk & ce, mk & te
    loc = LogisticRegression(C=1.0, max_iter=300).fit(X[a_], y[a_])
    tg = [("overall", None)]
    for g in GROUPS:
        v = gcols[g][a_]
        for lev in np.unique(v):
            if y[a_][v == lev].sum() >= 20:
                tg.append((g, lev))
    def idx_of(m):
        sub, out = {}, []
        for g, lev in tg:
            if g == "overall":
                out.append(np.arange(int(m.sum())))
            else:
                if g not in sub:
                    sub[g] = gcols[g][m]
                out.append(np.where(sub[g] == lev)[0])
        return out
    pl_c, pp_c = loc.predict_proba(X[c_])[:, 1], pooled.predict_proba(X[c_])[:, 1]
    yc = y[c_].astype(float); ic = idx_of(c_)
    sel1 = select(pl_c, pp_c, yc, ic, 0.05)
    selK = select(pl_c, pp_c, yc, ic, 0.05 / nh)                 # simultaneous guarantee: alpha / number of hospitals
    pl_t, pp_t = loc.predict_proba(X[t_])[:, 1], pooled.predict_proba(X[t_])[:, 1]
    yt = y[t_]; sex = gcols["gender"][t_]
    pc1 = pl_t if sel1 < 0 else cand(pl_t, pp_t, BLENDS[sel1])
    base_t = clip_loss(pl_t, yt.astype(float), B)
    rows.append(dict(hospital=k, sel_alpha005=0.0 if sel1 < 0 else BLENDS[sel1], sel_bonferroni=0.0 if selK < 0 else BLENDS[selK],
                     eo_local=eo_gap(pl_t, yt, sex), eo_pooled=eo_gap(pp_t, yt, sex), eo_certified=eo_gap(pc1, yt, sex),
                     diff_pooled=(clip_loss(pp_t, yt.astype(float), B) - base_t).mean(),
                     diff_certified=(clip_loss(pc1, yt.astype(float), B) - base_t).mean()))
res = pd.DataFrame(rows)
res.to_csv(ROOT / f"outputs/tables/extra_metrics_seed{seed}.csv", index=False)
w = lambda a, b: stats.wilcoxon(a, b).pvalue if np.any(np.asarray(a) != np.asarray(b)) else 1.0
print(f"seed {seed}: hospitals {nh}; non-local at alpha 0.05: {(res.sel_alpha005 > 0).sum()}, at alpha/{nh} (simultaneous): {(res.sel_bonferroni > 0).sum()}")
print(f"equalized-odds gap (mean over hospitals): local {res.eo_local.mean():.4f}, pooled {res.eo_pooled.mean():.4f}, certified {res.eo_certified.mean():.4f}")
print(f"Wilcoxon, pooled vs local loss difference: median {res.diff_pooled.median():+.5f}, p = {stats.wilcoxon(res.diff_pooled).pvalue:.2e}")
print(f"Wilcoxon, certified vs local loss difference: median {res.diff_certified.median():+.6f}, p = {stats.wilcoxon(res.diff_certified).pvalue if (res.diff_certified != 0).any() else 1.0:.2e}")