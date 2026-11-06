import sys, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from noharm_fl.data.features import load, ROOT
from noharm_fl.certify.bounds import clip_loss, betting_pvalue, calibration_gap_bound_clipped

warnings.filterwarnings("ignore")
model, seed = sys.argv[1], int(sys.argv[2])
assert model in ("mlp", "gbm")
B, ALPHA, C = 5.0, 0.05, 0.05
BLENDS = [0.25, 0.5, 0.75, 1.0]
EPS = [0.005, 0.01, 0.02]
DELTA = [0.01, 0.015, 0.02]
HEAD = list(zip(EPS, DELTA))
GROUPS = ["gender", "age_band", "race_grp"]

X, y, h, split, keep = load(seed)
gdf = pd.read_parquet(ROOT / "data/interim/sparcs_2022_adults.parquet", columns=GROUPS).reset_index(drop=True)
gcols = {g: gdf[g].values for g in GROUPS}
tr, ce, te = split == "train", split == "cert", split == "test"


def min_gap(l0, u0, lb, ub):
    pts = [(a, b) for a in (l0, u0) for b in (lb, ub)]
    if l0 <= 0 <= u0:
        pts += [(0.0, lb), (0.0, ub)]
    pts += [(-b, b) for b in (lb, ub) if l0 <= -b <= u0]
    pts += [(a, -a) for a in (l0, u0) if lb <= -a <= ub]
    return min(abs(a + b) - abs(a) for a, b in pts)


def gap_stats(pt, pl, yy, z):
    if len(pl) < 2:
        return 0.0, -1.0
    e0, dd = pl - yy, pt - pl
    point = abs(e0.mean() + dd.mean()) - abs(e0.mean())
    m0, s0 = e0.mean(), e0.std(ddof=1) / np.sqrt(len(e0))
    mb, sb = dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd))
    return point, min_gap(m0 - z * s0, m0 + z * s0, mb - z * sb, mb + z * sb)


def cand(pl, ps, a):
    return np.clip(np.clip((1 - a) * pl + a * ps, pl - C, pl + C), 0.0, 1.0)


def mlp(batch):
    return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-4, batch_size=batch, max_iter=30, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=3, random_state=seed)


def fit_local(Xk, yk):
    if model == "mlp":
        return mlp(256).fit(Xk, yk)
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=3, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed).fit(Xk.toarray(), yk)


def proba(m, Xk):
    return m.predict_proba(Xk.toarray() if model == "gbm" else Xk)[:, 1]


t0 = time.time()
pooled = mlp(2048).fit(X[tr], y[tr]) if model == "mlp" else LogisticRegression(C=1.0, max_iter=300).fit(X[tr], y[tr])
print(f"{model}: pooled model done in {time.time() - t0:.0f}s", flush=True)

rows = []
for n_done, k in enumerate(keep):
    mk = h == k
    a_, c_, t_ = mk & tr, mk & ce, mk & te
    loc = fit_local(X[a_], y[a_])
    lr = LogisticRegression(C=1.0, max_iter=300).fit(X[a_], y[a_])

    tg = [("overall", None)]                                   # testable groups from TRAINING counts only
    for g in GROUPS:
        v = gcols[g][a_]
        for lev in np.unique(v):
            if y[a_][v == lev].sum() >= 20:
                tg.append((g, lev))

    def idx_of(m, glist):
        sub, out = {}, []
        for g, lev in glist:
            if g == "overall":
                out.append(np.arange(int(m.sum())))
            else:
                if g not in sub:
                    sub[g] = gcols[g][m]
                out.append(np.where(sub[g] == lev)[0])
        return out

    pl_c, pp_c = proba(loc, X[c_]), pooled.predict_proba(X[c_])[:, 1]
    pl_t, pp_t = proba(loc, X[t_]), pooled.predict_proba(X[t_])[:, 1]
    lr_t = lr.predict_proba(X[t_])[:, 1]

    yc = y[c_].astype(float); ic = idx_of(c_, tg); base_c = clip_loss(pl_c, yc, B)
    lossp, calmax = {}, {}
    for j, a in enumerate(BLENDS):
        pc = cand(pl_c, pp_c, a)
        dl = clip_loss(pc, yc, B) - base_c
        for eps in EPS:
            lossp[(j, eps)] = betting_pvalue(dl, eps, -B, B, ALPHA)
        calmax[j] = max(calibration_gap_bound_clipped(pc[i], pl_c[i], yc[i], C, ALPHA) for i in ic)
    chosen = {}
    for eps, delta in HEAD:
        sel = -1
        for j in range(len(BLENDS)):
            if lossp[(j, eps)] <= ALPHA and calmax[j] <= delta:
                sel = j
            else:
                break
        chosen[(eps, delta)] = sel

    yt = y[t_].astype(float); it = idx_of(t_, tg); base_t = clip_loss(pl_t, yt, B)
    zc = stats.norm.ppf(1 - 0.025 / len(tg))

    def evaluate(pt):
        dl = clip_loss(pt, yt, B) - base_t
        gs = [gap_stats(pt[i], pl_t[i], yt[i], zc) for i in it]
        return {"diff": dl.mean(), "se": dl.std(ddof=1) / np.sqrt(len(dl)) if dl.std() > 0 else 0.0,
                "auroc": roc_auc_score(yt, pt) if 0 < yt.sum() < len(yt) else np.nan,
                "gapmax": max(g[0] for g in gs), "gaplb": max(g[1] for g in gs)}

    EV = {"local": evaluate(pl_t), "pool_full": evaluate(pp_t)}
    for a in BLENDS:
        EV[f"b{a}"] = evaluate(cand(pl_t, pp_t, a))

    r = {"hospital": k, "n_train": int(a_.sum()), "deaths_train": int(y[a_].sum()), "n_test": int(t_.sum()),
         "deaths_test": int(yt.sum()), "n_groups_tested": len(tg),
         "lr_local_diff": (clip_loss(lr_t, yt, B) - base_t).mean(),
         "lr_local_auroc": roc_auc_score(yt, lr_t) if 0 < yt.sum() < len(yt) else np.nan}
    for name, e in EV.items():
        for m_, v in e.items():
            r[f"{name}__{m_}"] = v
    for (eps, delta), sel in chosen.items():
        pre = f"cert_e{eps}_d{delta}"
        e = EV["local"] if sel < 0 else EV[f"b{BLENDS[sel]}"]
        r[f"{pre}__blend"] = 0.0 if sel < 0 else BLENDS[sel]
        for m_, v in e.items():
            r[f"{pre}__{m_}"] = v
    rows.append(r)
    if (n_done + 1) % 20 == 0:
        print(f"  {n_done + 1}/{len(keep)} hospitals, {time.time() - t0:.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(ROOT / f"outputs/tables/model_{model}_seed{seed}.csv", index=False)


def loss_mask(diff, se, eps):
    z = np.where(se > 0, (diff - eps) / np.where(se > 0, se, 1), -np.inf)
    return multipletests(1 - stats.norm.cdf(z), alpha=0.05, method="holm")[0]


out = []
for eps, delta in HEAD:
    for label, pre in [("pooled (unclipped)", "pool_full"), ("certified pooled", f"cert_e{eps}_d{delta}")]:
        lm = loss_mask(res[f"{pre}__diff"].values, res[f"{pre}__se"].values, eps)
        cm = res[f"{pre}__gaplb"].values > delta
        nl = int((res[f"{pre}__blend"] > 0).sum()) if "cert" in pre else None
        out.append((eps, delta, label, int(lm.sum()), int(cm.sum()), int((lm | cm).sum()), int((res[f"{pre}__gapmax"] > delta).sum()),
                    nl, res[f"{pre}__diff"].mean(), (res[f"{pre}__auroc"] - res["local__auroc"]).mean()))
print(f"seed {seed}, model class {model}: {len(res)} hospitals")
print(f"mean test AUROC: local {model} {res['local__auroc'].mean():.4f}, local logistic {res['lr_local_auroc'].mean():.4f}; "
      f"mean loss of local logistic minus local {model}: {res['lr_local_diff'].mean():+.5f}")
print(pd.DataFrame(out, columns=["eps", "delta", "method", "loss_harm", "cal_harm_sig", "either_sig", "cal_point_above",
                                 "nonlocal", "mean_loss_diff", "auroc_gain"]).round(5).to_string())
print(f"total {time.time() - t0:.0f}s")