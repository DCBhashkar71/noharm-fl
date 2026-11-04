import sys, time
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from noharm_fl.data.features import load, ROOT
from noharm_fl.certify.bounds import clip_loss, betting_pvalue, fixed_sequence, calibration_gap_bound_clipped

seed = int(sys.argv[1])
B, ALPHA, C = 5.0, 0.05, 0.05
BLENDS = [0.25, 0.5, 0.75, 1.0]
EPS = [0.005, 0.01, 0.02]
DELTA = [0.01, 0.015, 0.02]
LAM, FT_IT = 1e-5, 5
GROUPS = ["gender", "age_band", "race_grp"]

X, y, h, split, keep = load(seed)
df = pd.read_parquet(ROOT / "data/interim/sparcs_2022_adults.parquet").reset_index(drop=True)
gcols = {g: df[g].values for g in GROUPS}
Xb = sp.hstack([X, sp.csr_matrix(np.ones((X.shape[0], 1)))]).tocsr()
tr, ce, te = split == "train", split == "cert", split == "test"


def local_solve(Xk, yk, w0, maxiter, lam=LAM):
    n = Xk.shape[0]
    def f(w):
        z = Xk @ w
        loss = np.mean(np.logaddexp(0, z) - yk * z) + 0.5 * lam * (w[:-1] @ w[:-1])
        g = np.asarray(Xk.T @ (expit(z) - yk)).ravel() / n
        g[:-1] += lam * w[:-1]
        return loss, g
    return minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter}).x


def min_gap(l0, u0, lb, ub):
    """min of |a+b| - |a| over the box: corners plus where the kink lines meet the box edges."""
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


t0 = time.time()
pooled = LogisticRegression(C=1.0, max_iter=300).fit(X[tr], y[tr])
w_pool = np.append(pooled.coef_[0], pooled.intercept_[0])
print(f"pooled model done in {time.time() - t0:.0f}s", flush=True)

rows = []
for k in keep:
    mk = h == k
    a_, c_, t_ = mk & tr, mk & ce, mk & te
    loc = LogisticRegression(C=1.0, max_iter=300).fit(X[a_], y[a_])
    w_ft = local_solve(Xb[a_], y[a_], w_pool, FT_IT)

    tg = [("overall", None)]                                   # testable groups from TRAINING counts only
    for g in GROUPS:
        v = gcols[g][a_]
        for lev in np.unique(v):
            if y[a_][v == lev].sum() >= 20:
                tg.append((g, lev))

    def group_idx(m):
        sub = {g: gcols[g][m] for g in GROUPS}
        return [np.arange(int(m.sum())) if g == "overall" else np.where(sub[g] == lev)[0] for g, lev in tg]

    PR = {}
    for name, m in (("cert", c_), ("test", t_)):
        PR[name] = {"local": loc.predict_proba(X[m])[:, 1], "pool": pooled.predict_proba(X[m])[:, 1], "ft": expit(Xb[m] @ w_ft)}

    # certification side
    yc = y[c_].astype(float); ic = group_idx(c_); pl_c = PR["cert"]["local"]
    base_c = clip_loss(pl_c, yc, B)
    lossp, calmax = {}, {}
    for src in ("pool", "ft"):
        for j, a in enumerate(BLENDS):
            pc = cand(pl_c, PR["cert"][src], a)
            dl = clip_loss(pc, yc, B) - base_c
            for eps in EPS:
                lossp[(src, j, eps)] = betting_pvalue(dl, eps, -B, B, ALPHA)
            calmax[(src, j)] = max(calibration_gap_bound_clipped(pc[i], pl_c[i], yc[i], C, ALPHA) for i in ic)
    chosen = {}
    for src in ("pool", "ft"):
        for eps in EPS:
            for delta in DELTA:
                sel = -1
                for j in range(len(BLENDS)):
                    if lossp[(src, j, eps)] <= ALPHA and calmax[(src, j)] <= delta:
                        sel = j
                    else:
                        break
                chosen[(src, eps, delta)] = sel

    # test side
    yt = y[t_].astype(float); it = group_idx(t_); pl_t = PR["test"]["local"]
    base_t = clip_loss(pl_t, yt, B)
    zc = stats.norm.ppf(1 - 0.025 / len(tg))

    def evaluate(pt):
        dl = clip_loss(pt, yt, B) - base_t
        gs = [gap_stats(pt[i], pl_t[i], yt[i], zc) for i in it]
        return {"diff": dl.mean(), "se": dl.std(ddof=1) / np.sqrt(len(dl)) if dl.std() > 0 else 0.0,
                "auroc": roc_auc_score(yt, pt) if 0 < yt.sum() < len(yt) else np.nan,
                "gapmax": max(g[0] for g in gs), "gaplb": max(g[1] for g in gs)}

    EV = {"local": evaluate(pl_t), "pool_full": evaluate(PR["test"]["pool"]), "ft_full": evaluate(PR["test"]["ft"])}
    for src in ("pool", "ft"):
        for a in BLENDS:
            EV[f"{src}_b{a}"] = evaluate(cand(pl_t, PR["test"][src], a))

    r = {"hospital": k, "n_train": int(a_.sum()), "deaths_train": int(y[a_].sum()), "n_test": int(t_.sum()),
         "deaths_test": int(yt.sum()), "n_groups_tested": len(tg)}
    for name, e in EV.items():
        for m_, v in e.items():
            r[f"{name}__{m_}"] = v
    for (src, eps, delta), sel in chosen.items():
        pre = f"cert_{src}_e{eps}_d{delta}"
        e = EV["local"] if sel < 0 else EV[f"{src}_b{BLENDS[sel]}"]
        r[f"{pre}__blend"] = 0.0 if sel < 0 else BLENDS[sel]
        for m_, v in e.items():
            r[f"{pre}__{m_}"] = v
    rows.append(r)

res = pd.DataFrame(rows)
res.to_csv(ROOT / f"outputs/tables/cert_full_seed{seed}.csv", index=False)


def loss_mask(diff, se, eps):
    z = np.where(se > 0, (diff - eps) / np.where(se > 0, se, 1), -np.inf)
    return multipletests(1 - stats.norm.cdf(z), alpha=0.05, method="holm")[0]


out = []
for eps, delta in zip(EPS, DELTA):
    for label, pre in [("pooled (unclipped)", "pool_full"), ("fine-tuning (unclipped)", "ft_full"),
                       ("certified pooled", f"cert_pool_e{eps}_d{delta}"), ("certified fine-tuned", f"cert_ft_e{eps}_d{delta}")]:
        lm = loss_mask(res[f"{pre}__diff"].values, res[f"{pre}__se"].values, eps)
        cm = res[f"{pre}__gaplb"].values > delta
        nl = int((res[f"{pre}__blend"] > 0).sum()) if "cert" in pre else None
        out.append((eps, delta, label, int(lm.sum()), int(cm.sum()), int((lm | cm).sum()),
                    int((res[f"{pre}__gapmax"] > delta).sum()), nl, res[f"{pre}__diff"].mean(),
                    (res[f"{pre}__auroc"] - res["local__auroc"]).mean()))
print(f"seed {seed}: {len(res)} hospitals, mean groups tested {res['n_groups_tested'].mean():.1f}")
print(pd.DataFrame(out, columns=["eps", "delta", "method", "loss_harm", "cal_harm_sig", "either_sig", "cal_point_above",
                                 "nonlocal", "mean_loss_diff", "auroc_gain"]).round(5).to_string())
print(f"total {time.time() - t0:.0f}s")