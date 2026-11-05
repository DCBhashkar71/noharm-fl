import sys, time
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from noharm_fl.data.features import load_temporal, ROOT
from noharm_fl.certify.bounds import clip_loss, betting_pvalue, fixed_sequence, calibration_gap_bound_clipped

seed = int(sys.argv[1])
year = sys.argv[2] if len(sys.argv) > 2 else "2023"
B, ALPHA, C = 5.0, 0.05, 0.05
BLENDS = [0.25, 0.5, 0.75, 1.0]
EPS = [0.005, 0.01, 0.02]
DELTA = [0.01, 0.015, 0.02]
HEAD = list(zip(EPS, DELTA))
LAM, FT_IT, MIN_N = 1e-5, 5, 200
GROUPS = ["gender", "age_band", "race_grp"]

X, y, h, split, keep, X2, y2, h2, df2 = load_temporal(seed, year)
df = pd.read_parquet(ROOT / "data/interim/sparcs_2022_adults.parquet").reset_index(drop=True)
g1 = {g: df[g].values for g in GROUPS}
g2 = {g: df2[g].values for g in GROUPS}
Xb = sp.hstack([X, sp.csr_matrix(np.ones((X.shape[0], 1)))]).tocsr()
Xb2 = sp.hstack([X2, sp.csr_matrix(np.ones((X2.shape[0], 1)))]).tocsr()
tr, ce = split == "train", split == "cert"


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

rows, skipped = [], 0
for k in keep:
    mk = h == k
    t2 = h2 == k
    if t2.sum() < MIN_N:
        skipped += 1
        continue
    a_, c_ = mk & tr, mk & ce
    loc = LogisticRegression(C=1.0, max_iter=300).fit(X[a_], y[a_])
    w_ft = local_solve(Xb[a_], y[a_], w_pool, FT_IT)

    tg = [("overall", None)]                       # testable groups from 2022 TRAINING counts only
    for g in GROUPS:
        v = g1[g][a_]
        for lev in np.unique(v):
            if y[a_][v == lev].sum() >= 20:
                tg.append((g, lev))

    def idx_of(src, m, glist, n):
        out = []
        for g, lev in glist:
            out.append(np.arange(n) if g == "overall" else np.where(src[g][m] == lev)[0])
        return out

    # certification (2022 certification split, unchanged)
    yc = y[c_].astype(float); pl_c = loc.predict_proba(X[c_])[:, 1]
    ps = {"pool": pooled.predict_proba(X[c_])[:, 1], "ft": expit(Xb[c_] @ w_ft)}
    ic = idx_of(g1, c_, tg, int(c_.sum()))
    base_c = clip_loss(pl_c, yc, B)
    lossp, calmax = {}, {}
    for src in ("pool", "ft"):
        for j, a in enumerate(BLENDS):
            pc = cand(pl_c, ps[src], a)
            dl = clip_loss(pc, yc, B) - base_c
            for eps in EPS:
                lossp[(src, j, eps)] = betting_pvalue(dl, eps, -B, B, ALPHA)
            calmax[(src, j)] = max(calibration_gap_bound_clipped(pc[i], pl_c[i], yc[i], C, ALPHA) for i in ic)
    chosen = {}
    for src in ("pool", "ft"):
        for eps, delta in HEAD:
            sel = -1
            for j in range(len(BLENDS)):
                if lossp[(src, j, eps)] <= ALPHA and calmax[(src, j)] <= delta:
                    sel = j
                else:
                    break
            chosen[(src, eps, delta)] = sel

    # test on the other year
    yt = y2[t2].astype(float); pl_t = loc.predict_proba(X2[t2])[:, 1]
    pt_ = {"pool": pooled.predict_proba(X2[t2])[:, 1], "ft": expit(Xb2[t2] @ w_ft)}
    it = idx_of(g2, t2, tg, int(t2.sum()))
    base_t = clip_loss(pl_t, yt, B)
    zc = stats.norm.ppf(1 - 0.025 / len(tg))

    def evaluate(pt):
        dl = clip_loss(pt, yt, B) - base_t
        gs = [gap_stats(pt[i], pl_t[i], yt[i], zc) for i in it]
        return {"diff": dl.mean(), "se": dl.std(ddof=1) / np.sqrt(len(dl)) if dl.std() > 0 else 0.0,
                "auroc": roc_auc_score(yt, pt) if 0 < yt.sum() < len(yt) else np.nan,
                "gapmax": max(g[0] for g in gs), "gaplb": max(g[1] for g in gs)}

    EV = {"local": evaluate(pl_t), "pool_full": evaluate(pt_["pool"]), "ft_full": evaluate(pt_["ft"])}
    for src in ("pool", "ft"):
        for a in BLENDS:
            EV[f"{src}_b{a}"] = evaluate(cand(pl_t, pt_[src], a))

    r = {"hospital": k, "n_train": int(a_.sum()), "n_test_year": int(t2.sum()), "deaths_test_year": int(yt.sum()),
         "rate_2022": float(y[mk].mean()), "rate_other_year": float(yt.mean()), "n_groups_tested": len(tg)}
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
res.to_csv(ROOT / f"outputs/tables/temporal_{year}_seed{seed}.csv", index=False)


def loss_mask(diff, se, eps):
    z = np.where(se > 0, (diff - eps) / np.where(se > 0, se, 1), -np.inf)
    return multipletests(1 - stats.norm.cdf(z), alpha=0.05, method="holm")[0]


out = []
for eps, delta in HEAD:
    for label, pre in [("pooled (unclipped)", "pool_full"), ("fine-tuning (unclipped)", "ft_full"),
                       ("certified pooled", f"cert_pool_e{eps}_d{delta}"), ("certified fine-tuned", f"cert_ft_e{eps}_d{delta}")]:
        lm = loss_mask(res[f"{pre}__diff"].values, res[f"{pre}__se"].values, eps)
        cm = res[f"{pre}__gaplb"].values > delta
        nl = int((res[f"{pre}__blend"] > 0).sum()) if "cert" in pre else None
        out.append((eps, delta, label, int(lm.sum()), int(cm.sum()), int((lm | cm).sum()),
                    int((res[f"{pre}__gapmax"] > delta).sum()), nl, res[f"{pre}__diff"].mean(),
                    (res[f"{pre}__auroc"] - res["local__auroc"]).mean()))
print(f"seed {seed}, test year {year}: {len(res)} hospitals ({skipped} skipped, < {MIN_N} adult discharges)")
print(f"mean death rate 2022 {res['rate_2022'].mean():.4f}, {year} {res['rate_other_year'].mean():.4f}")
print(pd.DataFrame(out, columns=["eps", "delta", "method", "loss_harm", "cal_harm_sig", "either_sig", "cal_point_above",
                                 "nonlocal", "mean_loss_diff", "auroc_gain"]).round(5).to_string())
print(f"total {time.time() - t0:.0f}s")