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
opts = sys.argv[2:]
with_apr = "apr" in opts
cut = 100
for a in opts:
    if a.startswith("cut"):
        cut = int(a[3:])
tag = ("_apr" if with_apr else "") + (f"_cut{cut}" if cut != 100 else "")
B, ALPHA, C = 5.0, 0.05, 0.05
BLENDS = [0.25, 0.5, 0.75, 1.0]
EPS = [0.005, 0.01, 0.02]
DELTA = [0.01, 0.015, 0.02]
HEAD = list(zip(EPS, DELTA))              # headline (eps, delta) pairs
LAM, FT_IT = 1e-5, 5
GROUPS = ["gender", "age_band", "race_grp"]

X, y, h, split, keep = load(seed, with_apr, cut)
df = pd.read_parquet(ROOT / "data/interim/sparcs_2022_adults.parquet").reset_index(drop=True)
gcols = {g: df[g].values for g in GROUPS}
gcols["race_x_age"] = (df["race_grp"].astype(str) + "|" + df["age_band"].astype(str)).values
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


def bin_rows(k, grp, lev, mname, p, yy, nb):
    if len(p) < 10 * nb:
        return []
    try:
        g = pd.DataFrame({"p": p, "y": yy, "b": pd.qcut(p, nb, duplicates="drop")}).groupby("b", observed=True).agg(
            n=("y", "size"), mean_p=("p", "mean"), mean_y=("y", "mean"))
        return [(seed, k, grp, lev, mname, i, r.n, r.mean_p, r.mean_y) for i, r in enumerate(g.itertuples())]
    except Exception:
        return []


t0 = time.time()
print(f"seed {seed}, features {'WITH' if with_apr else 'without'} APR fields, minimum deaths {cut}", flush=True)
pooled = LogisticRegression(C=1.0, max_iter=300).fit(X[tr], y[tr])
w_pool = np.append(pooled.coef_[0], pooled.intercept_[0])
print(f"pooled model done in {time.time() - t0:.0f}s", flush=True)

META = {"local": (np.nan, np.nan), "pool_full": (np.nan, np.nan), "ft_full": (np.nan, np.nan)}
for src in ("pool", "ft"):
    for eps, delta in HEAD:
        META[f"cert_{src}_e{eps}_d{delta}"] = (eps, delta)

rows, grows, bins = [], [], []
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
    ig = []                                                    # intersections: descriptive only
    vi = gcols["race_x_age"][a_]
    for lev in np.unique(vi):
        if y[a_][vi == lev].sum() >= 20:
            ig.append(("race_x_age", lev))

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

    PR = {}
    for name, m in (("cert", c_), ("test", t_)):
        PR[name] = {"local": loc.predict_proba(X[m])[:, 1], "pool": pooled.predict_proba(X[m])[:, 1], "ft": expit(Xb[m] @ w_ft)}

    # certification side
    yc = y[c_].astype(float); ic = idx_of(c_, tg); pl_c = PR["cert"]["local"]
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
    yt = y[t_].astype(float); it = idx_of(t_, tg); pl_t = PR["test"]["local"]
    base_t = clip_loss(pl_t, yt, B)
    zc = stats.norm.ppf(1 - 0.025 / len(tg))

    def evaluate(pt):
        dl = clip_loss(pt, yt, B) - base_t
        gs = [gap_stats(pt[i], pl_t[i], yt[i], zc) for i in it]
        return {"diff": dl.mean(), "se": dl.std(ddof=1) / np.sqrt(len(dl)) if dl.std() > 0 else 0.0,
                "auroc": roc_auc_score(yt, pt) if 0 < yt.sum() < len(yt) else np.nan,
                "gapmax": max(g[0] for g in gs), "gaplb": max(g[1] for g in gs)}

    PT = {"local": pl_t, "pool_full": PR["test"]["pool"], "ft_full": PR["test"]["ft"]}
    EV = {"local": evaluate(pl_t), "pool_full": evaluate(PR["test"]["pool"]), "ft_full": evaluate(PR["test"]["ft"])}
    for src in ("pool", "ft"):
        for a in BLENDS:
            pb = cand(pl_t, PR["test"][src], a)
            EV[f"{src}_b{a}"] = evaluate(pb)
            PT[f"{src}_b{a}"] = pb
    for src in ("pool", "ft"):
        for eps, delta in HEAD:
            sel = chosen[(src, eps, delta)]
            PT[f"cert_{src}_e{eps}_d{delta}"] = pl_t if sel < 0 else PT[f"{src}_b{BLENDS[sel]}"]

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

    # per-group rows (marginal groups tested by the certificate; intersections descriptive only)
    zi = stats.norm.ppf(1 - 0.025 / max(len(ig), 1))
    for kind, glist, zz in (("marginal", tg, zc), ("intersect", ig, zi)):
        ii = it if kind == "marginal" else idx_of(t_, ig)
        for (g, lev), ix in zip(glist, ii):
            if len(ix) < 2:
                continue
            for mname, meta in META.items():
                pt = PT[mname]
                point, lb = gap_stats(pt[ix], pl_t[ix], yt[ix], zz)
                grows.append((seed, k, kind, g, lev, len(ix), int(yt[ix].sum()), mname, meta[0], meta[1],
                              (pl_t[ix] - yt[ix]).mean(), (pt[ix] - yt[ix]).mean(), point, lb))

    # calibration bins for reliability curves
    for mname in ("local", "pool_full", "cert_pool_e0.01_d0.015", "cert_ft_e0.01_d0.015"):
        bins += bin_rows(k, "overall", "all", mname, PT[mname], yt, 10)
        for (g, lev), ix in zip(tg[1:], it[1:]):
            bins += bin_rows(k, g, lev, mname, PT[mname][ix], yt[ix], 5)

res = pd.DataFrame(rows)
res.to_csv(ROOT / f"outputs/tables/cert_full_seed{seed}{tag}.csv", index=False)
pd.DataFrame(grows, columns=["seed", "hospital", "kind", "var", "level", "n_test", "deaths_test", "method", "eps", "delta",
                             "e0", "em", "gap_point", "gap_lb"]).to_csv(ROOT / f"outputs/tables/cert_groups_seed{seed}{tag}.csv", index=False)
pd.DataFrame(bins, columns=["seed", "hospital", "var", "level", "method", "bin", "n", "mean_p", "mean_y"]).to_csv(
    ROOT / f"outputs/tables/cert_bins_seed{seed}{tag}.csv", index=False)


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
print(f"seed {seed}{tag}: {len(res)} hospitals, mean groups tested {res['n_groups_tested'].mean():.1f}, "
      f"local mean AUROC {res['local__auroc'].mean():.4f}")
print(pd.DataFrame(out, columns=["eps", "delta", "method", "loss_harm", "cal_harm_sig", "either_sig", "cal_point_above",
                                 "nonlocal", "mean_loss_diff", "auroc_gain"]).round(5).to_string())
print(f"total {time.time() - t0:.0f}s")