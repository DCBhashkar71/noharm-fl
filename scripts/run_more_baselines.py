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
from noharm_fl.certify.bounds import clip_loss, betting_pvalue, calibration_gap_bound_clipped

seed = int(sys.argv[1])
B, ALPHA, C = 5.0, 0.05, 0.05
BLENDS = [0.25, 0.5, 0.75, 1.0]
HEAD = [(0.005, 0.01), (0.01, 0.015), (0.02, 0.02)]
LAM = 1e-5
T_ROUNDS, ETA, INNER = 50, 10.0, 10
FLOORS = [0.0, 0.5]
MC_BINS, MC_SWEEPS, MC_MIN = 10, 3, 200
GROUPS = ["gender", "age_band", "race_grp"]

X, y, h, split, keep = load(seed)
gdf = pd.read_parquet(ROOT / "data/interim/sparcs_2022_adults.parquet", columns=GROUPS).reset_index(drop=True)
gcols = {g: gdf[g].values for g in GROUPS}
Xb = sp.hstack([X, sp.csr_matrix(np.ones((X.shape[0], 1)))]).tocsr()
tr, ce, te = split == "train", split == "cert", split == "test"
idx = np.where(tr)[0]
Xt, yt_ = Xb[idx], y[idx].astype(float)


def weighted_fit(rw, w0, maxiter):
    def f(w):
        z = Xt @ w
        l = np.logaddexp(0, z) - yt_ * z
        loss = np.sum(rw * l) + 0.5 * LAM * (w[:-1] @ w[:-1])
        g = np.asarray(Xt.T @ (rw * (expit(z) - yt_))).ravel()
        g[:-1] += LAM * w[:-1]
        return loss, g
    return minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter}).x


def group_losses(w, gid, nk):
    z = Xt @ w
    return np.bincount(gid, weights=np.logaddexp(0, z) - yt_ * z) / nk


def minimax(gid, floor, w0):
    m = int(gid.max()) + 1
    nk = np.bincount(gid).astype(float)
    mu = np.ones(m) / m
    ws = []
    for t in range(T_ROUNDS):
        lam = floor / m + (1 - floor) * mu
        w = weighted_fit(lam[gid] / nk[gid], w0 if t == 0 else ws[-1], INNER)
        ws.append(w)
        Lg = group_losses(w, gid, nk)
        mu = mu * np.exp(ETA * (Lg - Lg.mean())); mu /= mu.sum()
    return np.mean(ws[T_ROUNDS // 2:], axis=0)


def solve_shift(s, yy):
    ybar = (yy.sum() + expit(s).mean()) / (len(yy) + 1)
    lo, hi = -3.0, 3.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if expit(s + mid).mean() < ybar:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_multicalib(s_tr, gtr, y_tr):
    edges = np.quantile(s_tr, np.linspace(0, 1, MC_BINS + 1)[1:-1])
    shifts, s = [], s_tr.copy()
    for _ in range(MC_SWEEPS):
        for var in GROUPS:
            for lev in np.unique(gtr[var]):
                mg = gtr[var] == lev
                b = np.digitize(s, edges)
                dl = np.zeros(MC_BINS)
                for j in range(MC_BINS):
                    mm = mg & (b == j)
                    n = int(mm.sum())
                    if n >= MC_MIN:
                        dl[j] = solve_shift(s[mm], y_tr[mm]) * n / (n + MC_MIN)
                s = s + np.where(mg, dl[b], 0.0)
                shifts.append((var, lev, dl))
    return edges, shifts


def apply_multicalib(s, g, edges, shifts):
    s = s.copy()
    for var, lev, dl in shifts:
        b = np.digitize(s, edges)
        s = s + np.where(g[var] == lev, dl[b], 0.0)
    return s


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

gid_h = pd.factorize(h[idx])[0]
nk_h = np.bincount(gid_h).astype(float)
gid_d = pd.factorize(pd.Series(gcols["gender"][idx]).astype(str) + "|" + pd.Series(gcols["age_band"][idx]).astype(str) + "|" +
                     pd.Series(gcols["race_grp"][idx]).astype(str))[0]
nk_d = np.bincount(gid_d).astype(float)

W = {}
for fl in FLOORS:
    W[f"afl{fl:g}"] = minimax(gid_h, fl, w_pool)
    print(f"AFL floor {fl:g}: done at {time.time() - t0:.0f}s; worst-hospital TRAINING loss pooled {group_losses(w_pool, gid_h, nk_h).max():.4f} "
          f"-> {group_losses(W[f'afl{fl:g}'], gid_h, nk_h).max():.4f}; mean training loss "
          f"{np.mean(np.logaddexp(0, Xt @ W[f'afl{fl:g}']) - yt_ * (Xt @ W[f'afl{fl:g}'])):.5f}", flush=True)
W["fedminmax"] = minimax(gid_d, 0.0, w_pool)
print(f"FedMinMax objective: done at {time.time() - t0:.0f}s; worst-group TRAINING loss pooled {group_losses(w_pool, gid_d, nk_d).max():.4f} "
      f"-> {group_losses(W['fedminmax'], gid_d, nk_d).max():.4f}", flush=True)
gtr = {g: gcols[g][idx] for g in GROUPS}
edges, shifts = fit_multicalib(pooled.decision_function(X[tr]), gtr, y[idx])
print(f"multicalibration fitted at {time.time() - t0:.0f}s with {sum(int((d != 0).sum()) for _, _, d in shifts)} non-zero cell shifts", flush=True)
SRC = ["pool", "afl0", "afl0.5", "fedminmax", "multicalib"]

rows = []
for n_done, k in enumerate(keep):
    mk = h == k
    a_, c_, t_ = mk & tr, mk & ce, mk & te
    loc = LogisticRegression(C=1.0, max_iter=300).fit(X[a_], y[a_])

    tg = [("overall", None)]                                   # testable groups from TRAINING counts only
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

    PR = {}
    for name, m in (("cert", c_), ("test", t_)):
        d = {"local": loc.predict_proba(X[m])[:, 1], "pool": pooled.predict_proba(X[m])[:, 1]}
        for key in ("afl0", "afl0.5", "fedminmax"):
            d[key] = expit(Xb[m] @ W[key])
        d["multicalib"] = expit(apply_multicalib(pooled.decision_function(X[m]), {g: gcols[g][m] for g in GROUPS}, edges, shifts))
        PR[name] = d

    yc = y[c_].astype(float); ic = idx_of(c_); pl_c = PR["cert"]["local"]; base_c = clip_loss(pl_c, yc, B)
    lossp, calmax = {}, {}
    for src in SRC:
        for j, a in enumerate(BLENDS):
            pc = cand(pl_c, PR["cert"][src], a)
            dl = clip_loss(pc, yc, B) - base_c
            for eps, _ in HEAD:
                lossp[(src, j, eps)] = betting_pvalue(dl, eps, -B, B, ALPHA)
            calmax[(src, j)] = max(calibration_gap_bound_clipped(pc[i], pl_c[i], yc[i], C, ALPHA) for i in ic)

    yt = y[t_].astype(float); it = idx_of(t_); pl_t = PR["test"]["local"]; base_t = clip_loss(pl_t, yt, B)
    zc = stats.norm.ppf(1 - 0.025 / len(tg))

    def evaluate(pt):
        dl = clip_loss(pt, yt, B) - base_t
        gs = [gap_stats(pt[i], pl_t[i], yt[i], zc) for i in it]
        return {"diff": dl.mean(), "se": dl.std(ddof=1) / np.sqrt(len(dl)) if dl.std() > 0 else 0.0,
                "auroc": roc_auc_score(yt, pt) if 0 < yt.sum() < len(yt) else np.nan,
                "gapmax": max(g[0] for g in gs), "gaplb": max(g[1] for g in gs)}

    r = {"hospital": k, "n_train": int(a_.sum()), "deaths_train": int(y[a_].sum()), "n_groups_tested": len(tg)}
    for m_, v in evaluate(pl_t).items():
        r[f"local__{m_}"] = v
    for src in SRC:
        for m_, v in evaluate(PR["test"][src]).items():
            r[f"{src}__{m_}"] = v
        for eps, delta in HEAD:
            sel = -1
            for j in range(len(BLENDS)):
                if lossp[(src, j, eps)] <= ALPHA and calmax[(src, j)] <= delta:
                    sel = j
                else:
                    break
            pt = pl_t if sel < 0 else cand(pl_t, PR["test"][src], BLENDS[sel])
            pre = f"cert_{src}_e{eps}_d{delta}"
            r[f"{pre}__blend"] = 0.0 if sel < 0 else BLENDS[sel]
            for m_, v in evaluate(pt).items():
                r[f"{pre}__{m_}"] = v
    rows.append(r)
    if (n_done + 1) % 20 == 0:
        print(f"  {n_done + 1}/{len(keep)} hospitals, {time.time() - t0:.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(ROOT / f"outputs/tables/more_baselines_seed{seed}.csv", index=False)


def loss_mask(diff, se, eps):
    z = np.where(se > 0, (diff - eps) / np.where(se > 0, se, 1), -np.inf)
    return multipletests(1 - stats.norm.cdf(z), alpha=0.05, method="holm")[0]


out = []
for eps, delta in HEAD:
    for src in SRC:
        for label, pre in [(src, src), (src + " certified", f"cert_{src}_e{eps}_d{delta}")]:
            lm = loss_mask(res[f"{pre}__diff"].values, res[f"{pre}__se"].values, eps)
            cm = res[f"{pre}__gaplb"].values > delta
            nl = int((res[f"{pre}__blend"] > 0).sum()) if "cert" in pre else None
            out.append((eps, delta, label, int(lm.sum()), int(cm.sum()), int((lm | cm).sum()), int((res[f"{pre}__gapmax"] > delta).sum()),
                        nl, res[f"{pre}__diff"].mean(), (res[f"{pre}__auroc"] - res["local__auroc"]).mean()))
print(f"seed {seed}: {len(res)} hospitals")
print(pd.DataFrame(out, columns=["eps", "delta", "method", "loss_harm", "cal_harm_sig", "either_sig", "cal_point_above",
                                 "nonlocal", "mean_loss_diff", "auroc_gain"]).round(5).to_string())
print(f"total {time.time() - t0:.0f}s")