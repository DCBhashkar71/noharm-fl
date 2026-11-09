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
T_ROUNDS, INNER = 50, 10
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


def hosp_losses(w, gid, nk):
    z = Xt @ w
    return np.bincount(gid, weights=np.logaddexp(0, z) - yt_ * z, minlength=len(nk)) / nk


def maxfl(variant, rho, isk, w0):
    ws = []
    for t in range(T_ROUNDS):
        w_prev = w0 if t == 0 else ws[-1]
        F = hosp_losses(w_prev, gid, nk)
        s = expit(F - np.where(isk, rho, F))               # equals 0.5 for hospitals without a rho
        q = s if variant == "alg" else s * (1 - s)         # alg: sigma (Algorithm 1); grad: sigma*(1-sigma) (objective's gradient)
        rw = (q / nk)[gid]
        ws.append(weighted_fit(rw / q.sum(), w_prev, INNER))
    return np.mean(ws[T_ROUNDS // 2:], axis=0)


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
gid, uniq = pd.factorize(h[idx])
nk = np.bincount(gid).astype(float)

# rho_k = training loss of hospital k's own local model (the paper's rho_k = F_k of a solo-trained model)
isk = np.isin(uniq, keep)
rho = np.zeros(len(uniq))
LOC = {}
for j in np.where(isk)[0]:
    mk = (h == uniq[j]) & tr
    LOC[uniq[j]] = LogisticRegression(C=1.0, max_iter=300).fit(X[mk], y[mk])
    p = np.clip(LOC[uniq[j]].predict_proba(X[mk])[:, 1], 1e-12, 1 - 1e-12)
    yy = y[mk].astype(float)
    rho[j] = -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))
print(f"rho computed at {time.time() - t0:.0f}s; mean rho {rho[isk].mean():.4f}; pooled mean loss at those hospitals "
      f"{hosp_losses(w_pool, gid, nk)[isk].mean():.4f}", flush=True)

W = {}
for var in ("alg", "grad"):
    W[f"maxfl_{var}"] = maxfl(var, rho, isk, w_pool)
    F = hosp_losses(W[f"maxfl_{var}"], gid, nk)
    print(f"MaxFL-{var}: done at {time.time() - t0:.0f}s; hospitals with F_k < rho_k on training: "
          f"{int((F[isk] < rho[isk]).sum())} of {int(isk.sum())} (pooled: {int((hosp_losses(w_pool, gid, nk)[isk] < rho[isk]).sum())})", flush=True)
SRC = ["pool", "maxfl_alg", "maxfl_grad"]

rows = []
for n_done, k in enumerate(keep):
    mk = h == k
    a_, c_, t_ = mk & tr, mk & ce, mk & te
    loc = LOC[k]

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

    PR = {}
    for name, m in (("cert", c_), ("test", t_)):
        d = {"local": loc.predict_proba(X[m])[:, 1], "pool": pooled.predict_proba(X[m])[:, 1]}
        for key in ("maxfl_alg", "maxfl_grad"):
            d[key] = expit(Xb[m] @ W[key])
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
res.to_csv(ROOT / f"outputs/tables/maxfl_seed{seed}.csv", index=False)


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
            out.append((eps, delta, label, int(lm.sum()), int(cm.sum()), int((lm | cm).sum()), nl,
                        res[f"{pre}__diff"].mean(), (res[f"{pre}__auroc"] - res["local__auroc"]).mean()))
print(f"seed {seed}: {len(res)} hospitals")
print(pd.DataFrame(out, columns=["eps", "delta", "method", "loss_harm", "cal_harm_sig", "either_sig", "nonlocal",
                                 "mean_loss_diff", "auroc_gain"]).round(5).to_string())
print("hospitals where the model beats local on test loss (GM-APPEAL numerator): " +
      ", ".join(f"{s} {int((res[f'{s}__diff'] < 0).sum())}" for s in SRC))
print(f"total {time.time() - t0:.0f}s")