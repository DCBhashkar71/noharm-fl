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

# Amendment 21: test of C3 (constrained primal-dual training) against pooled training + certificate
seed = int(sys.argv[1])
B, ALPHA, C = 5.0, 0.05, 0.05
BLENDS = [0.25, 0.5, 0.75, 1.0]
HEAD = [(0.005, 0.01), (0.01, 0.015), (0.02, 0.02)]
LAM = 1e-5
T_ROUNDS, T0, ETA, INNER = 50, 10, 1.0, 10
RHOS = [0.01, 0.05]
CON_FRAC = 0.2
GROUPS = ["gender", "age_band", "race_grp"]

X, y, h, split, keep = load(seed)
gdf = pd.read_parquet(ROOT / "data/interim/sparcs_2022_adults.parquet", columns=GROUPS).reset_index(drop=True)
gcols = {g: gdf[g].values for g in GROUPS}
Xb = sp.hstack([X, sp.csr_matrix(np.ones((X.shape[0], 1)))]).tocsr()
tr, ce, te = split == "train", split == "cert", split == "test"
idx = np.where(tr)[0]

# constraint split: 20% of training rows, never used by certification or test
rng = np.random.default_rng(seed + 9000)
con = rng.random(len(idx)) < CON_FRAC
idx80, idxc = idx[~con], idx[con]
X80, y80 = Xb[idx80], y[idx80].astype(float)
Xc, yc_ = Xb[idxc], y[idxc].astype(float)
codes, uniq = pd.factorize(h[idx])
codes80, codesc = codes[~con], codes[con]
m = len(uniq)
nk80 = np.maximum(np.bincount(codes80, minlength=m).astype(float), 1)
nkc = np.maximum(np.bincount(codesc, minlength=m).astype(float), 1)
pi = nk80 / nk80.sum()
isk = np.isin(uniq, keep)


def wfit(rw, w0, maxiter):
    def f(w):
        z = X80 @ w
        l = np.logaddexp(0, z) - y80 * z
        loss = np.sum(rw * l) + 0.5 * LAM * (w[:-1] @ w[:-1])
        g = np.asarray(X80.T @ (rw * (expit(z) - y80))).ravel()
        g[:-1] += LAM * w[:-1]
        return loss, g
    return minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter}).x


def con_losses(w):
    z = Xc @ w
    return np.bincount(codesc, weights=np.logaddexp(0, z) - yc_ * z, minlength=m) / nkc


def pd_train(L0, eps, rho, w0):
    lam, ws = np.zeros(m), []
    for t in range(T_ROUNDS):
        rw = (pi + lam)[codes80] / nk80[codes80]
        rw = rw / (pi + lam).sum()
        w = wfit(rw, w0 if t == 0 else ws[-1], INNER)
        ws.append(w)
        v = con_losses(w) - L0 - eps
        v[~isk] = 0.0
        if t >= T0:
            lam = np.clip(lam + ETA * v, 0.0, rho)
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
w_pool80 = wfit(np.full(len(y80), 1.0 / len(y80)), np.zeros(Xb.shape[1]), 300)
print(f"pool80 fitted at {time.time() - t0:.0f}s", flush=True)

# local baselines on the constraint split (local models fitted on the 80% only)
L0 = np.full(m, np.nan)
for j in np.where(isk)[0]:
    a80, ac = codes80 == j, codesc == j
    lm = LogisticRegression(C=1.0, max_iter=300).fit(X80[a80][:, :-1], y80[a80])
    p = np.clip(lm.predict_proba(Xc[ac][:, :-1])[:, 1], 1e-12, 1 - 1e-12)
    yy = yc_[ac]
    L0[j] = -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))
print(f"local baselines done at {time.time() - t0:.0f}s", flush=True)

W = {"pool80": w_pool80}
v_pool = con_losses(w_pool80) - L0
for eps, _ in HEAD:
    best = None
    for rho in RHOS:
        w = pd_train(L0, eps, rho, w_pool80)
        v = (con_losses(w) - L0 - eps)[isk]
        nviol, mloss = int((v > 0).sum()), float(np.mean(np.logaddexp(0, Xc @ w) - yc_ * (Xc @ w)))
        print(f"  eps {eps} rho {rho}: constraint-split violations {nviol} (pool80: {int(((v_pool - eps)[isk] > 0).sum())}), "
              f"mean constraint loss {mloss:.5f}", flush=True)
        if best is None or (nviol, mloss) < best[:2]:
            best = (nviol, mloss, rho, w)          # rho chosen on the constraint split only
    W[f"pd_e{eps}"] = best[3]
    print(f"eps {eps}: chosen rho {best[2]} at {time.time() - t0:.0f}s", flush=True)
SRC = ["pool", "pool80"] + [f"pd_e{e}" for e, _ in HEAD]

rows = []
for n_done, k in enumerate(keep):
    mk = h == k
    a_, c_, t_ = mk & tr, mk & ce, mk & te
    loc = LogisticRegression(C=1.0, max_iter=300).fit(X[a_], y[a_])

    tg = [("overall", None)]
    for g in GROUPS:
        v = gcols[g][a_]
        for lev in np.unique(v):
            if y[a_][v == lev].sum() >= 20:
                tg.append((g, lev))

    def idx_of(mm):
        sub, out = {}, []
        for g, lev in tg:
            if g == "overall":
                out.append(np.arange(int(mm.sum())))
            else:
                if g not in sub:
                    sub[g] = gcols[g][mm]
                out.append(np.where(sub[g] == lev)[0])
        return out

    PR = {}
    for name, mm in (("cert", c_), ("test", t_)):
        d = {"local": loc.predict_proba(X[mm])[:, 1], "pool": pooled.predict_proba(X[mm])[:, 1]}
        for key in SRC[1:]:
            d[key] = expit(Xb[mm] @ W[key])
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
res.to_csv(ROOT / f"outputs/tables/c3_primal_dual_seed{seed}.csv", index=False)


def loss_mask(diff, se, eps):
    z = np.where(se > 0, (diff - eps) / np.where(se > 0, se, 1), -np.inf)
    return multipletests(1 - stats.norm.cdf(z), alpha=0.05, method="holm")[0]


out = []
for eps, delta in HEAD:
    pdk = f"pd_e{eps}"
    for src in ["pool", "pool80", pdk]:
        for label, pre in [(src, src), (src + " certified", f"cert_{src}_e{eps}_d{delta}")]:
            lm = loss_mask(res[f"{pre}__diff"].values, res[f"{pre}__se"].values, eps)
            cm = res[f"{pre}__gaplb"].values > delta
            nl = int((res[f"{pre}__blend"] > 0).sum()) if "cert" in pre else None
            out.append((eps, delta, label, int(lm.sum()), int(cm.sum()), int((lm | cm).sum()), nl,
                        res[f"{pre}__diff"].mean(), (res[f"{pre}__auroc"] - res["local__auroc"]).mean()))
print(f"seed {seed}: {len(res)} hospitals")
print(pd.DataFrame(out, columns=["eps", "delta", "method", "loss_harm", "cal_harm_sig", "either_sig", "nonlocal",
                                 "mean_loss_diff", "auroc_gain"]).round(5).to_string())

print("\nC3 decision quantities (this seed):")
for eps, delta in HEAD:
    pdk = f"pd_e{eps}"
    b_pd = (res[f"cert_{pdk}_e{eps}_d{delta}__blend"] > 0).values
    b_p0 = (res[f"cert_pool80_e{eps}_d{delta}__blend"] > 0).values
    n10, n01 = int((b_pd & ~b_p0).sum()), int((~b_pd & b_p0).sum())
    pv = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 > 0 else 1.0
    auc_cost = (res["pool80__auroc"] - res[f"{pdk}__auroc"]).mean()
    print(f"  eps {eps}: certified share pd {b_pd.sum()} vs pool80 {b_p0.sum()} (gain {b_pd.sum() - b_p0.sum()}, "
          f"McNemar p {pv:.3f}); uncertified AUROC cost {auc_cost:.4f}")
print(f"total {time.time() - t0:.0f}s")