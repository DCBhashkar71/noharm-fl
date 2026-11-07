import sys, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from noharm_fl.data.features import ROOT
from noharm_fl.certify.bounds import clip_loss, betting_pvalue, calibration_gap_bound_clipped

warnings.filterwarnings("ignore")
seed = int(sys.argv[1])
B, ALPHA, C = 5.0, 0.05, 0.05
BLENDS = [0.25, 0.5, 0.75, 1.0]
EPS = [0.005, 0.01, 0.02]
DELTA = [0.01, 0.015, 0.02]
HEAD = list(zip(EPS, DELTA))
LOGC, LAM, FT_IT = 0.1, 1e-2, 5

d = pd.read_parquet(ROOT / "data/interim/physionet2012.parquet").reset_index()
y = d["y"].values.astype(int)
h = d["icu_type"].values.astype(int)

rng = np.random.default_rng(seed)
u = pd.Series(rng.random(len(d)))
rk = u.groupby([d["icu_type"], d["y"]]).rank(pct=True)
split = np.where(rk <= 0.5, "train", np.where(rk <= 0.75, "cert", "test"))
pd.DataFrame({"rid": d["rid"], "split": split}).to_parquet(ROOT / f"data/splits/icu2012_seed{seed}.parquet")
tr, ce, te = split == "train", split == "cert", split == "test"

meta = {"rid", "set", "icu_type", "y"}
feat = [c for c in d.columns if c not in meta]
F = d[feat].astype(float).copy()
cnt = [c for c in feat if c.endswith("_count")]
F[cnt] = F[cnt].fillna(0.0)
ind = (F[cnt] == 0).astype(float).values
F = F.fillna(F[tr].median()).fillna(0.0)
mu, sd = F[tr].mean(), F[tr].std().replace(0, 1.0)
Z = ((F - mu) / sd).clip(-5, 5)
Xf = np.hstack([Z.values, ind])
X = np.hstack([Xf, np.ones((len(d), 1))])                     # last column = intercept

gender = d["gender"].values
age = d["age"].fillna(d["age"].median()).values
gcols = {"gender": np.where(gender == 1, "M", np.where(gender == 0, "F", "U")),
         "age_band": np.where(age < 55, "<55", np.where(age < 75, "55-74", "75+"))}
GROUPS = list(gcols)


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
pooled = LogisticRegression(C=LOGC, max_iter=500).fit(Xf[tr], y[tr])
w_pool = np.append(pooled.coef_[0], pooled.intercept_[0])

rows, diag = [], []
for k in sorted(np.unique(h)):
    mk = h == k
    a_, c_, t_ = mk & tr, mk & ce, mk & te
    loc = LogisticRegression(C=LOGC, max_iter=500).fit(Xf[a_], y[a_])
    w_ft = local_solve(X[a_], y[a_], w_pool, FT_IT)

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

    PR = {}
    for name, m in (("cert", c_), ("test", t_)):
        PR[name] = {"local": loc.predict_proba(Xf[m])[:, 1], "pool": pooled.predict_proba(Xf[m])[:, 1], "ft": expit(X[m] @ w_ft)}

    yc = y[c_].astype(float); ic = idx_of(c_, tg); pl_c = PR["cert"]["local"]; base_c = clip_loss(pl_c, yc, B)
    lossp, calmax = {}, {}
    for src in ("pool", "ft"):
        for j, a in enumerate(BLENDS):
            pc = cand(pl_c, PR["cert"][src], a)
            dl = clip_loss(pc, yc, B) - base_c
            for eps in EPS:
                lossp[(src, j, eps)] = betting_pvalue(dl, eps, -B, B, ALPHA)
            calmax[(src, j)] = max(calibration_gap_bound_clipped(pc[i], pl_c[i], yc[i], C, ALPHA) for i in ic)

    # diagnostic output (amendment 19): which test blocks approval, per candidate and group
    for src in ("pool", "ft"):
        for j, a in enumerate(BLENDS):
            pc_ = cand(pl_c, PR["cert"][src], a)
            for gi, (gname, glev) in enumerate(tg):
                diag.append((seed, int(k), src, a, gname, "" if glev is None else str(glev), len(ic[gi]), int(yc[ic[gi]].sum()),
                             calibration_gap_bound_clipped(pc_[ic[gi]], pl_c[ic[gi]], yc[ic[gi]], C, ALPHA),
                             float(np.mean(clip_loss(pc_, yc, B) - base_c)),
                             lossp[(src, j, 0.005)], lossp[(src, j, 0.01)], lossp[(src, j, 0.02)]))

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

    yt = y[t_].astype(float); it = idx_of(t_, tg); pl_t = PR["test"]["local"]; base_t = clip_loss(pl_t, yt, B)
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

    r = {"icu_type": int(k), "seed": seed, "n_train": int(a_.sum()), "deaths_train": int(y[a_].sum()), "n_test": int(t_.sum()),
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
res.to_csv(ROOT / f"outputs/tables/icu2012_seed{seed}.csv", index=False)
pd.DataFrame(diag, columns=["seed", "icu_type", "source", "blend", "group", "level", "n_cert", "deaths_cert", "cal_bound",
                            "loss_diff_cert", "p_eps0.005", "p_eps0.01", "p_eps0.02"]).to_csv(
    ROOT / f"outputs/tables/icu2012_diag_seed{seed}.csv", index=False)

out = []
for _, r in res.iterrows():
    for eps, delta in HEAD:
        for label, pre in [("pooled", "pool_full"), ("certified pooled", f"cert_pool_e{eps}_d{delta}"),
                           ("certified fine-tuned", f"cert_ft_e{eps}_d{delta}")]:
            out.append((int(r["icu_type"]), eps, delta, label, r.get(f"{pre}__blend", np.nan), r[f"{pre}__diff"],
                        r[f"{pre}__gapmax"], r[f"{pre}__auroc"] - r["local__auroc"]))
print(f"seed {seed}: ICU units, local AUROC by unit: " + ", ".join(f"{int(a)}: {b:.3f}" for a, b in zip(res.icu_type, res['local__auroc'])))
print(res[["icu_type", "n_train", "deaths_train", "n_test", "deaths_test", "n_groups_tested"]].to_string(index=False))
print(pd.DataFrame(out, columns=["icu", "eps", "delta", "method", "blend", "loss_diff", "cal_gap_max", "auroc_gain"]).round(4).to_string(index=False))
print(f"total {time.time() - t0:.0f}s")