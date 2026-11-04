import sys, time
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from statsmodels.stats.multitest import multipletests
from noharm_fl.data.features import load, ROOT
from noharm_fl.certify.bounds import clip_loss, betting_pvalue, fixed_sequence

seed = int(sys.argv[1])
with_apr = len(sys.argv) > 2 and sys.argv[2] == "apr"
B, ALPHA = 5.0, 0.05
BLENDS = [0.0, 0.25, 0.5, 0.75, 1.0]
EPS = [0.005, 0.01, 0.02]
MU, ROUNDS, LOCAL_IT, LAM, FT_IT = 0.01, 30, 10, 1e-5, 5      # fixed in advance, not tuned

X, y, h, split, keep = load(seed, with_apr)
Xb = sp.hstack([X, sp.csr_matrix(np.ones((X.shape[0], 1)))]).tocsr()
tr, ce, te = split == "train", split == "cert", split == "test"


def local_solve(Xk, yk, w0, mu, wt, maxiter, lam=LAM):
    n = Xk.shape[0]
    def f(w):
        z = Xk @ w
        loss = np.mean(np.logaddexp(0, z) - yk * z) + 0.5 * lam * (w[:-1] @ w[:-1]) + 0.5 * mu * np.sum((w - wt) ** 2)
        g = np.asarray(Xk.T @ (expit(z) - yk)).ravel() / n
        g[:-1] += lam * w[:-1]
        g += mu * (w - wt)
        return loss, g
    return minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter}).x


def federated(mu):
    clients = [np.where((h == k) & tr)[0] for k in np.unique(h)]
    clients = [c for c in clients if len(c) >= 20]
    data = [(Xb[c], y[c], len(c)) for c in clients]
    p0 = y[tr].mean()
    w = np.zeros(Xb.shape[1]); w[-1] = np.log(p0 / (1 - p0))
    for _ in range(ROUNDS):
        acc = np.zeros_like(w)
        for Xk, yk, n in data:
            acc += n * local_solve(Xk, yk, w, mu, w, LOCAL_IT)
        w = acc / sum(d[2] for d in data)
    return w


t0 = time.time()
pooled = LogisticRegression(C=1.0, max_iter=300).fit(X[tr], y[tr])
w_pool = np.append(pooled.coef_[0], pooled.intercept_[0])
w_avg = federated(0.0)
w_prox = federated(MU)
print(f"training done in {time.time() - t0:.0f}s", flush=True)


def metrics(p, yy):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"loss": clip_loss(p, yy, B).mean(), "cal_int": float(p.mean() - yy.mean()),
           "auroc": roc_auc_score(yy, p) if 0 < yy.sum() < len(yy) else np.nan,
           "auprc": average_precision_score(yy, p) if yy.sum() > 0 else np.nan, "cal_slope": np.nan}
    if yy.sum() >= 20:
        try:
            out["cal_slope"] = LogisticRegression(C=1e6, max_iter=200).fit(np.log(pc / (1 - pc)).reshape(-1, 1), yy).coef_[0][0]
        except Exception:
            pass
    return out


def decile_rows(k, name, p, yy):
    if len(p) < 50:
        return []
    try:
        g = pd.DataFrame({"p": p, "y": yy, "b": pd.qcut(p, 10, duplicates="drop")}).groupby("b", observed=True).agg(
            n=("y", "size"), mean_p=("p", "mean"), mean_y=("y", "mean"))
        return [(k, name, i, r.n, r.mean_p, r.mean_y) for i, r in enumerate(g.itertuples())]
    except Exception:
        return []


rows, bins = [], []
for k in keep:
    mk = h == k
    a_, c_, t_ = mk & tr, mk & ce, mk & te
    loc = LogisticRegression(C=1.0, max_iter=300).fit(X[a_], y[a_])
    w_ft = local_solve(Xb[a_], y[a_], w_pool, 0.0, w_pool, FT_IT)
    rc = LogisticRegression(C=1e6, max_iter=300).fit((Xb[a_] @ w_pool).reshape(-1, 1), y[a_])
    P = {}
    for name, m in (("cert", c_), ("test", t_)):
        pl = loc.predict_proba(X[m])[:, 1]
        pf = pooled.predict_proba(X[m])[:, 1]
        d = {"local": pl, "fedavg": expit(Xb[m] @ w_avg), "fedprox": expit(Xb[m] @ w_prox),
             "finetune": expit(Xb[m] @ w_ft), "recal": rc.predict_proba((Xb[m] @ w_pool).reshape(-1, 1))[:, 1]}
        for a in BLENDS[1:]:
            d[f"b{a}"] = (1 - a) * pl + a * pf
            d[f"f{a}"] = (1 - a) * pl + a * d["finetune"]
        P[name] = d
    base_c = clip_loss(P["cert"]["local"], y[c_], B)
    sel = {}
    for eps in EPS:
        pv = [betting_pvalue(clip_loss(P["cert"][f"b{a}"], y[c_], B) - base_c, eps, -B, B, ALPHA) for a in BLENDS[1:]]
        j = fixed_sequence(pv, ALPHA) + 1
        sel[eps] = BLENDS[j]
        P["test"][f"cert{eps}"] = P["test"]["local"] if j == 0 else P["test"][f"b{BLENDS[j]}"]
    for eps in EPS:
        pv = [betting_pvalue(clip_loss(P["cert"][f"f{a}"], y[c_], B) - base_c, eps, -B, B, ALPHA) for a in BLENDS[1:]]
        j = fixed_sequence(pv, ALPHA) + 1
        sel[f"ft{eps}"] = BLENDS[j]
        P["test"][f"certft{eps}"] = P["test"]["local"] if j == 0 else P["test"][f"f{BLENDS[j]}"]
    yt = y[t_]
    base_t = clip_loss(P["test"]["local"], yt, B)
    r = {"hospital": k, "n_train": int(a_.sum()), "deaths_train": int(y[a_].sum()), "n_test": int(t_.sum()), "deaths_test": int(yt.sum())}
    for eps in EPS:
        r[f"sel_eps{eps}"] = sel[eps]
        r[f"sel_ft{eps}"] = sel[f"ft{eps}"]
    for name, p in P["test"].items():
        dl = clip_loss(p, yt, B) - base_t
        r[f"{name}__diff"] = dl.mean()
        r[f"{name}__se"] = dl.std(ddof=1) / np.sqrt(len(dl)) if dl.std() > 0 else 0.0
        for mname, v in metrics(p, yt).items():
            r[f"{name}__{mname}"] = v
    rows.append(r)
    for name in ("local", "b1.0", "cert0.01", "finetune", "certft0.01"):
        bins += decile_rows(k, name, P["test"][name], yt)

res = pd.DataFrame(rows)
tag = "apr" if with_apr else "noapr"
res.to_csv(ROOT / f"outputs/tables/harm_seed{seed}_v3_{tag}.csv", index=False)
pd.DataFrame(bins, columns=["hospital", "method", "bin", "n", "mean_p", "mean_y"]).to_csv(ROOT / f"outputs/tables/bins_seed{seed}_{tag}.csv", index=False)


def summ(diff, se):
    z = np.where(se > 0, diff / np.where(se > 0, se, 1), 0.0)
    w = multipletests(1 - stats.norm.cdf(z), alpha=0.05, method="holm")[0]
    b = multipletests(stats.norm.cdf(z), alpha=0.05, method="holm")[0]
    return int(w.sum()), int(b.sum())


out = {}
for m in ["b0.25", "b0.5", "b0.75", "b1.0" if "b1.0__diff" in res else "fedavg", "fedavg", "fedprox", "finetune", "recal"] + [f"cert{e}" for e in EPS] + [f"certft{e}" for e in EPS]:
    if f"{m}__diff" not in res:
        continue
    sw, sb = summ(res[f"{m}__diff"].values, res[f"{m}__se"].values)
    out[m] = {"mean_diff": res[f"{m}__diff"].mean(), "raw_worse": (res[f"{m}__diff"] > 0).mean(), "sig_worse": sw, "sig_better": sb,
              "auroc": res[f"{m}__auroc"].mean(), "auroc_gain_vs_local": (res[f"{m}__auroc"] - res["local__auroc"]).mean()}
print(f"seed {seed} {tag}: {len(res)} hospitals, local mean AUROC {res['local__auroc'].mean():.4f}")
print(pd.DataFrame(out).T.round(5))
print(f"total {time.time() - t0:.0f}s")