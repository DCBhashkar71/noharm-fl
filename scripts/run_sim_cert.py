import sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from noharm_fl.data.features import ROOT
from noharm_fl.certify.bounds import clip_loss, betting_pvalue, calibration_gap_bound_clipped

r_a, r_b = int(sys.argv[1]), int(sys.argv[2])
SCEN = sys.argv[3] if len(sys.argv) > 3 else "base"      # base | icpt | shift
assert SCEN in ("base", "icpt", "shift")
SHIFT = 0.3 if SCEN == "shift" else 0.0
D, B0, K, ALPHA, B, C = 10, -3.8, 30, 0.05, 5.0, 0.05
HS = [0.0, 0.1, 0.2, 0.4, 0.8]
NS = [500, 2000, 8000, 30000]
BLENDS = [0.25, 0.5, 0.75, 1.0]
PAIRS = [(0.005, 0.01), (0.01, 0.015), (0.02, 0.02)]
NT = 100000

hs_sum = pd.read_csv(ROOT / "docs/hospital_summary_2022.csv")
logn = np.log(hs_sum["n"].clip(lower=50)); MU, SIG = logn.mean(), logn.std()
theta_bar = np.random.default_rng(0).normal(0, 0.5, D)


def unit(rng):
    u = rng.normal(size=D); return u / np.linalg.norm(u)


def gen(rng, theta, gam, n, with_y=True, b0=B0):
    X = rng.normal(size=(n, D)); g = (rng.random(n) < 0.3).astype(float)
    q = 1 / (1 + np.exp(-(b0 + X @ theta + gam * g)))
    Xg = np.column_stack([X, g])
    return (Xg, q, (rng.random(n) < q).astype(int)) if with_y else (Xg, q)


def exp_loss(p, q):
    return q * clip_loss(p, np.ones_like(p), B) + (1 - q) * clip_loss(p, np.zeros_like(p), B)


def cand(pl, ps, a):
    return np.clip(np.clip((1 - a) * pl + a * ps, pl - C, pl + C), 0.0, 1.0)


rows, t0 = [], time.time()
for rep in range(r_a, r_b):
    for hi, h in enumerate(HS):
        rng = np.random.default_rng([rep, hi, 7])
        sizes = np.clip(np.exp(rng.normal(MU, SIG, K)), 500, 50000).astype(int)
        Xs, ys = [], []
        for n in sizes:
            th_k, gam_k = theta_bar + h * unit(rng), 0.3 + h * rng.normal()
            b0_k = B0 + (h * rng.normal() if SCEN == "icpt" else 0.0)
            Xk, _, yk = gen(rng, th_k, gam_k, n, b0=b0_k)
            Xs.append(Xk); ys.append(yk)
        pooled = LogisticRegression(max_iter=300).fit(np.vstack(Xs), np.concatenate(ys))
        th_f, gam_f = theta_bar + h * unit(rng), 0.3 + h * rng.normal()
        b0_f = B0 + (h * rng.normal() if SCEN == "icpt" else 0.0)
        Xt, qt = gen(rng, th_f, gam_f, NT, with_y=False, b0=b0_f + SHIFT)   # truth distribution (shifted in 'shift')
        pp_t = pooled.predict_proba(Xt)[:, 1]; gt = Xt[:, -1]
        for n in NS:
            Xf, qf, yf = gen(rng, th_f, gam_f, n, b0=b0_f)                    # certification data: never shifted
            tr, ce = slice(0, n // 2), slice(n // 2, n // 2 + n // 4)
            if yf[tr].sum() >= 3:
                loc = LogisticRegression(max_iter=300).fit(Xf[tr], yf[tr])
                pl_t = loc.predict_proba(Xt)[:, 1]; pl_c = loc.predict_proba(Xf[ce])[:, 1]
            else:
                m = max(yf[tr].mean(), 1e-3); pl_t = np.full(NT, m); pl_c = np.full(n // 4, m)
            pp_c = pooled.predict_proba(Xf[ce])[:, 1]
            yc = yf[ce].astype(float); gc = Xf[ce][:, -1]; gtr = Xf[tr][:, -1]; ytr = yf[tr]
            cm, tm = [np.ones(len(yc), bool)], [np.ones(NT, bool)]          # testable groups from TRAINING counts
            for lev in (0.0, 1.0):
                if ytr[gtr == lev].sum() >= 20:
                    cm.append(gc == lev); tm.append(gt == lev)
            base_t = exp_loss(pl_t, qt).mean(); base_c = clip_loss(pl_c, yc, B)

            def truth(pc_t):
                ld = exp_loss(pc_t, qt).mean() - base_t
                gap = max(abs((pc_t[m] - qt[m]).mean()) - abs((pl_t[m] - qt[m]).mean()) for m in tm)
                return ld, gap

            T = [truth(cand(pl_t, pp_t, a)) for a in BLENDS]
            full_ld, full_gap = truth(pp_t)
            lossp, calmax = {}, []
            for j, a in enumerate(BLENDS):
                pc = cand(pl_c, pp_c, a); dl = clip_loss(pc, yc, B) - base_c
                for eps, _ in PAIRS:
                    lossp[(j, eps)] = betting_pvalue(dl, eps, -B, B, ALPHA)
                calmax.append(max(calibration_gap_bound_clipped(pc[m], pl_c[m], yc[m], C, ALPHA) for m in cm))
            for eps, delta in PAIRS:
                sel = -1
                for j in range(len(BLENDS)):
                    if lossp[(j, eps)] <= ALPHA and calmax[j] <= delta:
                        sel = j
                    else:
                        break
                harm = (sel >= 0) and (T[sel][0] > eps or T[sel][1] > delta)
                rows.append((rep, h, n, eps, delta, 0.0 if sel < 0 else BLENDS[sel], int(harm), 0.0 if sel < 0 else -T[sel][0],
                             int(sel >= 0), int(full_ld > eps or full_gap > delta),
                             int(T[3][0] > eps or T[3][1] > delta), int(sel == 3)))
    print(f"rep {rep} done, {time.time() - t0:.0f}s", flush=True)

tag = "" if SCEN == "base" else f"{SCEN}_"
pd.DataFrame(rows, columns=["rep", "h", "n", "eps", "delta", "blend", "harm_certified", "gain_certified", "nonlocal",
                            "harm_pool_full", "harm_pool_clipped", "approved_pool"]).to_csv(
    ROOT / f"outputs/tables/sim_cert_{tag}{r_a}-{r_b}.csv", index=False)
print("saved", SCEN)