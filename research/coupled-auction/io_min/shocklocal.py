"""Variant (d): local-level shock forecaster (protocol addendum 5). Usage: shocklocal.py SP PANEL_DIR POST.npz OUT_PREFIX NMTU SPLIT VAL_PRED_RECS.pkl
Forecast u_t = sum_{k=1..7} w_k u_{t-k days}, w_k ∝ lam^(k-1), normalised; lam chosen on t < SPLIT by summed shock RMSE."""
import sys, json, pickle
import numpy as np, pandas as pd
SP, PANEL, POST, OUTP, NMTU, SPLIT, VAL = sys.argv[1:8]; NMTU = int(NMTU); SPLIT = pd.Timestamp(SPLIT, tz="UTC")
sys.argv = ["shockar.py", SP, PANEL, POST, "/dev/null", "0", "2026-07-01"]
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import shockar as A, solve11 as V
K = 7; GRID = np.round(np.arange(0.0, 0.91, 0.1), 2)
w = lambda lam: (lam ** np.arange(K)) / (lam ** np.arange(K)).sum() if lam > 0 else np.r_[1.0, np.zeros(K - 1)]
def lags(t):
    L = [A.shock(t - k * A.DAY) for k in range(1, K + 1)]
    return None if any(l is None for l in L) else np.stack(L, 1)          # (NZ, K)
rng = np.random.default_rng(5)
grid = pd.date_range("2025-10-08 00:00", "2026-09-07 23:45", freq="15min", tz="UTC")
times = sorted(pd.Timestamp(t) for t in rng.choice(grid, NMTU, replace=False))
T, Y, X = [], [], []
for t in times:
    u = A.shock(t); L = lags(t)
    if u is None or L is None: continue
    T.append(t); Y.append(u); X.append(L)
T = pd.DatetimeIndex(T); Y = np.array(Y); X = np.array(X); tr, va = T < SPLIT, T >= SPLIT
rs = lambda lam, m: float(np.sqrt(np.mean(((X[m] @ w(lam)).sum(1) - Y[m].sum(1)) ** 2)))
sel = {float(l): rs(l, tr) for l in GRID}; lam = min(sel, key=sel.get)
out = dict(n_train=int(tr.sum()), n_val=int(va.sum()), train_rmse_by_lambda=sel, lambda_=lam, weights=np.round(w(lam), 3).tolist(),
           V1d=dict(b_sum=rs(0.0, va), d_sum=rs(lam, va), b_DE=float(np.sqrt(np.mean((X[va][:, 0, 0] - Y[va][:, 0]) ** 2))),
                    d_DE=float(np.sqrt(np.mean((X[va][:, 0] @ w(lam) - Y[va][:, 0]) ** 2))), b_FR=float(np.sqrt(np.mean((X[va][:, 1, 0] - Y[va][:, 1]) ** 2))),
                    d_FR=float(np.sqrt(np.mean((X[va][:, 1] @ w(lam) - Y[va][:, 1]) ** 2)))))
print(json.dumps(out, indent=1), flush=True)
recs = pickle.load(open(VAL, "rb"))["recs"]; res = []
for r in recs:
    L = lags(r["t"])
    if L is None: continue
    c, tm = V.curves_logistic(A.TH, A.NB, r); row = dict(t=r["t"])
    for name, u in (("b", L[:, 0]), ("d", L @ w(lam))):
        p, y, ok, _ = V.solve_ip(c, tm, r["D"] + r["X"] + 1000 * u, r["H"], r["A"], r["b"], r["e"], r["e0"], r["lo"], r["hi"])
        row[name] = float(p[0]) if ok else np.nan
    res.append(row)
R = pd.DataFrame(res); R["obs"] = A.B.price["DE"].reindex(pd.DatetimeIndex(R.t)).to_numpy(); bo = R.dropna()
out["V2d"] = dict(n=len(bo), mae_b=float((bo.b - bo.obs).abs().mean()), mae_d=float((bo.d - bo.obs).abs().mean()), bias_b=float((bo.b - bo.obs).mean()), bias_d=float((bo.d - bo.obs).mean()))
print("V2d", out["V2d"], flush=True); json.dump(out, open(OUTP + ".json", "w"), indent=1)
