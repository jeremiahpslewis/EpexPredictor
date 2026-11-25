"""
Shock persistence forecaster, variant (c) (prereg/OOS_PROTOCOL.md addendum 4). Usage:
    shockar.py SP PANEL_DIR POST.npz OUT_PREFIX NMTU SPLIT [VAL_PRED_RECS.pkl]
1. Recovers the eleven zonal shocks (posterior draw 0) at t and at t-1d, t-2d, t-7d for NMTU random training MTUs.
2. Per zone, Bayesian linear regression u_t = c + phi1 u_t-1d + phi2 u_t-2d + phi7 u_t-7d + e, prior phi ~ N(0, 0.5^2),
   c ~ N(0, 5^2), noise variance plugged in from OLS residuals; posterior mean coefficients. Fitted on t < SPLIT.
3. V1: shock RMSE on t >= SPLIT for (a) the fitted mean (mu_DE, 0 elsewhere), (b) carry u_t-1d, (c) the AR.
4. V2 (if VAL_PRED_RECS given): German price on those pre-gate prediction records cleared with (b) and (c).
Writes OUT_PREFIX.json (coefficients, V1, V2) and OUT_PREFIX.coef.npz (coefficients fitted on ALL training MTUs, for use
on the holdout).
"""
import sys, json, pickle
import numpy as np, pandas as pd
SP, PANEL, POST, OUTP, NMTU, SPLIT = sys.argv[1:7]; NMTU = int(NMTU); SPLIT = pd.Timestamp(SPLIT, tz="UTC")
VAL = sys.argv[7] if len(sys.argv) > 7 else None
sys.argv = ["predict11_build.py", SP, PANEL, "/dev/null", "2025-10-01", "2025-10-02"]
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import predict11_build as B, net11 as N, solve11 as V
P = dict(np.load(POST, allow_pickle=True)); i = 0; KEYS = [str(x) for x in P["de_keys"]]
TH = {k: float(P[k][i]) for k in KEYS}; NB = dict(sbar=P["sbar"][1:], al=P["al"][i], be=P["be"][i], aa=P["aa"][i], bb=P["bb"][i], sg=P["sg"][i])
MU = np.zeros(N.NZ); MU[0] = float(P["mu_de"][i]); DAY = pd.Timedelta(days=1); LAGS = (1, 2, 7)
_cache = {}
def shock(t):
    if t in _cache: return _cache[t]
    o = B.outcome_record(t); u = None
    if o is not None:
        c, tm = V.curves_logistic(TH, NB, o); S, _ = V.supply_fn(c, tm)(o["p"]); u = (S - o["D"] - o["X"] - o["n"]) / 1000.0
        if not np.all(np.isfinite(u)): u = None
    _cache[t] = u; return u

def design(times):
    rows, X, Y = [], [], []
    for t in times:
        u = shock(t); lg = [shock(t - k * DAY) for k in LAGS]
        if u is None or any(l is None for l in lg): continue
        rows.append(t); Y.append(u); X.append(np.stack(lg, 1))          # X: (NZ, 3)
    return pd.DatetimeIndex(rows), np.array(Y), np.array(X)

def fit(X, Y):
    coef = np.zeros((N.NZ, 4))
    for z in range(N.NZ):
        A = np.c_[np.ones(len(Y)), X[:, z, :]]; y = Y[:, z]
        b_ols = np.linalg.lstsq(A, y, rcond=None)[0]; s2 = float(np.mean((y - A @ b_ols) ** 2))
        prec0 = np.diag([1 / 25.0] + [1 / 0.25] * 3)
        coef[z] = np.linalg.solve(A.T @ A / s2 + prec0, A.T @ y / s2)
    return coef
predict_c = lambda coef, lags: coef[:, 0] + (coef[:, 1:] * lags).sum(1)       # lags: (NZ, 3)

if __name__ == "__main__":
    rng = np.random.default_rng(5)
    grid = pd.date_range("2025-10-08 00:00", "2026-09-07 23:45", freq="15min", tz="UTC")
    times = sorted(rng.choice(grid, NMTU, replace=False))
    T, Y, X = design([pd.Timestamp(t) for t in times]); print("MTUs with all lags:", len(T), flush=True)
    tr, va = T < SPLIT, T >= SPLIT
    coef = fit(X[tr], Y[tr])
    pa = np.tile(MU, (va.sum(), 1)); pb = X[va][:, :, 0]; pc = np.array([predict_c(coef, x) for x in X[va]])
    rmse = lambda p, z=None: float(np.sqrt(np.mean((p - Y[va]) ** 2))) if z is None else float(np.sqrt(np.mean((p[:, z] - Y[va][:, z]) ** 2)))
    rs = lambda p: float(np.sqrt(np.mean((p.sum(1) - Y[va].sum(1)) ** 2)))
    V1 = {f: dict(DE=rmse(p, 0), FR=rmse(p, 1), sum_over_zones=rs(p), all=rmse(p)) for f, p in (("a_mean", pa), ("b_carry", pb), ("c_ar", pc))}
    out = dict(n_train=int(tr.sum()), n_val=int(va.sum()), coef={z: np.round(coef[k], 3).tolist() for k, z in enumerate(N.Z)}, V1=V1)
    print(json.dumps(out, indent=1), flush=True)
    if VAL:
        recs = pickle.load(open(VAL, "rb"))["recs"]; const_cache = {}; res = []
        for r in recs:
            lg = [shock(r["t"] - k * DAY) for k in LAGS]
            if any(l is None for l in lg): continue
            c, tm = V.curves_logistic(TH, NB, r); row = dict(t=r["t"])
            for name, u in (("b", lg[0]), ("c", predict_c(coef, np.stack(lg, 1)))):
                p, y, ok, _ = V.solve_ip(c, tm, r["D"] + r["X"] + 1000 * u, r["H"], r["A"], r["b"], r["e"], r["e0"], r["lo"], r["hi"])
                row[name] = float(p[0]) if ok else np.nan
            res.append(row)
        R = pd.DataFrame(res); R["obs"] = B.price["DE"].reindex(pd.DatetimeIndex(R.t)).to_numpy()
        both = R.dropna(); out["V2"] = dict(n=len(both), mae_b=float((both.b - both.obs).abs().mean()), mae_c=float((both.c - both.obs).abs().mean()),
                                           bias_b=float((both.b - both.obs).mean()), bias_c=float((both.c - both.obs).mean()), missed=int(len(R) - len(both)))
        print("V2", out["V2"], flush=True)
    json.dump(out, open(OUTP + ".json", "w"), indent=1)
    np.savez(OUTP + ".coef.npz", coef=fit(X, Y), zones=np.array(N.Z), lags=np.array(LAGS))   # all training MTUs, for holdout use
