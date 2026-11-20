import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M, live_common as lc
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; F = lc.load(); D = F["DE"]; idx = D.index; loc = idx.tz_convert("Europe/Berlin")
WE = ((D["weekday"].to_numpy(float) >= 5) | (D["holiday"].to_numpy(float) >= 0.5)); key = loc.hour.to_numpy() + 24 * WE.astype(int); shape = np.load(f"{SP}/out/core_shape.npy")
Z = np.load(f"{SP}/out/shift15_lite.npz"); X = Z["rl"] - Z["other"]; U = X - Z["Sp"]; p = Z["p"]; DL = np.load(f"{SP}/out/day_look.npz")
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); R.index = R.index.tz_convert("UTC"); gas = R.gas.reindex(idx, method="ffill").shift(192).ffill().bfill().to_numpy(); L = 1.85 * gas + 0.37 * 75.0
for per, tag in (("winter", "mid"), ("summer", "mid"), ("autumn", "v2"), ("winterf", "mid"), ("summerf", "mid")):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); cuts = I["cuts"].astype(int); a = DL[f"{per}_a"].astype(float); n = len(cuts)
    hrs = sorted(set(loc.hour[cuts])); chk = np.nanmax(np.abs(np.stack([p[c:c + 96] / 10 for c in cuts]) - a)); lvl = np.array([np.nanmean(U[c - 672:c]) for c in cuts])
    ut = np.stack([U[c:c + 96] for c in cuts]); xt = np.stack([X[c:c + 96] for c in cuts]); xf = T["xh"].astype(float)[:, :96]; sh = np.stack([shape[key[c:c + 96]] for c in cuts])
    rm = lambda e: float(np.sqrt(np.nanmean(e ** 2)))
    # shift error of the level alone and of level + shape against the true shift; the true shift's own deviation from its day mean vs the shape
    dv = ut - np.nanmean(ut, 1, keepdims=True)
    def pr(eps, x): return np.stack([np.asarray(M.clear(th, x[j], np.full(96, L[cuts[j]:cuts[j] + 96].mean()), np.full(96, 75.0), eps=eps[j])) / 10 for j in range(n)])
    E = lambda v: np.nanmean(np.abs(v - a))
    print(f"{per:8s} cut local hours {hrs} | prices match day_look: {chk:.3f} | shift RMSE: level {rm(ut - lvl[:, None]):.2f}  level+shape {rm(ut - lvl[:, None] - sh):.2f} | corr(shape, true deviation) {np.corrcoef(sh.ravel(), np.nan_to_num(dv.ravel()))[0, 1]:.2f} | demand forecast error RMSE {rm(xf - xt):.2f} GW"
          + f" | price MAE with REALISED demand: level {E(pr(np.repeat(lvl[:, None], 96, 1), xt)):.3f} level+shape {E(pr(lvl[:, None] + sh, xt)):.3f} | with FORECAST demand: level {E(pr(np.repeat(lvl[:, None], 96, 1), xf)):.3f} level+shape {E(pr(lvl[:, None] + sh, xf)):.3f}")
