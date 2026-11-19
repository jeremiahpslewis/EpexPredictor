"""D5 (notes 21.138): price error over 2.7 years of the curves with a frozen shift profile, with a slow level state, and with level state + frozen first-year shape."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M, live_common as lc
SP = M.SP; P_ = np.load(f"{SP}/out/market1n_post.npz"); th = {k: float(np.median(P_[k])) for k in M.PN}; F = lc.load(); D = F["DE"]; idx = D.index; loc = idx.tz_convert("Europe/Berlin")
WE = ((D["weekday"].to_numpy(float) >= 5) | (D["holiday"].to_numpy(float) >= 0.5)); hod = loc.hour.to_numpy(); qtr = loc.to_period("Q"); key = hod + 24 * WE.astype(int)
Z = np.load(f"{SP}/out/shift15_lite.npz"); X = Z["rl"] - Z["other"]; U = X - Z["Sp"]; p = Z["p"]
R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); R.index = R.index.tz_convert("UTC"); gas = R.gas.reindex(idx, method="ffill").shift(192).ffill().bfill().to_numpy(); L = 1.85 * gas + 0.37 * 75.0
fitw = np.asarray(idx < pd.Timestamp("2025-12-01", tz="UTC")); first = np.asarray(idx < pd.Timestamp("2025-03-01", tz="UTC"))
day = np.asarray((loc.normalize() - loc.normalize()[0]).days); dm = pd.Series(U).groupby(day).mean(); lvl7 = dm.rolling(7, min_periods=4).mean().shift(1).reindex(day).to_numpy()       # mean of the 7 whole days before the day
prof_a = pd.Series(U[fitw]).groupby(key[fitw]).mean().reindex(range(48)).to_numpy()[key]
dev = U - pd.Series(U).groupby(day).transform("mean").to_numpy(); shape1 = pd.Series(dev[first]).groupby(key[first]).mean().reindex(range(48)).to_numpy()[key]
ok = np.isfinite(p) & np.isfinite(X) & np.isfinite(L) & np.isfinite(lvl7)
def price(eps): out = np.full(len(idx), np.nan); out[ok] = np.asarray(M.clear(th, X[ok], L[ok], np.full(ok.sum(), 75.0), eps=eps[ok])); return out
V = {"(a) frozen shift profile": price(prof_a), "(b) level of the last 7 days": price(lvl7), "(c) level + first-year shape": price(lvl7 + shape1)}
E = pd.DataFrame({k: np.abs(v - p) / 10 for k, v in V.items()}, index=idx)[ok]; lev = pd.Series(p / 10, index=idx)[ok].groupby(qtr[ok]).mean(); T = E.groupby(qtr[ok]).mean()
print("price MAE ct/kWh by quarter"); print(T.round(2).T.to_string()); print("\nas a share of the mean price, %"); print((100 * T.div(lev, axis=0)).round(0).T.to_string()); print("\nmean price ct:", lev.round(1).to_dict())
(100 * T.div(lev, axis=0)).to_csv(f"{SP}/out/durable5.csv"); T.to_csv(f"{SP}/out/durable5_abs.csv")
