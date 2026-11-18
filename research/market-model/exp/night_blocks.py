"""Which block carries the night-to-night surprise in the supply shift (notes 21.45)? Night (00:00-06:00) mean shock of each block in M1, its change
from one night to the next, and how each block's change lines up with the layer's error on the target nights."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; th = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/market1_post.npz").items() if np.ndim(v) == 1}; R = pd.read_parquet(f"{SP}/out/market1_rows.parquet"); R["fe"] = R.rlreal - R.rl; R["tot"] = R.d_l + R.d_c + R.d_g + R.eX + R.eS + R.eps; ng = R[R.index.hour < 6]; BL = (("lignite", "d_l"), ("hard coal", "d_c"), ("gas", "d_g"), ("net imports", "eX"), ("storage", "eS"), ("unmeasured", "eps"), ("load: realised - published", "fe"), ("total", "tot"))
N = ng.groupby(ng.index.normalize())[[c for _, c in BL]].mean().asfreq("D"); dN = N.diff(); print("night-mean shock of each block, GW: sd of the level | sd of the change from one night to the next | share of the variance of the total's change (covariance share)")
vt = dN.tot.var()
for nm, c in BL: print(f"   {nm:28s} {N[c].std():5.2f} | {dN[c].std():5.2f} | {dN[[c, 'tot']].cov().iloc[0, 1] / vt:+.2f}")
E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; H = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.r_[E, np.full(max(0, len(H) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(H) * 4].reshape(-1, 4), 1), index=H.index).ffill(); rows = []
for per in ("winter", "summer", "autumn"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float)[:, :24]; fk = np.load(f"{SP}/out/structfc_m1m_{per}_price.npz")["vint"].astype(float)[:, :24]
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); L = np.full(24, 1.85 * float(I["gas"][j]) + 0.37 * cj); C = np.full(24, cj); xk = np.load(f"{SP}/out/struct_m1m/{str(c)[:10]}.npz")["x_known"].astype(float)[:24]; sup = lambda p: M.supply(th, p, L, C) - th["K_s"] * (p < -M.EPS) - 0.5 * th["K_s"] * (np.abs(p) <= M.EPS); tday = t0.tz_convert("Europe/Berlin").normalize() + pd.Timedelta(days=0)
        tday = (t0.tz_convert("Europe/Berlin") + pd.Timedelta(hours=1)).normalize(); rows.append((per, tday, float(((xk - sup(a[j] * 10)) - (xk - sup(fk[j] * 10))).mean())))
T = pd.DataFrame(rows, columns=["per", "day", "miss"]).set_index("day"); J = T.join(dN.add_prefix("ch_"), how="left").join(N.add_prefix("lv_"), how="left"); print(f"\ntarget nights: {len(J)}; the layer's miss on the night's supply shift (true - predicted): sd {J.miss.std():.2f} GW. Correlation of that miss with each block's change since the night before, and the block's own size on those nights:")
for nm, c in BL: m = J[["miss", f"ch_{c}"]].dropna(); print(f"   {nm:28s} correlation {np.corrcoef(m.miss, m[f'ch_{c}'])[0, 1]:+.2f}   sd of its change {m[f'ch_{c}'].std():.2f} GW")
X = J[[f"ch_{c}" for _, c in BL[:7]]].dropna(); y = J.miss.loc[X.index]; A = np.column_stack([X.values, np.ones(len(X))]); be = np.linalg.lstsq(A, y.values, rcond=None)[0]; print(f"   all blocks' changes together explain {100 * (1 - np.var(y.values - A @ be) / np.var(y.values)):.0f}% of the miss")
