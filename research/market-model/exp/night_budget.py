"""Error budget of the night forecast (notes 21.45), merit-order model M1 with the layer, midnight load forecast. What would the 00:00-06:00
error be if (a) the load were known, (b) also the night's mean supply shift were known, (c) also its hour-by-hour shape?"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M
SP = M.SP; th = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/market1_post.npz").items() if np.ndim(v) == 1}; R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1), index=R.index).ffill(); N = 24
for per in ("winter", "summer", "autumn"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float)[:, :N]; z = np.load(f"{SP}/out/structfc_m1m_{per}_price.npz"); fts, fk = z["vint_ts"].astype(float)[:, :N], z["vint"].astype(float)[:, :N]; out = {k: np.zeros_like(a) for k in ("lvl", "lvl_hr", "dlev", "dprev")}; du = []; dprev = []
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; s = np.load(f"{SP}/out/struct_m1m/{str(c)[:10]}.npz"); xk = s["x_known"].astype(float)[:N]; L = np.full(N, Lj); C = np.full(N, cj); sup = lambda p: M.supply(th, p, L, C) - th["K_s"] * (p < -M.EPS) - 0.5 * th["K_s"] * (np.abs(p) <= M.EPS)
        ut = xk - sup(a[j] * 10); up = xk - sup(fk[j] * 10); out["lvl"][j] = M.clear(th, xk, L, C, eps=up - up.mean() + ut.mean()) / 10; uh = ut.reshape(6, 4).mean(1).repeat(4); out["lvl_hr"][j] = M.clear(th, xk, L, C, eps=uh) / 10; du.append(up.mean() - ut.mean())
        prev = s["R1"][-1].astype(float)[:6] / 10; dprev.append(np.nanmean(prev) - ut.mean())    # the same hours of the day before the target, whose prices are known
    e = lambda v: np.abs(v - a).mean(); du = np.array(du); dprev = np.array(dprev)
    print(f"{per}: night MAE ct | forecast as is {e(fts):.2f} | load known {e(fk):.2f} | + the night's mean supply shift known {e(out['lvl']):.2f} | + its hourly shape known {e(out['lvl_hr']):.2f} (what is left is within the hour) || error of the layer's night shift: MAE {np.abs(du).mean():.2f} GW, bias {du.mean():+.2f}; simply taking last night's shift: MAE {np.nanmean(np.abs(dprev)):.2f} GW; sd of the true night-to-night change {np.nanstd(dprev):.2f} GW")
