"""Test for notes 21.33: is the block-order model's night error the price of not knowing the target day's mean price? Clear the night with the TRUE
day's mean price given (an oracle), everything else as in the forecast (midnight load, learned shift)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M, market1h as Mh
SP = M.SP; th = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/market1h_post.npz").items() if np.ndim(v) == 1}; R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; co2f = pd.Series(np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1), index=R.index).ffill()
for per in ("winter", "summer", "autumn"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float); U = np.load(f"{SP}/out/structfc_m1hm_ts_dyn2xxh.npz")["vint"].astype(float) if per == "autumn" else None; z = np.load(f"{SP}/out/structfc_m1hm_{per}_price.npz"); f0 = z["vint_ts"].astype(float); out = np.zeros((len(a), 24)); dm = []
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; s = np.load(f"{SP}/out/struct_m1hm/{str(c)[:10]}.npz"); x = s["x_ts"].astype(float)[:96]; L = np.full(96, Lj); C = np.full(96, cj)
        # the learned shift is not stored per period; recover it from the forecast itself: the shift that makes the model clear at the forecast price with the forecast's own day mean
        pf = f0[j, :96] * 10; eps = x - (Mh.supply(th, pf, L, C, pf.mean()) - th["K_s"] * (pf < -M.EPS) - 0.5 * th["K_s"] * (np.abs(pf) <= M.EPS)); pt = a[j, :96].mean() * 10; out[j] = Mh.clear_h(th, x[:24], L[:24], C[:24], pt, eps=eps[:24]) / 10; dm.append(pf.mean() / 10 - pt / 10)
    e = lambda v: np.abs(v - a[:, :24]).mean(); m1 = np.load(f"{SP}/out/structfc_m1m_{per}_price.npz")["vint_ts"].astype(float)[:, :24]
    print(f"{per}: night MAE ct: merit order {e(m1):.2f} | with block orders {e(f0[:, :24]):.2f} | block orders with the TRUE day's mean price given {e(out):.2f} | error of the forecast day's mean price: MAE {np.mean(np.abs(dm)):.2f}, bias {np.mean(dm):+.2f} ct")
