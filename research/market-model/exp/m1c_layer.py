"""The layer in GW on the merit-order model M1 (notes 21.27): as mm_layer.py AXIS=pub gw, with fuel costs from gas and CO2 separately.
Each known price is inverted into the supply shift it implies; dyn2.py learns it by demand and hour; the market is cleared again. usage: m1_layer.py RUN"""
import sys, os, subprocess, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M, market1h as Mh
os.environ.setdefault("DAYRE", "1"); import market1c as Mc
NH = 200; PZ = np.load(f"{M.SP}/out/market1cmbr_post.npz"); HC, _ = Mc.frame(); HC.index = HC.index.tz_convert("UTC"); ZC = {"l": "avs_z", "c": "avs_z", "g": "avg_z"}; STT = PZ["st_eu_temperature"]
def shift(av, tz, we): return {f: th[f"bA_{f}"] * av[f] + th[f"bT_{f}"] * tz + th[f"bW_{f}"] * we for f in M.FU}
MID = os.environ.get("MID", "0") == "1"   # midnight issue: published load of the day before the target is known; fresher load forecast (gen_mid)
SP = M.SP; RUN = sys.argv[1]; th = {k: float(np.median(v)) for k, v in np.load(f"{SP}/out/market1cmbr_post.npz").items() if np.ndim(v) == 1}
def clr(D, L, c, eps=0.0, E_=None):   # the target day (96 quarters) and the start of the next (57) each share a mean price
    return np.concatenate([Mh.clear_day(th, D[None, a:b], L[None, a:b], c[None, a:b], np.ones((1, b - a), bool), eps=(eps[None, a:b] if np.ndim(eps) else 0.0), e={f: v_[None, a:b] for f, v_ in E_.items()})[0][0] for a, b in ((0, 96), (96, 153))]); NH = 200
S = pd.read_parquet(f"{SP}/out/star_data.parquet")[["rl_de"]]; S.index = S.index.tz_convert("UTC"); R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; R["co2"] = np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1); R["L"] = 1.85 * R.gas + 0.37 * R.co2
full = pd.date_range(R.index.min().floor("D"), R.index.max().ceil("D"), freq="1h", tz="UTC"); G = R.reindex(full); co2f = G.co2.ffill(); os.makedirs(f"{SP}/out/struct_{RUN}", exist_ok=True)
CR = {"winter": "hw2", "summer": "hs2", "autumn": "ha"}; TS = {"winter": ("win_ts", "median"), "summer": ("ssn_ts", "median"), "autumn": ("tsl_123_8k", "vint")}
for per in ("winter", "summer", "autumn"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); D1 = np.load(f"{SP}/out/d1_ts_{per}.npz"); GEN = np.load(f"{SP}/out/gen_{'mid' if MID else 'ts'}_{per}.npz"); a = I["actual"].astype(float); meta = []
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); tag = str(c)[:10]; cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; tt = t0 + pd.Timedelta(minutes=15) * np.arange(153)
        hh = pd.date_range(t0 - pd.Timedelta(hours=24 * NH), t0 - pd.Timedelta(hours=1), freq="1h"); g = G.reindex(hh); P = g.p.values.reshape(NH, 24); oth = np.nanmean(g.other.values.reshape(NH, 24)[-9:-2], 0); X = (g.rl.values - g.other.values).reshape(NH, 24).copy(); X[-2] = g.rl.values.reshape(NH, 24)[-2] - oth; X[-1] = (g.rl.values.reshape(NH, 24)[-1] if MID else D1["rl_d1"][j].astype(float).reshape(24, 4).mean(1)) - oth
        Lh = g.L.values.reshape(NH, 24).copy(); Ch = g.co2.values.reshape(NH, 24).copy(); Lh[-1] = Lj; Ch[-1] = cj; ok = ~np.isnan(P + X + Lh + Ch); wd = th["K_s"] * (P < -M.EPS) + 0.5 * th["K_s"] * (np.abs(P) <= M.EPS); u = np.full((NH, 24), np.nan); pb = np.nanmean(P, 1, keepdims=True) * np.ones((1, 24)); pb = np.where(np.isnan(pb), 80.0, pb); hc = HC.reindex(hh); cv = lambda c_: np.nan_to_num(hc[c_].values.reshape(NH, 24)); eh = shift({f: cv(ZC[f]) for f in M.FU}, cv("eu_temperature_z"), cv("we")); u[ok] = (X - (Mh.supply(th, np.where(ok, P, 50.0), np.where(ok, Lh, 90.0), np.where(ok, Ch, 70.0), pb, eh) - wd))[ok]
        hod = (np.arange(153) // 4) % 24; et = shift({f: cv(ZC[f])[-1][hod] for f in M.FU}, (I["eu"][j, :, 3].astype(float) - STT[0]) / STT[1], (np.asarray(pd.DatetimeIndex(tt).tz_convert("Europe/Berlin").dayofweek) >= 5).astype(float)); xk = S.rl_de.reindex(tt).ffill().bfill().values - oth[hod]; xt = GEN["rl_pes"][j].astype(float) - oth[hod]; meta.append((Lj, cj, xk, xt, et))
        np.savez_compressed(f"{SP}/out/struct_{RUN}/{tag}.npz", R1=(10 * u).astype(np.float32), XH1=X.astype(np.float32), x_known=xk.astype(np.float32), known_shape=np.zeros(153, np.float32), x_ts=xt.astype(np.float32), ts_shape=np.zeros(153, np.float32))
    subprocess.run([sys.executable, "dyn2.py", per, RUN, "known,ts", "x+xh"], capture_output=True); out = {}
    for arm, k in (("known", 2), ("ts", 3)):
        U = np.load(f"{SP}/out/structfc_{RUN}_{arm}_dyn2xxh.npz")["vint"].astype(float); out[arm] = np.stack([clr(m_[k], np.full(153, m_[0]), np.full(153, m_[1]), U[j], m_[4]) / 10 for j, m_ in enumerate(meta)])
    bare = np.stack([clr(m_[2], np.full(153, m_[0]), np.full(153, m_[1]), 0.0, m_[4]) / 10 for m_ in meta]); np.savez_compressed(f"{SP}/out/structfc_{RUN}_{per}_price.npz", vint=out["known"].astype(np.float32), vint_ts=out["ts"].astype(np.float32), bare=bare.astype(np.float32))
    sc = lambda v: f"{np.abs(v - a)[:, :32].mean():.2f} / {np.abs(v - a)[:, :56].mean():.2f}"; tp = np.load(f"{SP}/out/{TS[per][0]}.npz")[TS[per][1]].astype(float); ct = np.load(f"{SP}/out/structfc_{CR[per]}_ts_dyn2xxh.npz")["vint"].astype(float)
    print(f"{per}: M1 with covariates and day shocks, bare (published load) {sc(bare)} | + layer, published load {sc(out['known'])} | + layer, Timestral load {sc(out['ts'])} | candidate's curve + layer {sc(ct)} | M1 + Timestral 50/50 {sc(0.5 * (out['ts'] + tp))} | candidate {sc(0.5 * (ct + tp))}")
