"""The layer in GW on the merit-order model M1 (notes 21.27): as mm_layer.py AXIS=pub gw, with fuel costs from gas and CO2 separately.
Each known price is inverted into the supply shift it implies; dyn2.py learns it by demand and hour; the market is cleared again. usage: m1_layer.py RUN"""
import sys, os, subprocess, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import market1 as M
MID = os.environ.get("MID", "0") == "1"   # midnight issue: published load of the day before the target is known; fresher load forecast (gen_mid)
NBMODE = os.environ.get("NBX", "0"); NBX = NBMODE == "1"   # NBX=model: M1n, neighbours inside the fit, target values given (oracle); NBX=carry: M1n, target values = same hour of the last known day   # neighbours' published residual loads shift net imports (coefficients from the fitting period); ORACLE for the target
SP = M.SP; RUN = sys.argv[1]; NBF = NBMODE in ("model", "carry", "fc"); _P = np.load(f"{SP}/out/market1{'n' if NBF else ''}_post.npz"); th = {k: float(np.median(_P[k])) for k in M.PN}; NH = 200
S = pd.read_parquet(f"{SP}/out/star_data.parquet")[["rl_de"]]; S.index = S.index.tz_convert("UTC"); R = pd.read_parquet(f"{SP}/out/market_hourly.parquet"); E = np.load(f"{SP}/out/eua_15min.npz")["eua"]; E = np.r_[E, np.full(max(0, len(R) * 4 - len(E)), np.nan)]; R["co2"] = np.nanmean(E[: len(R) * 4].reshape(-1, 4), 1); R["L"] = 1.85 * R.gas + 0.37 * R.co2
NBC = [f"rl_{c}" for c in ("fr", "nl", "be", "at", "pl", "cz", "ch", "dk")]; SN = pd.read_parquet(f"{SP}/out/star_data.parquet")[NBC]; SN.index = SN.index.tz_convert("UTC"); SNh = SN.resample("1h").mean()
if NBF: NBM = _P["NMU"]; NBB = np.median(_P["kn"], 0) / _P["NSD"]; NBX = True; print("M1n: net imports per GW of each neighbour's residual load:", np.round(NBB, 2))
elif NBX:
    R1_ = pd.read_parquet(f"{SP}/out/market1_rows.parquet"); R1_ = R1_[R1_.tr]; R1_.index = R1_.index.tz_convert("UTC"); JJ = R1_[["eX"]].join(SNh, how="inner").dropna(); NBM = JJ[NBC].mean().values; A_ = np.column_stack([JJ[NBC].values - NBM, np.ones(len(JJ))]); NBB = np.linalg.lstsq(A_, JJ.eX.values, rcond=None)[0][:8]; print("net imports per GW of each neighbour's residual load:", np.round(NBB, 2))
full = pd.date_range(R.index.min().floor("D"), R.index.max().ceil("D"), freq="1h", tz="UTC"); G = R.reindex(full); co2f = G.co2.ffill(); os.makedirs(f"{SP}/out/struct_{RUN}", exist_ok=True)
CR = {"winter": "hw2", "summer": "hs2", "autumn": "ha"}; TS = {"winter": ("win_ts", "median"), "summer": ("ssn_ts", "median"), "autumn": ("tsl_123_8k", "vint")}
for per in ("winter", "summer", "autumn"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); D1 = np.load(f"{SP}/out/d1_ts_{per}.npz"); GEN = np.load(f"{SP}/out/gen_{'mid' if MID else 'ts'}_{per}.npz"); a = I["actual"].astype(float); meta = []
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); tag = str(c)[:10]; cj = float(co2f.asof(t0 - pd.Timedelta(hours=34))); Lj = 1.85 * float(I["gas"][j]) + 0.37 * cj; tt = t0 + pd.Timedelta(minutes=15) * np.arange(153)
        hh = pd.date_range(t0 - pd.Timedelta(hours=24 * NH), t0 - pd.Timedelta(hours=1), freq="1h"); g = G.reindex(hh); P = g.p.values.reshape(NH, 24); oth = np.nanmean(g.other.values.reshape(NH, 24)[-9:-2], 0); X = (g.rl.values - g.other.values).reshape(NH, 24).copy(); X[-2] = g.rl.values.reshape(NH, 24)[-2] - oth; X[-1] = (g.rl.values.reshape(NH, 24)[-1] if MID else D1["rl_d1"][j].astype(float).reshape(24, 4).mean(1)) - oth
        Lh = g.L.values.reshape(NH, 24).copy(); Ch = g.co2.values.reshape(NH, 24).copy(); Lh[-1] = Lj; Ch[-1] = cj; ok = ~np.isnan(P + X + Lh + Ch); wd = th["K_s"] * (P < -M.EPS) + 0.5 * th["K_s"] * (np.abs(P) <= M.EPS); u = np.full((NH, 24), np.nan); u[ok] = (X - (M.supply(th, np.where(ok, P, 50.0), np.where(ok, Lh, 90.0), np.where(ok, Ch, 70.0)) - wd))[ok]
        nbt = np.zeros(153)
        if NBX: u = u - np.nan_to_num((SNh.reindex(hh).values - NBM) @ NBB).reshape(NH, 24); nbt = np.nan_to_num((SN.reindex(tt).ffill().bfill().values - NBM) @ NBB)
        if NBMODE == "carry": nbt = np.nan_to_num((SN.reindex(tt - pd.to_timedelta((np.arange(153) // 96 + 1) * 24, unit="h")).ffill().bfill().values - NBM) @ NBB)
        if NBMODE == "fc": nbt = np.nan_to_num((np.load(f"{SP}/out/nbmid{os.environ.get('NBSFX', '2')}_{per}.npz")["fc"][j].astype(float) - NBM) @ NBB)
        hod = (np.arange(153) // 4) % 24; xk = S.rl_de.reindex(tt).ffill().bfill().values - oth[hod]; xt = GEN["rl_pes"][j].astype(float) - oth[hod]; meta.append((Lj, cj, xk, xt, nbt))
        np.savez_compressed(f"{SP}/out/struct_{RUN}/{tag}.npz", R1=(10 * u).astype(np.float32), XH1=X.astype(np.float32), x_known=xk.astype(np.float32), known_shape=np.zeros(153, np.float32), x_ts=xt.astype(np.float32), ts_shape=np.zeros(153, np.float32))
    subprocess.run([sys.executable, "dyn2.py", per, RUN, "known,ts", "x+xh"], capture_output=True); out = {}
    for arm, k in (("known", 2), ("ts", 3)):
        U = np.load(f"{SP}/out/structfc_{RUN}_{arm}_dyn2xxh.npz")["vint"].astype(float); out[arm] = np.stack([M.clear(th, m_[k], np.full(153, m_[0]), np.full(153, m_[1]), eps=U[j] + m_[4]) / 10 for j, m_ in enumerate(meta)])
    bare = np.stack([M.clear(th, m_[2], np.full(153, m_[0]), np.full(153, m_[1]), eps=m_[4]) / 10 for m_ in meta]); np.savez_compressed(f"{SP}/out/structfc_{RUN}_{per}_price.npz", vint=out["known"].astype(np.float32), vint_ts=out["ts"].astype(np.float32), bare=bare.astype(np.float32))
    sc = lambda v: f"{np.abs(v - a)[:, :32].mean():.2f} / {np.abs(v - a)[:, :56].mean():.2f}"; tp = np.load(f"{SP}/out/{TS[per][0]}.npz")[TS[per][1]].astype(float); ct = np.load(f"{SP}/out/structfc_{CR[per]}_ts_dyn2xxh.npz")["vint"].astype(float)
    print(f"{per}: M1 bare (published load) {sc(bare)} | + layer, published load {sc(out['known'])} | + layer, Timestral load {sc(out['ts'])} | candidate's curve + layer {sc(ct)} | M1 + Timestral 50/50 {sc(0.5 * (out['ts'] + tp))} | candidate {sc(0.5 * (ct + tp))}")
