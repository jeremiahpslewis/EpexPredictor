"""Control for notes 21.18: the plain curve (level 1: price = L k(published residual load), frozen at November 2025) through the candidate's layer,
once learning the error in price (run l1p) and once as a shift along the load axis (run l1g). Same slots, same information as mm_layer.py AXIS=pub."""
import sys, os, subprocess, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; REFIT = os.environ.get("REFIT", "0") == "1"; SUF = "_r" if REFIT else ""
def curve(f):
    Z = np.load(f); K = Z["K"].astype(float); kk = np.median(Z["kn"], 0); kk = np.maximum.accumulate(kk) + 1e-6 * np.arange(len(kk))
    return (lambda x: np.interp(x, K, kk) + np.where(x > K[-1], (x - K[-1]) * (kk[-1] - kk[-2]) / (K[-1] - K[-2]), 0.0) + np.where(x < K[0], (x - K[0]) * (kk[1] - kk[0]) / (K[1] - K[0]), 0.0)), (lambda y: np.interp(y, kk, K))
k, kinv = curve(f"{SP}/out/zoom1_post.npz")
if REFIT:
    import glob; FITS = sorted((pd.Timestamp(f.split("_r")[-1][:8]).tz_localize("Europe/Berlin"), curve(f)) for f in glob.glob(f"{SP}/out/zoom1_r*_post.npz"))
S = pd.read_parquet(f"{SP}/out/star_data.parquet")[["rl_de"]]; S.index = S.index.tz_convert("UTC"); R = pd.read_parquet(f"{SP}/out/market0d_rows.parquet"); R.index = R.index.tz_convert("UTC"); full = pd.date_range(R.index.min().floor("D"), R.index.max().ceil("D"), freq="1h", tz="UTC"); G = R.reindex(full); co2 = G.co2.ffill(); NH = 200
CR = {"winter": "hw2", "summer": "hs2", "autumn": "ha"}; TS = {"winter": ("win_ts", "median"), "summer": ("ssn_ts", "median"), "autumn": ("tsl_123_8k", "vint")}
RUNS = ("l1p" + SUF, "l1g" + SUF)
for run in RUNS: os.makedirs(f"{SP}/out/struct_{run}", exist_ok=True)
for per in ("winter", "summer", "autumn"):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); D1 = np.load(f"{SP}/out/d1_ts_{per}.npz"); GEN = np.load(f"{SP}/out/gen_ts_{per}.npz"); a = I["actual"].astype(float); LJ = []; KJ = []
    for j, c in enumerate(I["cutoffs"]):
        t0 = pd.Timestamp(str(c)).tz_localize("UTC"); tag = str(c)[:10]
        if REFIT: k, kinv = [f for f in FITS if f[0] <= (t0 - pd.Timedelta(hours=34)).tz_convert("Europe/Berlin").normalize()][-1][1]
        KJ.append(k); Lj = 1.85 * float(I["gas"][j]) + 0.37 * float(co2.asof(t0 - pd.Timedelta(hours=34))); LJ.append(Lj); tt = t0 + pd.Timedelta(minutes=15) * np.arange(153); xk = S.rl_de.reindex(tt).ffill().bfill().values; xt = GEN["rl_pes"][j].astype(float)
        hh = pd.date_range(t0 - pd.Timedelta(hours=24 * NH), t0 - pd.Timedelta(hours=1), freq="1h"); g = G.reindex(hh); P = g.p.values.reshape(NH, 24); X = g.rl.values.reshape(NH, 24).copy(); Lh = g.L.values.reshape(NH, 24).copy(); X[-1] = D1["rl_d1"][j].astype(float).reshape(24, 4).mean(1); Lh[-1] = Lj; ok = ~np.isnan(P + X + Lh)
        Rp = np.where(ok, P - Lh * k(np.where(ok, X, 30.0)), np.nan); u = np.where(ok, X - kinv(np.where(ok, P / Lh, 1.0)), np.nan)
        np.savez_compressed(f"{SP}/out/struct_{RUNS[0]}/{tag}.npz", R1=Rp.astype(np.float32), XH1=X.astype(np.float32), x_known=xk.astype(np.float32), known_shape=(Lj * k(xk) / 10).astype(np.float32), x_ts=xt.astype(np.float32), ts_shape=(Lj * k(xt) / 10).astype(np.float32))
        np.savez_compressed(f"{SP}/out/struct_{RUNS[1]}/{tag}.npz", R1=(10 * u).astype(np.float32), XH1=X.astype(np.float32), x_known=xk.astype(np.float32), known_shape=np.zeros(153, np.float32), x_ts=xt.astype(np.float32), ts_shape=np.zeros(153, np.float32))
    sc = lambda v: f"{np.abs(v - a)[:, :32].mean():.2f} / {np.abs(v - a)[:, :56].mean():.2f}"; out = {}; LJ = np.array(LJ)[:, None]
    for run in RUNS:
        subprocess.run([sys.executable, "dyn2.py", per, run, "known,ts", "x+xh"], capture_output=True)
        for arm in ("known", "ts"):
            z = np.load(f"{SP}/out/structfc_{run}_{arm}_dyn2xxh.npz"); v = z["vint"].astype(float)
            if run == RUNS[1]: x = np.stack([np.load(f"{SP}/out/struct_{run}/{str(c)[:10]}.npz")[f"x_{arm}"].astype(float) for c in I["cutoffs"]]); v = np.stack([LJ[j, 0] * KJ[j](x[j] - v[j]) / 10 for j in range(len(x))])
            out[(run, arm)] = v; np.savez_compressed(f"{SP}/out/structfc_{run}_{per}_{arm}.npz", vint=v.astype(np.float32))
    bare = np.stack([np.load(f"{SP}/out/struct_{RUNS[0]}/{str(c)[:10]}.npz")["known_shape"].astype(float) for c in I["cutoffs"]]); tp = np.load(f"{SP}/out/{TS[per][0]}.npz")[TS[per][1]].astype(float)
    print(f"{per}: plain curve bare {sc(bare)} | PUBLISHED LOAD GIVEN: layer in price {sc(out[(RUNS[0], 'known')])}, shift along the load axis {sc(out[(RUNS[1], 'known')])} | TIMESTRAL LOAD: layer in price {sc(out[(RUNS[0], 'ts')])}, along the load axis {sc(out[(RUNS[1], 'ts')])}; 50/50 with Timestral price {sc(0.5 * (out[(RUNS[0], 'ts')] + tp))}, {sc(0.5 * (out[(RUNS[1], 'ts')] + tp))}")
