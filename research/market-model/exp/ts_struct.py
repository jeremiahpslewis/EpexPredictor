"""Timestral price model with the structural model's forecast as a covariate (live setup).
Context (85 days of prices through D+1): the structural curve's values for those days from the fit of this forecast
(published residual load; D+1 from the own forecast of its residual load). Horizon: the structural forecast for the arm.
So the foundation model sees what the fundamentals say and learns from the context how the price deviates from it.
usage: ts_struct.py NAME PERIOD RUN ARM MODE    MODE st: weather covariates as in the base model + structural; st0: structural, gas, holiday only"""
import os, sys, time
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
import live_common as lc
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
name, period, run, arm, mode = sys.argv[1:6]; CTX = 8192
F = lc.load(); V = lc.vintage(F); idx = F["DE"].index; I = np.load(f"{lc.SP}/out/inputs_{period}.npz"); cuts = I["cuts"]; tags = [str(c)[:10] for c in I["cutoffs"]]
cols = {z: [c for c in F[z].columns if c != "price"] for z in lc.ZONES}
y = F["DE"]["price"].to_numpy(float); gas = F["DE"]["gasprice"].to_numpy(float); wd = F["DE"]["weekday"].to_numpy(float)
def covs(W, eu):
    w = W[[c for c in W.columns if c.startswith("wind_")]]
    return np.vstack([w.iloc[:, :6].mean(axis=1), w.iloc[:, 6:].mean(axis=1), ((w.clip(upper=45) / 45) ** 3).mean(axis=1), eu["eu_wind"], eu["eu_wind_northsea"],
                      W[[c for c in W.columns if c.startswith("irradiance_")]].mean(axis=1), eu["eu_irradiance"], W[[c for c in W.columns if c.startswith("temp_")]].mean(axis=1), (W["holiday"] >= 0.5).astype(float)]).astype(float)
hist = covs(F["DE"], F["DE"]); f = TimestralForecaster.from_pretrained("mistralai/Timestral-123M", revision="5a147287addda85f6acd8da3fb8858fe9af7e8f7")
OUT = f"{lc.SP}/out/tsst_{name}"; os.makedirs(OUT, exist_ok=True); t0 = time.time()
for j, d0 in enumerate(cuts):
    d0 = int(d0); tag = tags[j]
    if os.path.exists(f"{OUT}/{tag}.npz"): continue
    z = np.load(f"{lc.SP}/out/struct_{run}/{tag}.npz"); a, b = d0 - CTX, d0 + lc.HZ
    ph = np.repeat(z["PH1"].astype(float).ravel(), 4) / 10                      # ends at d0 when the D+1 row is there, one day earlier otherwise
    if len(z["R1"]) == len(z["R"]): ph = np.concatenate([ph, np.full(96, np.nan)])
    s = pd.Series(np.concatenate([ph[-CTX:], z[f"{arm}_struct"].astype(float)])).interpolate(limit_direction="both").to_numpy()
    g = gas[a:b].copy(); g[CTX - lc.ISSUE:] = gas[d0 - lc.ISSUE]
    if mode == "st":
        k = hist[:, a:b].copy(); Wte = {zz: lc.test_matrix(F, V, zz, d0, cols[zz], True) for zz in lc.ZONES}; eu = lc.eu_feats(Wte); k[:, CTX:] = covs(Wte["DE"], eu); known = np.concatenate([k, g[None], s[None]])
    else: known = np.vstack([s, g, hist[8, a:b]])
    with torch.no_grad():
        r = f.forecast(y[a:d0][None, None, :], prediction_length=lc.HZ, categorical_known_covariates=wd[a:b][None, None, :], known_covariates=known[None])
    np.savez_compressed(f"{OUT}/{tag}.npz", median=np.asarray(r.median)[0, 0].astype(np.float32), **{f"q{int(q * 100):02d}": np.asarray(r.quantile(q))[0, 0].astype(np.float32) for q in (0.05, 0.1, 0.9, 0.95)})
    print(tag, f"{time.time() - t0:.0f}s", flush=True)
med = np.stack([np.load(f"{OUT}/{t}.npz")["median"] for t in tags])
np.savez_compressed(f"{lc.SP}/out/tsst_{name}.npz", cutoffs=I["cutoffs"], median=med, **{q: np.stack([np.load(f"{OUT}/{t}.npz")[q] for t in tags]) for q in ("q05", "q10", "q90", "q95")})
print("finished", flush=True)
