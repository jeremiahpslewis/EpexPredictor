"""Timestral-28M in the live setup (see live_common.py). usage: ts_live.py NAME START END STRIDE"""
import sys, time
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
import live_common as lc
name, start, end, stride = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); CTX = int(sys.argv[6]); MODEL = sys.argv[5]
F = lc.load(); V = lc.vintage(F); idx = F["DE"].index; cuts = lc.cutoffs(idx, start, end, stride)
cols = {z: [c for c in F[z].columns if c != "price"] for z in lc.ZONES}
y = F["DE"]["price"].to_numpy(float); gas = F["DE"]["gasprice"].to_numpy(float)
def covs(W, eu):
    w = W[[c for c in W.columns if c.startswith("wind_")]]
    return np.vstack([w.iloc[:, :6].mean(axis=1), w.iloc[:, 6:].mean(axis=1), ((w.clip(upper=45) / 45) ** 3).mean(axis=1), eu["eu_wind"], eu["eu_wind_northsea"],
                      W[[c for c in W.columns if c.startswith("irradiance_")]].mean(axis=1), eu["eu_irradiance"], W[[c for c in W.columns if c.startswith("temp_")]].mean(axis=1), (W["holiday"] >= 0.5).astype(float)]).astype(float)
hist = covs(F["DE"], F["DE"]); wd = F["DE"]["weekday"].to_numpy(float)
f = TimestralForecaster.from_pretrained(MODEL); out = {"vint": [], "vint_q90": [], "vint_q10": []}; t0 = time.time()
for d0 in cuts:
    a, b = d0 - CTX, d0 + lc.HZ
    for tag in ("vint",):
        k = hist[:, a:b].copy()
        if tag == "vint":
            Wte = {z: lc.test_matrix(F, V, z, d0, cols[z], True) for z in lc.ZONES}; eu = lc.eu_feats(Wte); k[:, CTX:] = covs(Wte["DE"], eu)
        g = gas[a:b].copy(); g[CTX - lc.ISSUE:] = gas[d0 - lc.ISSUE]
        with torch.no_grad():
            fc = f.forecast(y[a:d0][None, None, :], prediction_length=lc.HZ, known_covariates=np.concatenate([k, g[None]])[None], categorical_known_covariates=wd[a:b][None, None, :])
        out[tag].append(np.asarray(fc.median)[0, 0])
        if tag == "vint": out["vint_q90"].append(np.asarray(fc.quantile(0.9))[0, 0]); out["vint_q10"].append(np.asarray(fc.quantile(0.1))[0, 0])
np.savez_compressed(f"{lc.SP}/out/{name}.npz", cutoffs=idx[cuts].tz_convert(None).to_numpy(), **{k: np.stack(v).astype(np.float32) for k, v in out.items()})
print(name, len(cuts), "cutoffs in %.0fs" % (time.time() - t0), flush=True)
