"""Timestral alone in the live setup, with different covariate sets. usage: ts_live3.py NAME MODE
MODE cal: price history + weekday only; base: weather covariates as before; rl: base + residual load; rlc: residual load, its components, holiday, gas"""
import sys, time
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
import live_common as lc, dense_common as dc
name, mode = sys.argv[1], sys.argv[2]; CTX = 8192; DRL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 1
F = lc.load(); V = lc.vintage(F); idx = F["DE"].index; import os; cuts = lc.cutoffs(idx, os.environ.get("TS_START", "2026-09-03"), os.environ.get("TS_END", "2026-11-07"), 1); JS = list(range(0, len(cuts), STRIDE))
cols = {z: [c for c in F[z].columns if c != "price"] for z in lc.ZONES}
y = F["DE"]["price"].to_numpy(float); gas = F["DE"]["gasprice"].to_numpy(float); wd = F["DE"]["weekday"].to_numpy(float)
fc = dc.tso(idx); NEEDG = mode in ("rl", "rlc")
G = np.load(f"{lc.SP}/out/gen_fc_app.npz") if NEEDG else None
if NEEDG: assert len(G["cutoffs"]) == len(cuts)
def covs(W, eu):
    w = W[[c for c in W.columns if c.startswith("wind_")]]
    return np.vstack([w.iloc[:, :6].mean(axis=1), w.iloc[:, 6:].mean(axis=1), ((w.clip(upper=45) / 45) ** 3).mean(axis=1), eu["eu_wind"], eu["eu_wind_northsea"],
                      W[[c for c in W.columns if c.startswith("irradiance_")]].mean(axis=1), eu["eu_irradiance"], W[[c for c in W.columns if c.startswith("temp_")]].mean(axis=1), (W["holiday"] >= 0.5).astype(float)]).astype(float)
hist = covs(F["DE"], F["DE"]); f = TimestralForecaster.from_pretrained("mistralai/Timestral-123M")
out = {k: [] for k in ("median", "q10", "q90", "q05", "q95")}; t0 = time.time()
for j in JS:
    d0 = cuts[j]
    a, b = d0 - CTX, d0 + lc.HZ
    g = gas[a:b].copy(); g[CTX - lc.ISSUE:] = gas[d0 - lc.ISSUE]
    k = hist[:, a:b].copy()
    Wte = {z: lc.test_matrix(F, V, z, d0, cols[z], True) for z in lc.ZONES}; eu = lc.eu_feats(Wte); k[:, CTX:] = covs(Wte["DE"], eu)
    rows = {"rl": fc["rl"].to_numpy()[a:b].copy(), "wind": fc["wind"].to_numpy()[a:b].copy(), "solar": fc["solar"].to_numpy()[a:b].copy(), "load": fc["load"].to_numpy()[a:b].copy()}
    for c in (rows if NEEDG else []): rows[c][CTX:] = G[f"{c}_pes"][j]                      # own forecast from 2-3 day old weather for the forecast days
    rows["rl"][CTX:] += DRL
    if mode == "cal": known = None
    elif mode == "base": known = np.concatenate([k, g[None]])
    elif mode == "rl": known = np.concatenate([k, g[None], rows["rl"][None]])
    elif mode == "rlc": known = np.vstack([rows["rl"], rows["wind"], rows["solar"], rows["load"], k[8], g])
    kw = {} if known is None else {"known_covariates": known[None]}
    with torch.no_grad():
        r = f.forecast(y[a:d0][None, None, :], prediction_length=lc.HZ, categorical_known_covariates=wd[a:b][None, None, :], **kw)
    out["median"].append(np.asarray(r.median)[0, 0])
    for q in (0.1, 0.9, 0.05, 0.95): out[f"q{int(q*100):02d}"].append(np.asarray(r.quantile(q))[0, 0])
np.savez_compressed(f"{lc.SP}/out/{name}.npz", cutoffs=idx[[cuts[j] for j in JS]].tz_convert(None).to_numpy(), **{k: np.stack(v).astype(np.float32) for k, v in out.items()})
print(name, len(cuts), "cutoffs in %.0fs" % (time.time() - t0), flush=True)
