"""Per-origin forecast inputs for the structural model (live setup): Europe-wide weather as forecast, gas at issue,
own and published residual load. usage: prep_inputs.py NAME START END STRIDE GENFILE"""
import sys, numpy as np, pandas as pd
import live_common as lc, dense_common as dc
name, start, end, stride, genfile = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
F = lc.load(); V = lc.vintage(F); D = F["DE"]; idx = D.index; cuts = lc.cutoffs(idx, start, end, stride); fc = dc.tso(idx)
cols = {z: [c for c in F[z].columns if c != "price"] for z in lc.ZONES}; EU = ["eu_wind", "eu_wind_northsea", "eu_irradiance", "eu_temperature"]
g = np.load(f"{lc.SP}/out/{genfile}.npz") if genfile != "none" else None
out = {k: [] for k in ("eu", "gas", "rl_true", "wind_true", "solar_true", "load_true")}
for d0 in cuts:
    Wte = {z: lc.test_matrix(F, V, z, d0, cols[z], __import__("os").environ.get("VINT", "1") == "1") for z in lc.ZONES}; eu = lc.eu_feats(Wte)
    out["eu"].append(eu[EU].to_numpy(float)); out["gas"].append(D["gasprice"].to_numpy()[d0 - lc.ISSUE])
    for c in ("rl", "wind", "solar", "load"): out[f"{c}_true"].append(fc[c].to_numpy()[d0:d0 + lc.HZ])
R = {k: np.stack(v).astype(np.float32) for k, v in out.items()}
if g is not None:
    assert len(g["cutoffs"]) == len(cuts)
    for c in ("rl", "wind", "solar", "load"): R[f"{c}_pes"] = g[f"{c}_pes"].astype(np.float32)
np.savez_compressed(f"{lc.SP}/out/inputs_{name}.npz", cuts=np.array(cuts), cutoffs=idx[cuts].tz_convert(None).to_numpy(), actual=np.stack([D["price"].to_numpy()[d0:d0 + lc.HZ] for d0 in cuts]).astype(np.float32), **R)
print(name, len(cuts), "origins", flush=True)
