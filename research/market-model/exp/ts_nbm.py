"""MIDNIGHT ISSUE, per neighbour (notes 21.47): Timestral for the published residual load of each of the eight neighbours, jointly as eight variates.
Context to the start of the target day (the published figures for the day before the target are out at midnight); horizon = target day onwards.
Covariates: Europe-wide weather (archived in the context, as forecast in the horizon), weekday. usage: ts_nbm.py PERIOD"""
import os, sys, time
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
import live_common as lc
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
period = sys.argv[1]; CTX = int(os.environ.get("CTX", 4096)); SFX = os.environ.get("SFX", "2"); OUT = f"{lc.SP}/out/tsnbm{SFX}_{period}"; os.makedirs(OUT, exist_ok=True)
F = lc.load(); V = lc.vintage(F); cols = {z: [c for c in F[z].columns if c != "price"] for z in lc.ZONES}; D = F["DE"]; idx = D.index; I = np.load(f"{lc.SP}/out/inputs_{period}.npz"); cuts = I["cuts"]
EUC = ["eu_wind", "eu_wind_northsea", "eu_irradiance", "eu_temperature"]; NBC = [f"rl_{c}" for c in ("fr", "nl", "be", "at", "pl", "cz", "ch", "dk")]
nb = pd.read_parquet(f"{lc.SP}/out/nb_rl.parquet"); assert len(nb) == len(idx); Y = nb[NBC].ffill().to_numpy(float).T; eu_hist = D[EUC].to_numpy(float).T; wd = D["weekday"].to_numpy(float)
f = TimestralForecaster.from_pretrained("mistralai/Timestral-123M", revision="5a147287addda85f6acd8da3fb8858fe9af7e8f7"); t0 = time.time()
for j, d0 in enumerate(cuts):
    d0 = int(d0); tag = str(idx[d0].date())
    if os.path.exists(f"{OUT}/{tag}.npz"): continue
    a, b = d0 - CTX, d0 + lc.HZ; cov = eu_hist[:, a:b].copy(); cov[:, CTX:] = I["eu"][j].astype(float).T
    cov[:, CTX - 96:CTX] = lc.eu_feats({z: lc.test_matrix(F, V, z, d0 - 96, cols[z], True) for z in lc.ZONES})[EUC].to_numpy(float)[:96].T   # the day before the target has not happened at a midnight issue: weather as forecast, not archived (21.49)
    with torch.no_grad():
        r = f.forecast(Y[:, a:d0][None], prediction_length=b - d0, known_covariates=cov[None], categorical_known_covariates=wd[a:b][None, None, :])
    med = np.asarray(r.median)[0]; np.savez_compressed(f"{OUT}/{tag}.npz", fc=med.T.astype(np.float32), true=Y[:, d0:b].T.astype(np.float32))
    if j % 10 == 0: print(tag, f"{time.time() - t0:.0f}s", med.shape, flush=True)
np.savez_compressed(f"{lc.SP}/out/nbmid{SFX}_{period}.npz", fc=np.stack([np.load(f"{OUT}/{str(idx[int(d)].date())}.npz")["fc"] for d in cuts]), true=np.stack([np.load(f"{OUT}/{str(idx[int(d)].date())}.npz")["true"] for d in cuts]))
print("finished", flush=True)
