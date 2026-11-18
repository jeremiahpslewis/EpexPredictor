"""Timestral for the neighbours' residual load (sum of AT, BE, CH, CZ, DK, FR, NL, PL), live setup.
Published figures known through the issue day D; horizon D+1, D+2 and the start of D+3. Covariates: Europe-wide weather
(archived in the context, as forecast in the horizon), weekday. usage: ts_nb.py PERIOD"""
import os, sys, time
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
import live_common as lc
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
period = sys.argv[1]; CTX, LEAD = 8192, 96; OUT = f"{lc.SP}/out/tsnb_{period}"; os.makedirs(OUT, exist_ok=True)
F = lc.load(); V = lc.vintage(F); D = F["DE"]; idx = D.index; I = np.load(f"{lc.SP}/out/inputs_{period}.npz"); cuts = I["cuts"]
EUC = ["eu_wind", "eu_wind_northsea", "eu_irradiance", "eu_temperature"]; cols = {z: [c for c in F[z].columns if c != "price"] for z in lc.ZONES}
nb = pd.read_parquet(f"{lc.SP}/out/nb_rl.parquet"); y = nb[[c for c in nb.columns if c.startswith("rl_")]].sum(axis=1, min_count=8).to_numpy(float)
eu_hist = D[EUC].to_numpy(float).T; wd = D["weekday"].to_numpy(float)
f = TimestralForecaster.from_pretrained("mistralai/Timestral-123M", revision="5a147287addda85f6acd8da3fb8858fe9af7e8f7"); t0 = time.time()
for j, d0 in enumerate(cuts):
    d0 = int(d0); tag = str(idx[d0].date())
    if os.path.exists(f"{OUT}/{tag}.npz"): continue
    e = d0 - LEAD; a, b = e - CTX, d0 + lc.HZ
    cov = eu_hist[:, a:b].copy()
    W1 = {z: lc.test_matrix(F, V, z, e, cols[z], True) for z in lc.ZONES}; cov[:, CTX:CTX + LEAD] = lc.eu_feats(W1)[EUC].to_numpy(float)[:LEAD].T     # D+1, as forecast
    cov[:, CTX + LEAD:] = I["eu"][j].astype(float).T                                                                                                # D+2 on, as forecast
    with torch.no_grad():
        r = f.forecast(y[a:e][None, None, :], prediction_length=b - e, known_covariates=cov[None], categorical_known_covariates=wd[a:b][None, None, :])
    med = np.asarray(r.median)[0, 0]
    np.savez_compressed(f"{OUT}/{tag}.npz", d1=med[:LEAD], pes=med[LEAD:], d1_true=y[e:d0], true=y[d0:b])
    if j % 10 == 0: print(tag, f"{time.time() - t0:.0f}s", flush=True)
print("finished", flush=True)
