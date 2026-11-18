"""Timestral for the neighbours' residual load, zone by zone, each driven by its own weather (live setup).
Zones AT, BE, CH, CZ, DK, FR, NL, PL as one batch per forecast. Published figures known through the issue day D; horizon
D+1, D+2 and the start of D+3. Covariates per zone: wind (cubed, capped), irradiance and temperature of that zone's
weather points (CH: Austria's; CZ, PL: Germany's, as the data has no points there), the four Europe-wide aggregates,
weekday. Weather archived in the context, as forecast at the issue time in the horizon.
usage: ts_nbz.py PERIOD [SHARD NSHARDS]"""
import os, sys, time
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
import live_common as lc
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
period = sys.argv[1]; SH, NSH = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1); CTX, LEAD = int(os.environ.get("CTX", 8192)), 96; OUT = f"{lc.SP}/out/tsnbz_{period}"; os.makedirs(OUT, exist_ok=True)
F = lc.load(); V = lc.vintage(F); D = F["DE"]; idx = D.index; I = np.load(f"{lc.SP}/out/inputs_{period}.npz"); cuts = I["cuts"]
EUC = ["eu_wind", "eu_wind_northsea", "eu_irradiance", "eu_temperature"]; cols = {z: [c for c in F[z].columns if c != "price"] for z in lc.ZONES}
nb = pd.read_parquet(f"{lc.SP}/out/nb_rl.parquet"); ZN = ["at", "be", "ch", "cz", "dk", "fr", "nl", "pl"]; Y = np.stack([nb[f"rl_{c}"].to_numpy(float) for c in ZN])
SRC = {"at": ["AT"], "be": ["BE"], "ch": ["AT"], "cz": ["DE"], "dk": ["DK1", "DK2"], "fr": ["FR"], "nl": ["NL"], "pl": ["DE"]}
def zfeat(W, zs):
    """wind (cubed, capped), irradiance, temperature: means over the weather points of the zones zs; W: zone -> frame"""
    w = pd.concat([W[z][[c for c in W[z].columns if c.startswith("wind_")]] for z in zs], axis=1).clip(lower=0, upper=45.0)
    return np.vstack([((w / 45.0) ** 3).mean(axis=1), pd.concat([W[z][[c for c in W[z].columns if c.startswith("irradiance_")]] for z in zs], axis=1).mean(axis=1), pd.concat([W[z][[c for c in W[z].columns if c.startswith("temp_")]] for z in zs], axis=1).mean(axis=1)]).astype(float)
HIST = {c: zfeat(F, SRC[c]) for c in ZN}; eu_hist = D[EUC].to_numpy(float).T; wd = D["weekday"].to_numpy(float)
f = TimestralForecaster.from_pretrained("mistralai/Timestral-123M", revision="5a147287addda85f6acd8da3fb8858fe9af7e8f7"); t0 = time.time()
for j, d0 in enumerate(cuts):
    d0 = int(d0); tag = str(idx[d0].date())
    if j % NSH != SH or os.path.exists(f"{OUT}/{tag}.npz"): continue
    e = d0 - LEAD; a, b = e - CTX, d0 + lc.HZ
    W1 = {z: lc.test_matrix(F, V, z, e, cols[z], True) for z in lc.ZONES}; W2 = {z: lc.test_matrix(F, V, z, d0, cols[z], True) for z in lc.ZONES}
    eu = eu_hist[:, a:b].copy(); eu[:, CTX:CTX + LEAD] = lc.eu_feats(W1)[EUC].to_numpy(float)[:LEAD].T; eu[:, CTX + LEAD:] = I["eu"][j].astype(float).T
    covs = []
    for c in ZN:
        k = HIST[c][:, a:b].copy(); k[:, CTX:CTX + LEAD] = zfeat(W1, SRC[c])[:, :LEAD]; k[:, CTX + LEAD:] = zfeat(W2, SRC[c]); covs.append(np.vstack([k, eu]))
    cov = np.stack(covs); cov = np.where(np.isfinite(cov), cov, np.nanmean(cov, axis=2, keepdims=True))
    with torch.no_grad():
        r = f.forecast(Y[:, None, a:e], prediction_length=b - e, known_covariates=cov, categorical_known_covariates=np.broadcast_to(wd[a:b], (len(ZN), 1, b - a)).copy())
    med = np.asarray(r.median)[:, 0]
    np.savez_compressed(f"{OUT}/{tag}.npz", zones=np.array(ZN), d1=med[:, :LEAD].sum(0), pes=med[:, LEAD:].sum(0), d1_true=Y[:, e:d0].sum(0), true=Y[:, d0:b].sum(0), d1_z=med[:, :LEAD].astype(np.float32), pes_z=med[:, LEAD:].astype(np.float32), true_z=Y[:, d0:b].astype(np.float32))
    print(tag, f"{time.time() - t0:.0f}s", flush=True)
print("finished", flush=True)
