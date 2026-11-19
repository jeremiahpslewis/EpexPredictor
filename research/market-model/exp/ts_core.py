"""Sponge test (notes 21.128): Timestral forecasts the day-ahead innovations of the state core K3, the core is added back.
usage: ts_core.py PERIOD [GEN]   output out/tscore_<period>.npz: core (K3 for the target day), ts (Timestral's forecast of the innovations), xh, thin"""
import os, sys, time
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
import live_common as lc, core_state as cs
torch.set_num_threads(4); period = sys.argv[1]; GEN = sys.argv[2] if len(sys.argv) > 2 else "mid"; CTX = 8192; ND, WARM = 86, 40; JOIN = os.environ.get("JOIN", "k3"); cs.BYTYPE = JOIN != "k1a"
F = lc.load(); D = F["DE"]; wd = D["weekday"].to_numpy(float); WE = ((wd >= 5) | (D["holiday"].to_numpy(float) >= 0.5)); Z = np.load(f"{lc.SP}/out/shift15_lite.npz"); X = Z["rl"] - Z["other"]; U = X - Z["Sp"]
I = np.load(f"{lc.SP}/out/inputs_{period}.npz"); cuts = [int(c) for c in I["cuts"]]; G = np.load(f"{lc.SP}/out/gen_{GEN}_{period}.npz")
f = TimestralForecaster.from_pretrained("mistralai/Timestral-123M", revision="5a147287addda85f6acd8da3fb8858fe9af7e8f7"); out = {k: [] for k in ("core", "k1", "ts", "xh", "thin", "thin1")}; t0 = time.time()
for j, d0 in enumerate(cuts):
    nd = ND + WARM; xf = G["rl_pes"][j].astype(float)[:96] - 8.0
    Ud = np.r_[U[d0 - nd * 96:d0], np.full(96, np.nan)].reshape(nd + 1, 96); Xd = X[d0 - nd * 96:d0 + 96].reshape(nd + 1, 96); WEd = WE[d0 - nd * 96:d0 + 96].reshape(nd + 1, 96)[:, 48]
    R = cs.run3(Ud, Xd, WEd, WARM, xf); inn = (Ud[WARM:-1] - R["K3"][:-1]).ravel(); u_ctx = Ud[WARM:-1].ravel()
    x = np.r_[X[d0 - CTX:d0], xf]; x = pd.Series(x).ffill().bfill().to_numpy(); k3all = np.r_[R["K3"][:-1].ravel()[-CTX:], R["K3"][-1]]; k1all = np.r_[R["K1"][:-1].ravel()[-CTX:], R["K1"][-1]]
    if JOIN == "k3": e = inn[-CTX:]; cov = x[None]
    elif JOIN in ("k1", "k1a"): e = (Ud[WARM:-1] - R["K1"][:-1]).ravel()[-CTX:]; cov = x[None]
    else: e = u_ctx[-CTX:]; cov = np.stack([x, pd.Series(k3all).ffill().bfill().to_numpy()])
    e = pd.Series(e).ffill().bfill().to_numpy()
    with torch.no_grad():
        r = f.forecast(e[None, None, :], prediction_length=96, known_covariates=cov[None], categorical_known_covariates=wd[d0 - CTX:d0 + 96][None, None, :])
    out["core"].append(R["K3"][-1]); out["k1"].append(R["K1"][-1]); out["ts"].append(np.asarray(r.median)[0, 0]); out["xh"].append(xf); out["thin"].append(np.nanvar(inn[-CTX:]) / np.nanvar(u_ctx[-CTX:])); out["thin1"].append(np.nanvar((Ud[WARM:-1] - R["K1"][:-1]).ravel()[-CTX:]) / np.nanvar(u_ctx[-CTX:]))
    if j % 10 == 0: print(period, j, f"{time.time() - t0:.0f}s", flush=True)
np.savez_compressed(f"{lc.SP}/out/tscore{'' if JOIN == 'k3' else '_' + JOIN}_{period}.npz", **{k: np.stack(v).astype(np.float32) for k, v in out.items()}); print("finished", period, f"{time.time() - t0:.0f}s", flush=True)
