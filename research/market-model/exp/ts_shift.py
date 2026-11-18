"""Timestral on the supply shift in GW (notes 21.51), in place of the kernel layer. Per origin: u = (published residual load - other plant) - S(known price)
- neighbour term, quarter-hourly, context to the start of the target day; covariates: demand axis and neighbour term (published in the context, forecast in
the horizon), weekday. Other plant of the last two days and the horizon = slot-of-day mean of days -9..-2 (not yet measured at the issue).
usage: ts_shift.py PERIOD [GEN=mid]"""
import os, sys, time
import numpy as np, pandas as pd, torch
from timestral import TimestralForecaster
import live_common as lc
torch.set_num_threads(4); period = sys.argv[1]; GEN = sys.argv[2] if len(sys.argv) > 2 else "mid"; CTX = 8192; NONB = os.environ.get("NONB", "0") == "1"; SFX = "_m1nonb" if NONB else ""; LAGS = os.environ.get("LAGS", "0") == "1"; OSFX = "_c" if LAGS else ""; MOD = os.environ.get("MOD", ""); SFX = "_" + MOD if MOD else SFX   # version C (21.67): yesterday's scheduled dispatchable output as a known covariate
F = lc.load(); idx = F["DE"].index; wd = F["DE"]["weekday"].to_numpy(float); Z = np.load(f"{lc.SP}/out/shift15{SFX}.npz"); nbi = pd.read_parquet(f"{lc.SP}/out/nb_rl.parquet").index; assert len(nbi) == len(idx) and (nbi == idx).all()
I = np.load(f"{lc.SP}/out/inputs_{period}.npz"); cuts = [int(c) for c in I["cuts"]]; G = np.load(f"{lc.SP}/out/gen_{GEN}_{period}.npz"); NB = np.load(f"{lc.SP}/out/nbmid2_{period}.npz")["fc"].astype(float)
DISP = np.load(f"{lc.SP}/out/disp15.npz")["disp"].astype(float) if LAGS else None
TRO = os.environ.get("TRO", "")   # 21.86: e = trough depth (demand minus its 8-hour minimum) as a known covariate; f = subtract a fitted hour-of-day multiple of it from the shift, forecast the rest, add it back
if TRO: OSFX = "_" + TRO; TB = np.load(f"{lc.SP}/out/trough_beta.npz"); BH = TB["beta"][TB["hour_local"].astype(int)]
f = TimestralForecaster.from_pretrained("mistralai/Timestral-123M", revision="5a147287addda85f6acd8da3fb8858fe9af7e8f7"); QS = (0.05, 0.1, 0.25, 0.75, 0.9, 0.95); out = {k: [] for k in ["median", "xh", "nbh"] + [f"q{int(q * 100):02d}" for q in QS]}; t0 = time.time()
for j, d0 in enumerate(cuts):
    a, b = d0 - CTX, d0 + lc.HZ; oth = np.nanmean(Z["other"][d0 - 9 * 96:d0 - 2 * 96].reshape(7, 96), 0); ok_ = np.tile(oth, (b - (d0 - 192)) // 96 + 1)[: b - (d0 - 192)]; other = np.r_[Z["other"][a:d0 - 192], ok_]
    x = Z["rl"][a:b] - other; x[CTX:] = G["rl_pes"][j].astype(float) - other[CTX:]; nb = Z["nbterm"][a:b].copy(); nb[CTX:] = (NB[j] - Z["NMU"]) @ Z["B"]
    u = pd.Series(x[:CTX] - Z["Sp"][a:d0] - nb[:CTX]).ffill().bfill().to_numpy(); x = pd.Series(x).ffill().bfill().to_numpy()
    tro = (pd.Series(x) - pd.Series(x).rolling(32, min_periods=1).min()).to_numpy(); add = np.zeros(lc.HZ)
    if TRO == "f": u = u - BH[a:d0] * tro[:CTX]; add = BH[d0:b] * tro[CTX:]
    with torch.no_grad():
        r = f.forecast(u[None, None, :], prediction_length=lc.HZ, known_covariates=(x[None] if NONB else np.stack([x, nb] + ([tro] if TRO == "e" else []) + ([np.r_[DISP[a - 96:d0 - 96], np.r_[DISP[d0 - 96:d0], DISP[d0 - 96:d0]][: b - d0]]] if LAGS else [])))[None], categorical_known_covariates=wd[a:b][None, None, :])
    out["median"].append(np.asarray(r.median)[0, 0] + add); out["xh"].append(x[CTX:]); out["nbh"].append(nb[CTX:])
    for q in QS: out[f"q{int(q * 100):02d}"].append(np.asarray(r.quantile(q))[0, 0] + add)
np.savez_compressed(f"{lc.SP}/out/tsshift_{GEN}{SFX}{OSFX}_{period}.npz", **{k: np.stack(v).astype(np.float32) for k, v in out.items()}); print("finished", period, f"{time.time() - t0:.0f}s", flush=True)
