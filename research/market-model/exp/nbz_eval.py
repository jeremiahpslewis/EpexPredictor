"""Per-zone neighbour forecasts against the earlier one-sum forecast, on the forecasts both have. usage: nbz_eval.py PERIOD"""
import sys, os, glob, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; period = sys.argv[1]
fs = sorted(glob.glob(f"{SP}/out/tsnbz_{period}/2*.npz")); fs = [f for f in fs if os.path.exists(f.replace("tsnbz_", "tsnb_"))]
new = [np.load(f) for f in fs]; old = [np.load(f.replace("tsnbz_", "tsnb_")) for f in fs]
tr = np.stack([z["true"] for z in new]); pn = np.stack([z["pes"] for z in new]); po = np.stack([z["pes"] for z in old]); d1n = np.stack([z["d1"] for z in new]); d1o = np.stack([z["d1"] for z in old]); d1t = np.stack([z["d1_true"] for z in new])
m = lambda p, t, sl: np.nanmean(np.abs(p - t)[:, sl])
print(f"{period}: {len(fs)} forecasts. MAE of the neighbours' residual load (sum of 8 zones), GW: night / first 14 h / D+1")
print(f"   one sum, Europe-wide weather (before): {m(po, tr, slice(0, 32)):.2f} / {m(po, tr, slice(0, 56)):.2f} / {m(d1o, d1t, slice(0, 96)):.2f}")
print(f"   zone by zone, own weather (new):       {m(pn, tr, slice(0, 32)):.2f} / {m(pn, tr, slice(0, 56)):.2f} / {m(d1n, d1t, slice(0, 96)):.2f}   change first 14 h {100 * (m(pn, tr, slice(0, 56)) / m(po, tr, slice(0, 56)) - 1):+.0f}%")
print(f"   average of the two:                    {m((pn + po) / 2, tr, slice(0, 32)):.2f} / {m((pn + po) / 2, tr, slice(0, 56)):.2f}")
zn = [str(s) for s in new[0]["zones"]]; pz = np.stack([z["pes_z"] for z in new]); tz = np.stack([z["true_z"] for z in new])
print("   by zone, first 14 h MAE (bias): " + "  ".join(f"{c} {np.nanmean(np.abs(pz[:, i, :56] - tz[:, i, :56])):.2f} ({np.nanmean(pz[:, i, :56] - tz[:, i, :56]):+.2f})" for i, c in enumerate(zn)))
e = (pn - tr)[:, :56]; print(f"   error of the sum: bias {np.nanmean(e):+.2f}, sd of the forecast-mean error {np.nanstd(np.nanmean(e, 1)):.2f} (before: bias {np.nanmean((po - tr)[:, :56]):+.2f}, sd {np.nanstd(np.nanmean((po - tr)[:, :56], 1)):.2f})")
