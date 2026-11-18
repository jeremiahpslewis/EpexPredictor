"""Per-hour-of-day comparison of two stored backtests (DE, stage-2 predictions)."""
import sys

import numpy as np
import pandas as pd

out, base, var = sys.argv[1:4]
zb, zv = np.load(f"{out}/{base}.npz"), np.load(f"{out}/{var}.npz")
common = np.intersect1d(zb["cutoffs"], zv["cutoffs"])
ib, iv = np.searchsorted(zb["cutoffs"], common), np.searchsorted(zv["cutoffs"], common)
a = zb["actual_DE"][ib].astype(float)
eb = zb["s2_DE"][ib].astype(float) - a
ev = zv["s2_DE"][iv].astype(float) - a
t = pd.DatetimeIndex((common[:, None] + (np.arange(289) * np.timedelta64(15, "m"))[None, :]).ravel(), tz="UTC").tz_convert("Europe/Berlin")
hr = np.asarray(t.hour).reshape(a.shape)
n = len(common)
print(f"{var} vs {base}: n={n} forecast days; change in MAE / RMSE by local hour (negative = better)")
print("hour | " + " | ".join(f"{h+1}d MAE   RMSE  (base MAE)" for h in range(3)))
worse = {0: [], 1: [], 2: []}
for hh in range(24):
    cells = []
    for h in range(3):
        sl = slice(h * 96, (h + 1) * 96)
        m = hr[:, sl] == hh
        b, v = eb[:, sl][m], ev[:, sl][m]
        dm = np.abs(v).mean() / np.abs(b).mean() - 1
        dr = np.sqrt((v ** 2).mean()) / np.sqrt((b ** 2).mean()) - 1
        if dm > 0 or dr > 0:
            worse[h].append(hh)
        cells.append(f"{100*dm:+5.0f}% {100*dr:+5.0f}%   ({np.abs(b).mean():.2f})")
    print(f" {hh:02d}  | " + " | ".join(cells))
print("hours where MAE or RMSE got worse:", {f"{h+1}d": worse[h] for h in range(3)})

# blocks with a paired block bootstrap over forecast days
rng = np.random.default_rng(0)
blocks = [(0, 8), (8, 12), (12, 16), (16, 20), (20, 24), (0, 24)]
print("\nblock  | " + " | ".join(f"{h+1}d MAE change [95% interval]   RMSE change [95% interval]" for h in range(3)))
for lo, hi in blocks:
    cells = []
    for h in range(3):
        sl = slice(h * 96, (h + 1) * 96)
        m = (hr[:, sl] >= lo) & (hr[:, sl] < hi)
        B, V = np.where(m, eb[:, sl], np.nan), np.where(m, ev[:, sl], np.nan)
        ab, av = np.nanmean(np.abs(B), axis=1), np.nanmean(np.abs(V), axis=1)
        sb, sv = np.nanmean(B ** 2, axis=1), np.nanmean(V ** 2, axis=1)
        blk = 7
        boots_m, boots_r = [], []
        for _ in range(1000):
            starts = rng.integers(0, n - blk + 1, int(np.ceil(n / blk)))
            idx = (starts[:, None] + np.arange(blk)[None, :]).ravel()[:n]
            boots_m.append(av[idx].mean() / ab[idx].mean() - 1)
            boots_r.append(np.sqrt(sv[idx].mean() / sb[idx].mean()) - 1)
        lm, um = np.percentile(boots_m, [2.5, 97.5]); lr, ur = np.percentile(boots_r, [2.5, 97.5])
        cells.append(f"{100*(av.mean()/ab.mean()-1):+4.0f}% [{100*lm:+4.0f},{100*um:+4.0f}]   {100*(np.sqrt(sv.mean()/sb.mean())-1):+4.0f}% [{100*lr:+4.0f},{100*ur:+4.0f}]")
    print(f"{lo:02d}-{hi:02d}  | " + " | ".join(cells))

# price regimes (all horizons pooled)
act = a[:, :288]
B, V = eb[:, :288], ev[:, :288]
print("\nby actual price level (all three horizons pooled): share of slots, MAE base -> new, RMSE base -> new")
for label, m in [("price <= 0", act <= 0), ("0 < price <= p90", (act > 0) & (act <= np.percentile(act, 90))), ("top 10% prices", act > np.percentile(act, 90)), ("top 1% prices", act > np.percentile(act, 99))]:
    print(f"  {label:<18} {100*m.mean():4.1f}%  MAE {np.abs(B[m]).mean():.2f} -> {np.abs(V[m]).mean():.2f} ({100*(np.abs(V[m]).mean()/np.abs(B[m]).mean()-1):+.0f}%)   RMSE {np.sqrt((B[m]**2).mean()):.2f} -> {np.sqrt((V[m]**2).mean()):.2f} ({100*(np.sqrt((V[m]**2).mean()/(B[m]**2).mean())-1):+.0f}%)   bias {B[m].mean():+.2f} -> {V[m].mean():+.2f}")
