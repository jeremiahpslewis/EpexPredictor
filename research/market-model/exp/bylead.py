"""Change in error by hours since the last known price (first forecast day), paired block bootstrap."""
import sys

import numpy as np

out = sys.argv[1]
rng = np.random.default_rng(1)
bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, 18), (18, 24)]
for label, base, var in [("recent, changes 1+2", "ho_2s_DE_base", "ho_2s_DE_L540_ratio_all"), ("recent, changes 1+2+3", "ho_2s_DE_base", "ho_2s_DE_L540_ratio_all_x6"),
                         ("previous year, changes 1+2", "2s_DE_base", "2s_DE_L540_ratio_all"), ("previous year, changes 1+2+3", "2s_DE_base", "2s_DE_L540_ratio_all_x6")]:
    zb, zv = np.load(f"{out}/{base}.npz"), np.load(f"{out}/{var}.npz")
    common = np.intersect1d(zb["cutoffs"], zv["cutoffs"])
    ib, iv = np.searchsorted(zb["cutoffs"], common), np.searchsorted(zv["cutoffs"], common)
    a = zb["actual_DE"][ib].astype(float)
    eb, ev = np.abs(zb["s2_DE"][ib] - a), np.abs(zv["s2_DE"][iv] - a)
    n = len(common)
    cells = []
    for lo, hi in bins:
        b, v = eb[:, lo * 4:hi * 4].mean(axis=1), ev[:, lo * 4:hi * 4].mean(axis=1)
        boots = []
        for _ in range(1000):
            starts = rng.integers(0, n - 7 + 1, int(np.ceil(n / 7)))
            idx = (starts[:, None] + np.arange(7)[None, :]).ravel()[:n]
            boots.append(v[idx].mean() / b[idx].mean() - 1)
        l, u = np.percentile(boots, [2.5, 97.5])
        cells.append(f"{lo:>2}-{hi:<2}h {b.mean():.2f}>{v.mean():.2f} {100*(v.mean()/b.mean()-1):+4.0f}% [{100*l:+4.0f},{100*u:+4.0f}]")
    print(f"{label:<30} n={n:<3} " + " | ".join(cells))
