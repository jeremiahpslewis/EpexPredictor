"""Does the curve's error in a regime persist across visits? Cell = 5-GW bin of effective residual load x hour block.
For each day's visit to a cell: its mean error against the mean error of earlier visits to the same cell, by gap in days.
slope = best multiple of the earlier error to apply (through the origin); r = correlation; n = visits."""
import numpy as np, sys
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
BLK = np.array([0] * 6 + [1] * 4 + [2] * 6 + [3] * 8); GAPS = [(1, 3), (4, 10), (11, 30), (31, 90)]
def table(R, X, blocks, sel=None):
    n = len(R); xb = np.floor((X + 10) / 5).astype(int); out = {}
    cells = {}
    for v in range(n):
        ok = np.isfinite(R[v]) & np.isfinite(X[v])
        for key in set(zip(xb[v][ok], blocks[ok])):
            m = ok & (xb[v] == key[0]) & (blocks == key[1])
            if m.sum() >= 2: cells.setdefault(key, []).append((v, m.sum(), R[v][m].mean()))
    for g0, g1 in GAPS:
        xs, ys, ws = [], [], []
        for key, vis in cells.items():
            if sel is not None and not sel(key): continue
            for i, (v, k, r) in enumerate(vis):
                prev = [(k2, r2) for (v2, k2, r2) in vis[:i] if g0 <= v - v2 <= g1]
                if sum(k2 for k2, _ in prev) >= 3: xs.append(np.average([r2 for _, r2 in prev], weights=[k2 for k2, _ in prev])); ys.append(r); ws.append(k)
        xs, ys, ws = map(np.array, (xs, ys, ws))
        out[(g0, g1)] = (float((ws * xs * ys).sum() / (ws * xs * xs).sum()), float(np.corrcoef(xs, ys)[0, 1]), len(xs)) if len(xs) > 20 else (np.nan, np.nan, len(xs))
    return out
fmt = lambda t: "  ".join(f"{g0}-{g1} d: slope {s:+.2f} r {r:+.2f} n {n}" for (g0, g1), (s, r, n) in t.items())
for lab, run, tags in (("winter 25/26", "hw2", ["2026-02-26"]), ("autumn 2026", "ha", ["2026-11-06"]), ("summer 2026", "hs2", ["2026-08-25"]), ("autumn 2025", "h25", ["2025-11-27"])):
    import glob, os
    f = f"{SP}/out/struct_{run}/{tags[0]}.npz"
    if not os.path.exists(f): f = sorted(glob.glob(f"{SP}/out/struct_{run}/2*.npz"))[-1]
    z = np.load(f); R = z["R1"].astype(float) / 10; X = z["XH1"].astype(float); lo, hi = np.nanpercentile(R, [1, 99]); R = np.clip(R, lo, hi)
    print(f"\n{lab} (history of {len(R)} days before {os.path.basename(f)[:10]})")
    print("  cells x-bin x hour block, all        " + fmt(table(R, X, BLK)))
    print("  the same, x < 15 GW only             " + fmt(table(R, X, BLK, lambda k: k[0] <= 4)))
    print("  the same, x < 15 GW, night block     " + fmt(table(R, X, BLK, lambda k: k[0] <= 4 and k[1] == 0)))
    print("  cells x-bin only (hours pooled), all " + fmt(table(R, X, np.zeros(24, int))))
    print("  x-bin only, x < 15 GW                " + fmt(table(R, X, np.zeros(24, int), lambda k: k[0] <= 4)))
