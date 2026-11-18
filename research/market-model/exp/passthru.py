"""Does the price level flow through the model's functions?  Same season (Aug-Nov) in 2025 and 2026: German price by residual-load bin in the data, and in
the fitted model cleared (a) with each year's own gas and carbon, (b) for 2026 with the 2025 gas and carbon. The 2025->2026 change per bin, data against model,
is the pass-through of the cost level; which blocks sit at the margin in the model is printed alongside.  usage: passthru.py RUN PERIOD TAG"""
import sys, numpy as np, pandas as pd
import star as S
run, period, tag = sys.argv[1], sys.argv[2], sys.argv[3]; SP = S.lc.SP
z = np.load(f"{SP}/out/star_{run}/{tag}_s0.npz"); q = S.unpack({k[4:]: z[k] for k in z.files if k.startswith("raw_")}); nw = z["raw_c_e"].shape[0] + 1
ix = lambda s: int(np.searchsorted(S.idx.values, np.datetime64(pd.Timestamp(s, tz="Europe/Berlin").tz_convert("UTC").tz_localize(None))))
def clear(a, b, gas=None, eua=None, wk=None):
    H = S.window(ix(a), ix(b)); n = len(H["P"]); d = S.data_of(H, np.full(n, nw - 1) if wk is None else wk, nw)
    if gas is not None: d["gas"] = np.full(n, gas); d["eua"] = np.full(n, eua - 70.0)
    pe, e = S.equilibrium(q, d); cap, mc, wd, sl, shift, ct = S.build(q, d, np); wd = S.w3(wd, n); sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -60, 60)))
    pdm = S.day_mean(pe[:, 0], d["day"]); u = np.stack([sg((pe[:, 0] + q["lam"][0, k] * (pdm - pe[:, 0]) - mc[:, 0, k]) / wd[:, 0, k]) for k in range(1, 5)], 1)      # loading of German lignite, coal, combined cycle, peakers
    slope = np.stack([cap[:, 0, k] * u[:, k - 1] * (1 - u[:, k - 1]) / wd[:, 0, k] * (1 - q["lam"][0, k]) for k in range(1, 5)], 1)
    return H, d, pe, e, u, slope, mc[:, 0, 1:5]
H5, d5, p5, e5, u5, s5, m5 = clear("2025-08-01", "2025-11-07"); H6, d6, p6, e6, u6, s6, m6 = clear("2026-08-01", "2026-11-07")
g5, c5 = float(np.nanmean(H5["gas"])), float(np.nanmean(H5["eua"])) + 70; g6, c6 = float(np.nanmean(H6["gas"])), float(np.nanmean(H6["eua"])) + 70
_, _, p6c, e6c, u6c, s6c, m6c = clear("2026-08-01", "2026-11-07", gas=g5, eua=c5)
print(f"{run} {tag}. Aug-Nov 2025: gas {g5:.0f}, carbon {c5:.0f}; Aug-Nov 2026: gas {g6:.0f}, carbon {c6:.0f}. Textbook cost change: combined cycle {1.85 * (g6 - g5) + 0.37 * (c6 - c5):+.0f}, lignite {c6 - c5:+.0f}, hard coal {0.85 * (c6 - c5):+.0f} EUR/MWh")
print(f"model's German block costs, mean 2025 -> 2026: lignite {m5[:, 0].mean():.0f} -> {m6[:, 0].mean():.0f}, hard coal {m5[:, 1].mean():.0f} -> {m6[:, 1].mean():.0f}, combined cycle {m5[:, 2].mean():.0f} -> {m6[:, 2].mean():.0f}, peakers {m5[:, 3].mean():.0f} -> {m6[:, 3].mean():.0f}")
bins = [(-20, 5), (5, 15), (15, 25), (25, 35), (35, 45), (45, 90)]
print("residual load bin | data: price 2025, 2026, change | model: 2025, 2026, change | model 2026 with 2025 gas and carbon -> change due to costs alone | share of the German supply slope by block in 2026 (lignite, coal, cc, peak) and block loadings")
for a, b in bins:
    k5 = (H5["rl"][:, 0] >= a) & (H5["rl"][:, 0] < b) & np.isfinite(H5["P"][:, 0]); k6 = (H6["rl"][:, 0] >= a) & (H6["rl"][:, 0] < b) & np.isfinite(H6["P"][:, 0])
    if k5.sum() < 30 or k6.sum() < 30: continue
    md = lambda v: np.median(v); tot = s6[k6].sum(1).mean() + 1e-9
    print(f"  [{a:3d},{b:3d})  n {k5.sum():4d}/{k6.sum():4d} | {md(H5['P'][k5, 0]):4.0f} {md(H6['P'][k6, 0]):4.0f} {md(H6['P'][k6, 0]) - md(H5['P'][k5, 0]):+4.0f} | {md(p5[k5, 0]):4.0f} {md(p6[k6, 0]):4.0f} {md(p6[k6, 0]) - md(p5[k5, 0]):+4.0f} | {md(p6c[k6, 0]):4.0f} -> {md(p6[k6, 0]) - md(p6c[k6, 0]):+4.0f} | " + " ".join(f"{100 * s6[k6][:, i].mean() / tot:3.0f}%" for i in range(4)) + " | " + " ".join(f"{u6[k6][:, i].mean():.2f}" for i in range(4)))
print(f"model net exports 2026, mean by bin: " + "  ".join(f"[{a},{b}) {e6[(H6['rl'][:, 0] >= a) & (H6['rl'][:, 0] < b)].sum(1).mean():+.1f} (scheduled {np.nanmean(np.where(d6['xok'], d6['x'], np.nan)[(H6['rl'][:, 0] >= a) & (H6['rl'][:, 0] < b)].sum(1)):+.1f})" for a, b in bins))
print(f"overall: data price change {np.nanmean(H6['P'][:, 0]) - np.nanmean(H5['P'][:, 0]):+.0f} EUR/MWh; model {p6[:, 0].mean() - p5[:, 0].mean():+.0f}; model, costs alone {p6[:, 0].mean() - p6c[:, 0].mean():+.0f}; per EUR of gas: data {(np.nanmean(H6['P'][:, 0]) - np.nanmean(H5['P'][:, 0])) / (g6 - g5):.2f}, model {(p6[:, 0].mean() - p5[:, 0].mean()) / (g6 - g5):.2f}")
