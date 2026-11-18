"""What is in the unexplained part of the German balance (noise scale about 3.4 GW)?  Residual of the fitted model at observed prices against the
measured pieces the model does not carry: other German generation, exchange over borders not in the model, and the balance gap of the data themselves."""
import sys, numpy as np, pandas as pd
import star as S
run, period, tag = "v11", "autumn", "2026-09-18"; SP = S.lc.SP
z = np.load(f"{SP}/out/star_{run}/{tag}_s0.npz"); q = S.unpack({k[4:]: z[k] for k in z.files if k.startswith("raw_")})
I = np.load(f"{SP}/out/inputs_{period}.npz"); d0 = int(I["cuts"][[str(c)[:10] for c in I["cutoffs"]].index(tag)]); d1 = d0 - 96; lo = max(d1 - S.TRAIN_DAYS * 96, 15 * 96); lo += (-lo) % 4
H = S.window(lo, d1); week = (H["day"] - H["day"].min()) // 7; d = S.data_of(H, week, int(week.max()) + 1); P = H["P"]; n = len(P)
cap, mc, wd, sl, shift, ct = S.build(q, d, np); fl = S.floors(q, d, np).sum(1) * np.ones(n); hde = q["hz"][0] * d["hdd"][:, 0]
M = pd.read_parquet(f"{SP}/out/de_mix.parquet"); hq = lambda v: np.nanmean(np.asarray(v, float)[lo:lo + 4 * n].reshape(-1, 4), 1)
sup = S.supply(np.nan_to_num(P[:, :1]), cap[:, :1], mc[:, :1], wd[:1], sl[:1], np, pbar=d["pdmz"][:, :1], lam=q["lam"][:1])[:, 0]
dem = d["rl"][:, 0] + shift + np.where(d["xok"], d["x"], 0).sum(1) - fl - hde - np.where(d["psok"], d["ps"], 0) + q["capf"][0, 5]; res = sup - dem           # + = model supplies more than demanded
ok = d["ok"][:, 0] & d["xok"].all(1) & d["psok"]; other = sum(hq(M[c]) for c in ["Hydro Run-of-River", "Biomass", "Fossil oil", "Fossil coal-derived gas", "Geothermal", "Hydro water reservoir", "Others", "Waste"])
trade = hq(M["trade"]); x8 = -np.where(d["xok"], d["x"], 0).sum(1)       # trade: + = import (energy-charts); x8: imports over the eight modelled borders
xo = trade - x8; th = hq(M["thermal"]) if "thermal" in M else np.nan
gap = hq(M["Residual load"]) - (hq(M["lignite"]) + hq(M["coal"]) + hq(M["gas_gen"]) + other + hq(M["psgen"]) + hq(M["pump"]) + trade)                       # what the published data themselves leave open
k = ok & np.isfinite(other) & np.isfinite(trade) & np.isfinite(gap); print(f"hours with every piece measured: {k.sum()} of {n}")
print(f"model's German balance residual at observed prices: sd {res[k].std():.2f} GW, mean {res[k].mean():+.2f}, 5..95% {np.percentile(res[k], 5):+.1f}..{np.percentile(res[k], 95):+.1f}")
for nm, v in (("other German generation (run-of-river, biomass, waste, oil, ...)", other), ("imports over borders not in the model (Nordic, Luxembourg, ...)", xo), ("gap in the published data: residual load minus all listed sources", gap), ("residual load as used by the model minus energy-charts residual load", d["rl"][:, 0] - hq(M["Residual load"]))):
    print(f"  {nm}: mean {np.nanmean(v[k]):+.2f} GW, sd {np.nanstd(v[k]):.2f}, 5..95% {np.nanpercentile(v[k], 5):+.1f}..{np.nanpercentile(v[k], 95):+.1f}; correlation with the model residual {np.corrcoef(v[k], res[k])[0, 1]:+.2f}; with the German price {np.corrcoef(v[k], P[k, 0])[0, 1]:+.2f}")
X = np.column_stack([np.ones(k.sum()), other[k], xo[k], gap[k]]); b, *_ = np.linalg.lstsq(X, -res[k], rcond=None); print(f"regression of (demand - model supply) on the three pieces: coefficients {np.round(b[1:], 2)}, share of variance explained {100 * (1 - np.var(-res[k] - X @ b) / np.var(res[k])):.0f}%")
hr = H["hour"]; mo = pd.DatetimeIndex(S.local[lo:lo + 4 * n:4]).month.values
print("other German generation by month (GW): " + "  ".join(f"{m}: {np.nanmean(other[k & (mo == m)]):.1f}" for m in range(1, 13) if (k & (mo == m)).sum() > 50))
print("other German generation by hour block: " + "  ".join(f"{a}-{b}h {np.nanmean(other[k & (hr >= a) & (hr < b)]):.1f}" for a, b in ((0, 6), (6, 10), (10, 16), (16, 20), (20, 24))) + f" | against price: below 0: {np.nanmean(other[k & (P[:, 0] < 0)]):.1f}, 0-50: {np.nanmean(other[k & (P[:, 0] >= 0) & (P[:, 0] < 50)]):.1f}, 50-100: {np.nanmean(other[k & (P[:, 0] >= 50) & (P[:, 0] < 100)]):.1f}, 100-150: {np.nanmean(other[k & (P[:, 0] >= 100) & (P[:, 0] < 150)]):.1f}, above 150: {np.nanmean(other[k & (P[:, 0] >= 150)]):.1f}")
print("imports over other borders against price: " + "  ".join(f"{a}..{b}: {np.nanmean(xo[k & (P[:, 0] >= a) & (P[:, 0] < b)]):+.1f}" for a, b in ((-500, 0), (0, 50), (50, 100), (100, 150), (150, 1000))))
