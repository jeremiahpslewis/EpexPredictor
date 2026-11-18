"""Function-space checks. (1) The panel of func_stab.py evaluated at three cost points: gas 35 / carbon 75, gas 60 / carbon 80, and each fit's own current
gas and carbon (last 30 days of its window; reported in units of the gas-fired cost L). Full-window fits only. (2) Same data, 2500 against 8000 steps."""
import sys, os, numpy as np, pandas as pd
import oc
from ocx_state import *
dates = [t for t in fit_dates() if t >= "2025-10-06"]; h_a, h_b = hx("2025-07-01"), hx("2026-07-01"); sl = slice(h_a, h_b); d = rows2(sl, drl=np.zeros(h_b - h_a)); ok = np.isfinite(d["rl"]).all(1) & np.isfinite(d["sol"]) & np.isfinite(d["gasy"]) & np.isfinite(d["nuc"]) & np.isfinite(d["teff"])
dd = {k: (v if k == "G" else np.nan_to_num(v)) for k, v in d.items()}; dd["okx"] = ok; pick = np.flatnonzero(ok)[::7]; rl = d["rl"][pick, 0]; n = len(ok)
bins = (("< 15 GW", rl < 15), ("15-35 GW", (rl >= 15) & (rl < 35)), ("35-50 GW", (rl >= 35) & (rl < 50)), (">= 50 GW", rl >= 50))
def panel(A, g, e): return pcurve(A, index2(A, dd, np), np.full(n, g), np.full(n, e), np)[0][pick] / 10
own = []; P35, P60, POWN = [], [], []
for t in dates:
    A = get_fit(t); h1 = hx(t); g, e = float(np.nanmean(oc.GAS[h1 - 720:h1])), float(np.nanmean(oc.EUA[h1 - 720:h1])); own.append((g, e)); P35.append(panel(A, 35, 75)); P60.append(panel(A, 60, 80)); POWN.append(panel(A, g, e) / ((1.85 * g + 0.37 * e) / 10))
P35, P60, POWN = np.array(P35), np.array(P60), np.array(POWN); st = max(1, len(dates) // 8)
print(f"{len(dates)} full-window fits {dates[0]} .. {dates[-1]}; shown every {st}th fit: " + " ".join(dates[::st])); print("  own gas / carbon at the fit date:   " + " ".join(f"{g:.0f}/{e:.0f}" for g, e in own[::st]))
for lab, k in bins:
    print(f"  {lab:9s} at gas 35, carbon 75 [ct]: " + " ".join(f"{v:5.2f}" for v in P35[::st, k].mean(1)) + f"   | at gas 60, carbon 80 [ct]: " + " ".join(f"{v:5.2f}" for v in P60[::st, k].mean(1)) + f"   | at own costs [price / gas-fired cost]: " + " ".join(f"{v:4.2f}" for v in POWN[::st, k].mean(1)))
print("  implied pass-through between the two cost points (change in price per change in gas-fired cost, the latter = 1.85 x 25 + 0.37 x 5 = 48.1 EUR/MWh): " + " | ".join(f"{lab} " + " ".join(f"{v:.2f}" for v in ((P60[::st, k].mean(1) - P35[::st, k].mean(1)) * 10 / 48.1)) for lab, k in bins))
A0 = get_fit("2026-03-30"); A1 = {k: np.asarray(v, float) for k, v in np.load(f"{SP}/out/seedfit_2026-03-30_s0_n8000.npz").items()}; A2 = {k: np.asarray(v, float) for k, v in np.load(f"{SP}/out/seedfit_2026-03-30_s1_n2500.npz").items()}
print("\nsame data (fit 2026-03-30): 2500 steps (as cached) against 8000 steps (converged; 30000 steps gives the same log posterior), difference of the predicted price for the panel [ct]")
for g, e in ((35, 75), (60, 80), (90, 87)):
    a, b, c = panel(A0, g, e), panel(A1, g, e), panel(A2, g, e)
    print(f"  gas {g}, carbon {e}: " + " | ".join(f"{lab}: 8000 - 2500 mean {np.mean((b - a)[k]):+.2f}, mean abs {np.mean(np.abs(b - a)[k]):.2f}; other seed at 2500: mean abs {np.mean(np.abs(c - a)[k]):.2f}" for lab, k in bins))
