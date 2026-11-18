"""Stability in function space: for a fixed panel of hours (their own loads, weather and availability; gas and carbon held at 35 and 75), how much does the
predicted price move from one weekly fit to the next, and over all fits?  usage: func_stab.py [VARIANT]"""
import sys, os, numpy as np, pandas as pd
if len(sys.argv) > 1 and sys.argv[1]: os.environ["OCXV"] = sys.argv[1]
import oc
from ocx_state import *
dates = fit_dates(); h_a, h_b = hx("2025-07-01"), hx("2026-07-01"); sl = slice(h_a, h_b); d = rows2(sl, drl=np.zeros(h_b - h_a)); ok = np.isfinite(d["rl"]).all(1) & np.isfinite(d["sol"]) & np.isfinite(d["gasy"]) & np.isfinite(d["nuc"]) & np.isfinite(d["teff"])
dd = {k: (v if k == "G" else np.nan_to_num(v)) for k, v in d.items()}; dd["okx"] = ok; dd["gas"] = np.full(len(ok), 35.0); dd["eua"] = np.full(len(ok), 75.0); pick = np.flatnonzero(ok)[::7]
PR, XS = [], []
for t in dates:
    A = get_fit(t); x = index2(A, dd, np); XS.append(x[pick]); PR.append(pcurve(A, x, dd["gas"], dd["eua"], np)[0][pick] / 10)
PR, XS = np.array(PR), np.array(XS); x0 = XS.mean(0); mo = oc.HOUR[sl][pick]; rl = d["rl"][pick, 0]
print(f"{'variant ' + os.environ.get('OCXV', '') if os.environ.get('OCXV') else 'second model'}: {len(pick)} panel hours (Jul 2025 - Jun 2026), {len(dates)} weekly fits; predicted price at gas 35, carbon 75")
print(f"  week-to-week change of the prediction for the same hour: mean absolute {np.mean(np.abs(np.diff(PR, axis=0))):.3f} ct, 95th percentile {np.percentile(np.abs(np.diff(PR, axis=0)), 95):.3f}; largest weekly mean shift of the whole panel {np.max(np.abs(np.diff(PR, axis=0).mean(1))):.2f} ct")
print(f"  over all fits: sd across fits of the prediction for the same hour, mean over hours {PR.std(0).mean():.3f} ct; drift of the panel mean, first 10 fits -> last 10 fits: {PR[:10].mean():.2f} -> {PR[-10:].mean():.2f} ct")
for lab, k in (("residual load < 15 GW", rl < 15), ("15-35 GW", (rl >= 15) & (rl < 35)), ("35-50 GW", (rl >= 35) & (rl < 50)), (">= 50 GW", rl >= 50)):
    print(f"    {lab:22s} ({k.sum():4d} h): sd across fits {PR[:, k].std(0).mean():.2f} ct; panel mean by fit, every 9th: " + " ".join(f"{v:.2f}" for v in PR[::9, k].mean(1)))
print(f"  index for the same hour: sd across fits {XS.std(0).mean():.2f} GW; after removing each fit's mean offset {(XS - XS.mean(1, keepdims=True)).std(0).mean():.2f} GW; week-to-week mean absolute change {np.mean(np.abs(np.diff(XS, axis=0))):.2f} GW")
np.savez(f"{SP}/out/func_stab{SFX}.npz", PR=PR, XS=XS, rl=rl)
