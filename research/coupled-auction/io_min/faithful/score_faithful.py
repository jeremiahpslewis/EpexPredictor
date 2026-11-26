"""Score the faithful rerun (EXPECTATIONS.md F1-F7) against the stored-array members on the same 61 local days.
Usage: score_faithful.py [OUT_DIR] [RESULT.md]   (default SP/faithful/out_strict, ./RESULTS.md)
MAE in ct/kWh per day; paired circular 7-day block bootstrap (distill/common.py), 90% intervals."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "distill"))
from common import SP, load_members, day_mae, paired
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{SP}/faithful/out_strict"
RES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.md")
M = load_members(); a, cut = M["a"], M["cut"]
G = np.load(f"{SP}/out/gen_lwA_autumn.npz", allow_pickle=True)
gcut = {pd.Timestamp(c).tz_localize("UTC"): j for j, c in enumerate(G["cutoffs"])}
K = ("trees", "timestral", "market", "issued", "rl_trees", "rl_mixed", "rl_true", "actual")
F = {k: np.full_like(a, np.nan) for k in K}; rl_pes = np.full_like(a, np.nan); have = np.zeros(len(cut), bool); secs = []
for j, c in enumerate(cut):
    d = (c + pd.Timedelta(minutes=15)).tz_convert("Europe/Berlin").date(); fn = f"{OUT}/{d}.npz"
    if not os.path.exists(fn): continue
    z = np.load(fn); idx = pd.to_datetime(z["idx"], utc=True); want = c + pd.to_timedelta(np.arange(96) * 15, "min")
    pos = pd.Index(idx).get_indexer(want)
    if (pos < 0).any(): continue
    for k in K: F[k][j] = z[k][pos]
    if c in gcut: rl_pes[j] = G["rl_pes"][gcut[c], :96]
    have[j] = True; secs.append(json.loads(str(z["meta"]))["secs"]["total"])
nd = int(have.sum())
act_diff = float(np.nanmax(np.abs(F["actual"][have] - a[have]))) if nd else np.nan   # harness's actual price vs the stored one
H = {"faithful trees": F["trees"], "faithful Timestral": F["timestral"], "faithful market route": F["market"], "faithful issued blend": F["issued"]}
S = {"stored T trees (opt)": M["T"], "stored P Timestral": M["P"], "stored A market route": M["A"], "stored M3": (M["T"] + M["P"] + M["A"]) / 3}
sub = lambda X: X[have]; A_ = a[have]
rows = {k: {"MAE": round(float(day_mae(sub(v), A_).mean()), 3), "bias": round(float(np.nanmean(sub(v) - A_)), 3)} for k, v in {**H, **S}.items()}
loc_h = M["loc_h"][have]
for k, v in {**H, **S}.items():
    for lo, hi in [(0, 6), (6, 10), (10, 16), (16, 21), (21, 24)]:
        m = (loc_h >= lo) & (loc_h < hi); rows[k][f"{lo:02d}-{hi:02d}"] = round(float(np.abs(sub(v) - A_)[m].mean()), 2)
pairs = [("faithful issued blend", "stored M3"), ("faithful trees", "stored T trees (opt)"), ("faithful Timestral", "stored P Timestral"),
         ("faithful market route", "stored A market route"), ("faithful issued blend", "faithful trees"), ("faithful issued blend", "faithful Timestral"),
         ("faithful issued blend", "faithful market route")]
allF = {**H, **S}
pl = [f"- {x} − {y}: {paired(sub(allF[x]), sub(allF[y]), A_)}" for x, y in pairs]
# residual load (GW)
rl_tr = np.nanmean(np.abs(F["rl_trees"] - F["rl_true"])[have], 1); rl_mx = np.nanmean(np.abs(F["rl_mixed"] - F["rl_true"])[have], 1)
rl_ps = np.nanmean(np.abs(rl_pes - F["rl_true"])[have], 1); excess = rl_tr - rl_ps
days = [str((c + pd.Timedelta(minutes=15)).tz_convert("Europe/Berlin").date()) for c in cut[have]]
worst = sorted(zip(excess, days, rl_tr, rl_ps), reverse=True)[:8]
out = ["# Faithful rerun of the deployed DE ensemble: results (post hoc, window already used)", "",
       f"Days scored: {nd} of {len(cut)} (local 2026-09-08..11-07), 96 quarter-hours each; MAE in ct/kWh. "
       f"Runtime per day: median {np.median(secs):.0f} s, max {np.max(secs):.0f} s. "
       f"Largest gap between the harness's actual price and the stored one: {act_diff:.4f} ct/kWh." if nd else "no days", "",
       pd.DataFrame(rows).T.to_markdown(), "", "## Paired differences (first minus second; negative = first better), 90% block bootstrap", ""] + pl + [
       "", "## Residual load (GW, mean over days of daily MAE on D)", "",
       f"- faithful tree series: {np.nanmean(rl_tr):.2f}; Timestral-mixed series: {np.nanmean(rl_mx):.2f}; stored pessimistic (rl_pes): {np.nanmean(rl_ps):.2f}",
       f"- F6: days where the faithful tree series exceeds rl_pes by more than 1 GW: {int((excess > 1).sum())} of {int(np.isfinite(excess).sum())}",
       "- largest excesses (day, faithful, rl_pes): " + "; ".join(f"{d} {t:.2f} vs {p:.2f}" for e, d, t, p in worst)]
open(RES, "w").write("\n".join(out)); print("\n".join(out))
