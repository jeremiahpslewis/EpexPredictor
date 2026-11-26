"""Score a residual-load variant run against a base run on the same days (the days present in both directories).

usage: score_variants.py VARIANT_DIR BASE_DIR OUT.md
  e.g. score_variants.py SP/faithful/out_strict_rlboth SP/faithful/out_strict SP/faithful/score_both.md
Never writes RESULTS.md (refused). Do not point it at fresh-day output before the fresh run is meant to be scored.

Reports (MAE in ct/kWh, RL in GW; per day first, then the mean over days):
- price MAE per member (trees, Timestral, market) and the issued blend, variant - base with a paired circular 7-day
  block bootstrap (distill/common.py), 90 %; "n/a" with fewer than 7 days;
- tree RL (what the price trees read) MAE and bias on D, split by the variant's grid vintage per slot (rlc_k: d1 vs
  d2+), with the same bootstrap on the per-day differences;
- Timestral-mixed RL, without the last 3 slots of D (out_strict's rl_mixed is Timestral alone there, README erratum);
- where stored: the components of each model set (app, matched, control, held-out) on d2+ slots, and the alternative
  load-correction forms (study window, held-out residual on the app model).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "distill"))
from common import boot  # noqa: E402

VAR, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
assert os.path.basename(OUT) != "RESULTS.md", "refusing to write RESULTS.md"
days = sorted(f[:-4] for f in os.listdir(VAR) if f.endswith(".npz") and os.path.exists(f"{BASE}/{f}"))
assert days, "no common days"
V = {d: np.load(f"{VAR}/{d}.npz") for d in days}
B = {d: np.load(f"{BASE}/{d}.npz") for d in days}
for d in days:
    assert np.array_equal(V[d]["idx"], B[d]["idx"]) and np.allclose(V[d]["actual"], B[d]["actual"], equal_nan=True), d
n = len(days)
lines = [f"# {os.path.basename(VAR.rstrip('/'))} vs {os.path.basename(BASE.rstrip('/'))}", "",
         f"Days ({n}): {', '.join(days)}. Differences are variant minus base; negative = variant better. "
         f"Intervals: paired circular 7-day block bootstrap, 90 %" + (" (n/a: fewer than 7 days)." if n < 7 else "."), ""]


def ci(x):
    x = np.asarray(x, float)
    if n < 7:
        return f"{np.nanmean(x):+.3f} [n/a]"
    lo, hi = boot(x)
    return f"{np.nanmean(x):+.3f} [{lo:+.3f}, {hi:+.3f}]"


def mae(f, t, m=None):
    e = np.abs(np.asarray(f, float) - np.asarray(t, float))
    return float(np.nanmean(e if m is None else e[m])) if (m is None or m.any()) else np.nan


def bias(f, t, m):
    return float(np.nanmean((np.asarray(f, float) - np.asarray(t, float))[m])) if m.any() else np.nan


# ---- prices
lines += ["## Price MAE", "", "| member | base | variant | variant − base |", "|---|---|---|---|"]
per_day = {}
for k, name in (("trees", "trees"), ("timestral", "Timestral"), ("market", "market route"), ("issued", "issued blend")):
    b = np.array([mae(B[d][k], B[d]["actual"]) for d in days]); v = np.array([mae(V[d][k], V[d]["actual"]) for d in days])
    per_day[k] = (b, v)
    lines.append(f"| {name} | {b.mean():.3f} | {v.mean():.3f} | {ci(v - b)} |")
lines += ["", "Per day (base → variant): " + "; ".join(
    f"{d}: " + ", ".join(f"{k} {per_day[k][0][i]:.3f}→{per_day[k][1][i]:.3f}" for k in per_day) for i, d in enumerate(days)), ""]

# ---- residual load
lines += ["## Residual load on D (GW)", "", "Split by the variant's grid vintage per slot (`rlc_k`).", "",
          "| series | slots | base MAE | variant MAE | diff | base bias | variant bias |", "|---|---|---|---|---|---|---|"]
masks = {"all": lambda z: np.isfinite(z["rlc_k"]), "d1": lambda z: z["rlc_k"] == 1, "d2+": lambda z: z["rlc_k"] >= 2}
for seg, mf in masks.items():
    bm = np.array([mae(B[d]["rl_trees"], B[d]["rl_true"], mf(V[d])) for d in days])
    vm = np.array([mae(V[d]["rl_trees"], V[d]["rl_true"], mf(V[d])) for d in days])
    bb = np.mean([bias(B[d]["rl_trees"], B[d]["rl_true"], mf(V[d])) for d in days])
    vb = np.mean([bias(V[d]["rl_trees"], V[d]["rl_true"], mf(V[d])) for d in days])
    lines.append(f"| tree RL | {seg} | {bm.mean():.2f} | {vm.mean():.2f} | {ci(vm - bm)} | {bb:+.2f} | {vb:+.2f} |")
cut = lambda z: np.arange(len(z["rl_true"])) < len(z["rl_true"]) - 3   # noqa: E731
bm = np.array([mae(B[d]["rl_mixed"], B[d]["rl_true"], cut(B[d])) for d in days])
vm = np.array([mae(V[d]["rl_mixed"], V[d]["rl_true"], cut(V[d])) for d in days])
lines += [f"| Timestral-mixed RL | all but last 3 | {bm.mean():.2f} | {vm.mean():.2f} | {ci(vm - bm)} | | |", ""]

# ---- model sets, components on d2+ slots (and d1 for the held-out loads)
sets = [("app (cutoff-trained)", "rlc_base_"), ("matched (d2 weather, from 2025-10-01)", "rlc_matched_"),
        ("control (app weather, matched rows)", "rlc_control_")]
z0 = V[days[0]]
if any(p + "tso_wind" in z0.files for _, p in sets[1:]):
    lines += ["## Model sets on d2+ slots (stored, not all fed)", "",
              "| model set | RL MAE | RL bias | wind bias | solar bias | load bias | wind MAE | solar MAE | load MAE |", "|---|---|---|---|---|---|---|---|---|"]
    for name, p in sets:
        if p + "tso_wind" not in z0.files:
            continue
        r = {k: [] for k in ("rlm", "rlb", "wb", "sb", "lb", "wm", "sm", "lm")}
        for d in days:
            z = V[d]; m = z["rlc_k"] >= 2
            rl = z[p + "tso_load"] - z[p + "tso_wind"] - z[p + "tso_solar"]
            r["rlm"].append(mae(rl, z["rl_true"], m)); r["rlb"].append(bias(rl, z["rl_true"], m))
            for t, a in (("wind", "w"), ("solar", "s"), ("load", "l")):
                r[a + "b"].append(bias(z[p + "tso_" + t], z["true_tso_" + t], m)); r[a + "m"].append(mae(z[p + "tso_" + t], z["true_tso_" + t], m))
        mm = {k: np.mean(v) for k, v in r.items()}
        lines.append(f"| {name} | {mm['rlm']:.2f} | {mm['rlb']:+.2f} | {mm['wb']:+.2f} | {mm['sb']:+.2f} | {mm['lb']:+.2f} | "
                     f"{mm['wm']:.2f} | {mm['sm']:.2f} | {mm['lm']:.2f} |")
    lines.append("")

# ---- load-correction forms (load only, all slots of D)
if "rlc_hold_app_tso_load" in z0.files:
    forms = {"app load, uncorrected": lambda z: z["rlc_base_tso_load"],
             "held-out load, uncorrected": lambda z: z["rlc_hold_app_tso_load"],
             "PRIMARY held-out + own residual (window to gate − latency)": lambda z: z["rlc_hold_app_tso_load"] + z["rlc_corr_app"],
             "held-out + own residual, study window (to cutoff)": lambda z: z["rlc_hold_app_tso_load"] + z["rlc_corr_studywin_app"],
             "diagnostic: app load + held-out residual (gate window)": lambda z: z["rlc_base_tso_load"] + z["rlc_corr_app"]}
    lines += ["## Load forecast on D, app-weather forms (GW, all slots)", "", "| form | load MAE | load bias |", "|---|---|---|"]
    for name, f in forms.items():
        lines.append(f"| {name} | {np.mean([mae(f(V[d]), V[d]['true_tso_load']) for d in days]):.2f} | "
                     f"{np.mean([bias(f(V[d]), V[d]['true_tso_load'], np.isfinite(V[d]['rlc_k'])) for d in days]):+.2f} |")
    lines.append("")
    lines += ["Residuals per day (GW): " + "; ".join(f"{d}: {json.loads(str(V[d]['meta']))['rl_extra']['resid']}" for d in days), ""]
lines += ["Wiring checks per day: " + "; ".join(f"{d}: {json.loads(str(V[d]['meta'])).get('rl_check')}" for d in days),
          "", "Runtime per day (s): " + ", ".join(f"{d} {json.loads(str(V[d]['meta']))['secs']['total']}" for d in days)]
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
