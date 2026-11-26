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
  load-correction forms (study window, held-out residual on the app model);
- for `published` / `matchedpub`: the slots fed with the published load, and (base = a matched run) the deviation of
  the variant's tree RL from the base run's REF form (its wind/solar + published load), which must be float noise.
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
# ---- load forms at RL level (EXPECTATIONS addendum 3 amendment, F0-F4 + REF; needs a matched/both run with forms)
if "rlc_form_hold_matched_gate" in z0.files and "rlc_matched_tso_load" in z0.files:
    def form(z, kind):
        d2 = z["rlc_k"] >= 2
        pick = lambda a, m: np.where(d2, z[m], z[a])  # noqa: E731 - app set on d1 slots, matched set on d2+
        if kind == "F0":
            return pick("rlc_base_tso_load", "rlc_matched_tso_load")
        if kind == "REF":
            return z["pub_tso_load_gate"]
        tag = {"F1": "hold_{}_gate", "F2": "hold_{}_study", "F3": "diag_{}_gate", "F4": "diag_{}_study"}[kind]
        return pick("rlc_form_" + tag.format("app"), "rlc_form_" + tag.format("matched"))
    names = {"F0": "fed in matched: cutoff-trained load", "F1": "PRIMARY held-out + own resid, gate window",
             "F2": "held-out + own resid, study window", "F3": "cutoff-trained + held-out resid, gate window",
             "F4": "cutoff-trained + held-out resid, study window", "REF": "published TSO load at the gate (load error 0)"}
    ws = lambda z: z["rlc_sel_tso_wind"] + z["rlc_sel_tso_solar"]  # noqa: E731 - matched run's wind + solar
    lines += ["## Load forms at RL level (wind/solar as in matched; load per form)", "",
              "Diff = form − F0, per-day RL MAE, paired 7-day block bootstrap. Decision rule L5: a price pass only for the "
              "F1–F4 form with the largest all-slot gain ≥ 0.10 GW whose interval excludes 0.", "",
              "| form | RL MAE all | diff all | RL MAE d1 | diff d1 | RL MAE d2+ | diff d2+ | load MAE | load bias |",
              "|---|---|---|---|---|---|---|---|---|"]
    best = None
    for d in days:   # F0 is what a matched run feeds: its RL must equal the stored tree RL
        if json.loads(str(V[d]["meta"])).get("rl_variant") == "matched":
            assert np.allclose(form(V[d], "F0") - ws(V[d]), V[d]["rl_trees"], atol=1e-9, equal_nan=True), d
    f0 = {seg: np.array([mae(form(V[d], "F0") - ws(V[d]), V[d]["rl_true"], mf(V[d])) for d in days]) for seg, mf in masks.items()}
    for kind in ("F0", "F1", "F2", "F3", "F4", "REF"):
        r = {seg: np.array([mae(form(V[d], kind) - ws(V[d]), V[d]["rl_true"], mf(V[d])) for d in days]) for seg, mf in masks.items()}
        lm = np.mean([mae(form(V[d], kind), V[d]["true_tso_load"]) for d in days])
        lb = np.mean([bias(form(V[d], kind), V[d]["true_tso_load"], np.isfinite(V[d]["rlc_k"])) for d in days])
        cell = lambda seg: "—" if kind == "F0" else ci(r[seg] - f0[seg])  # noqa: E731
        lines.append(f"| {kind}: {names[kind]} | {r['all'].mean():.2f} | {cell('all')} | {r['d1'].mean():.2f} | {cell('d1')} | "
                     f"{r['d2+'].mean():.2f} | {cell('d2+')} | {lm:.2f} | {lb:+.2f} |")
        if kind in ("F1", "F2", "F3", "F4") and n >= 7:
            dd = r["all"] - f0["all"]; lo, hi = boot(dd)
            if dd.mean() <= -0.10 and hi < 0 and (best is None or dd.mean() < best[1]):
                best = (kind, dd.mean())
    lines += ["", "L5 decision: " + ("n/a (fewer than 7 days)" if n < 7 else
                                     (f"request a price pass for {best[0]} ({best[1]:+.3f} GW)" if best else "no form qualifies")), ""]

# ---- published-load variants (EXPECTATIONS addendum 5): fed load on D and the REF identity
if json.loads(str(z0["meta"])).get("rl_variant") in ("published", "matchedpub"):
    lines += ["## Published load on D", ""]
    for d in days:
        z, b = V[d], B[d]
        line = (f"- {d}: published-load slots {int(np.nansum(z['rlc_use_pub']))}/{len(z['idx'])}, "
                f"fed load − published max |diff| {np.max(np.abs(z['rlc_sel_tso_load'] - z['pub_tso_load_gate'])):.1e}")
        if "rlc_sel_tso_wind" in b.files and "pub_tso_load_gate" in b.files:
            # the base run's own wind/solar + the published load (its REF form) must be what the variant fed
            ref = b["pub_tso_load_gate"] - b["rlc_sel_tso_wind"] - b["rlc_sel_tso_solar"]
            dev = float(np.max(np.abs(z["rl_trees"] - ref)))
            line += f", rl_trees − base run's REF form max |diff| {dev:.1e}"
        elif "rl_base_model" in z.files:
            # base run without rlc arrays (out_strict): the variant's app-booster RL must equal the base run's tree RL
            line += f", rl_base_model − base rl_trees max |diff| {np.max(np.abs(z['rl_base_model'] - b['rl_trees'])):.1e}"
        line += f", rl_timestral − base max |diff| {np.max(np.abs(z['rl_timestral'] - b['rl_timestral'])):.1e}"
        lines.append(line)
    lines.append("")

lines += ["Wiring checks per day: " + "; ".join(f"{d}: {json.loads(str(V[d]['meta'])).get('rl_check')}" for d in days),
          "", "Runtime per day (s): " + ", ".join(f"{d} {json.loads(str(V[d]['meta']))['secs']['total']}" for d in days)]
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
