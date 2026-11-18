"""Figure: how much of a change in gas-fired cost each model passes on to the price at 30 GW, fit by fit; the gas price the fits had seen; and what a
prior on that response buys when a fit is frozen."""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
import oc, ocxp
from ocx_fits import fit_dates
from ocx2 import hx
SP = oc.SP; BLUE, ORANGE, AQUA, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
dates = fit_dates("2025-11-24", "2026-11-09"); T = pd.to_datetime(dates); pt = {}
for v in ("c", "p3"):
    pt[v] = [float(ocxp.passthru({k: np.asarray(x, float) for k, x in np.load(f"{SP}/out/ocxw_{v}/{t}.npz").items()}, np.array([30.0]), np)[0]) for t in dates]
gas = [float(np.nanmean(oc.GAS[hx(t) - 720:hx(t)])) for t in dates]
cand_t, cand_v = [], []
for period, run in (("winter", "hw2"), ("summer", "hs2"), ("autumn", "ha")):
    I = np.load(f"{SP}/out/inputs_{period}.npz")
    for c in I["cutoffs"]:
        z = np.load(f"{SP}/out/struct_{run}/{str(c)[:10]}.npz"); cand_t.append(pd.Timestamp(str(c)[:10])); cand_v.append(float(np.interp(30.0, z["grid"], z["h_curve"])) / 1.85)
fig = plt.figure(figsize=(13.2, 6.6)); gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], height_ratios=[1.6, 1.0], hspace=0.16, wspace=0.2)
ax = fig.add_subplot(gs[0, 0]); ax.plot(T, pt["c"], color=BLUE, linewidth=2, label="index model, priors as before"); ax.plot(T, pt["p3"], color=ORANGE, linewidth=2, label="index model, gas share pinned from 2021-2023"); ax.plot(cand_t, cand_v, color=AQUA, linewidth=2, label="candidate's curve: its gas term only (it also has a drifting price scale)")
for y, lab in ((0.81, "2021-2023 data: 0.81"), (1.03, "Mar-Nov 2026 data: 1.03")): ax.axhline(y, color=NEUT, linewidth=1, linestyle=(0, (4, 3))); ax.annotate(lab, (T[0], y), xytext=(2, 3), textcoords="offset points", fontsize=8, color=MUTED)
ax.set_ylabel("share of a change in gas-fired cost\nthat reaches the price at 30 GW"); ax.set_ylim(0.3, 1.3); ax.grid(True, axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower right", fontsize=8); plt.setp(ax.get_xticklabels(), visible=False)
ax.set_title("Share of a gas-cost change that reaches the price at 30 GW, fit by fit", loc="left", fontsize=10, fontweight="bold", color=INK)
a2 = fig.add_subplot(gs[1, 0], sharex=ax); a2.plot(T, gas, color=MUTED, linewidth=2); a2.set_ylabel("gas price at the fit date\nEUR/MWh, 30-day mean"); a2.grid(True, axis="y", color=GRID, linewidth=0.6); a2.set_axisbelow(True); a2.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); a2.set_ylim(20, 100)
ld = lambda f: np.load(f"{SP}/out/{f}.npz")["vint"].astype(float)
def night(f, period): a = np.load(f"{SP}/out/inputs_{period}.npz")["actual"].astype(float); return float(np.abs(ld(f) - a)[:, :32].mean())
a3 = fig.add_subplot(gs[:, 1]); groups = [("summer\nfit of 5 Jan", "summer", "0105"), ("summer\nfit of 30 Mar", "summer", "0330"), ("autumn\nfit of 5 Jan", "autumn", "0105"), ("autumn\nfit of 30 Mar", "autumn", "0330")]; w = 0.26
for i, (lab, period, fd) in enumerate(groups):
    vals = [(night(f"ocxs_frz{fd}_{period}_ts", period), BLUE), (night(f"ocxs_p3_frz{fd}_{period}_ts", period), ORANGE), (night(f"ocxs_c_{period}_ts", period), NEUT)]
    for k, (v, col) in enumerate(vals):
        a3.bar(i + (k - 1) * (w + 0.02), v, width=w, color=col, linewidth=0, label=("priors as before, frozen", "gas share pinned, frozen", "refitted every week")[k] if i == 0 else None); a3.annotate(f"{v:.2f}", (i + (k - 1) * (w + 0.02), v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5, color=MUTED)
a3.set_xticks(range(len(groups))); a3.set_xticklabels([g[0] for g in groups], fontsize=8); a3.set_ylabel("night MAE, ct/kWh"); a3.grid(True, axis="y", color=GRID, linewidth=0.6); a3.set_axisbelow(True); a3.legend(frameon=False, loc="upper left", fontsize=8); a3.set_ylim(0, 2.95)
a3.set_title("Night error when a fit cannot be refreshed", loc="left", fontsize=10, fontweight="bold", color=INK)
for a_ in (ax, a2, a3):
    for s_ in ("top", "right"): a_.spines[s_].set_visible(False)
out = f"{SP}/out/gas_fig.png"; fig.savefig(out, dpi=115, bbox_inches="tight"); print("figure:", out, "| candidate curve pass-through at 30 GW: first %.2f, min %.2f, max %.2f, last %.2f" % (cand_v[0], min(cand_v), max(cand_v), cand_v[-1]))
