"""Walk-forward refits of the market model and the ablation on the candidate's side (notes 21.21). Numbers in panels B and C are those printed by
mm_gw_score.py, l1_layer.py and dyn2.py."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"; LB, LO = "#a9c9ef", "#f7c3ad"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a, grid="y"):
    a.grid(True, axis=grid, color=GRID, linewidth=0.6); a.set_axisbelow(True)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
fig = plt.figure(figsize=(18.5, 4.9)); gs = fig.add_gridspec(1, 3, wspace=0.24, width_ratios=[1, 1.1, 1.25])
ax = fig.add_subplot(gs[0, 0]); fs = sorted(glob.glob(f"{SP}/out/market0e_r*_post.npz")); d = [pd.Timestamp(f.split("_r")[-1][:8]) for f in fs]; Z = [np.load(f) for f in fs]
for k, col, lab in (("x0", BLUE, "net imports at price = cost"), ("K_x", ORANGE, "step in net imports near zero price"), ("K_s", AQUA, "shelf at exactly zero")):
    q = np.array([np.percentile(z[k], [5, 50, 95]) for z in Z]); ax.fill_between(d, q[:, 0], q[:, 2], color=col, alpha=0.18, linewidth=0); ax.plot(d, q[:, 1], "-o", color=col, linewidth=1.8, markersize=3.5, label=lab)
ax.set_ylim(0, 6); ax.set_ylabel("GW (median and 90% band of each refit)"); ax.set_xlabel("data used up to"); ax.legend(frameon=False, loc="upper right", fontsize=8); clean(ax, "both"); ax.tick_params(axis="x", labelrotation=30)
x0 = [float(np.median(z["x0"])) for z in Z]; kx = [float(np.median(z["K_x"])) for z in Z]; ttl(ax, f"Refitted month by month, the import level walks down\n{x0[0] - x0[-1]:.1f} GW and the step near zero halves ({kx[0]:.1f} to {kx[-1]:.1f} GW)")
ax = fig.add_subplot(gs[0, 1]); per = ("winter", "summer", "autumn"); x = np.arange(3); w = 0.19; ser = (("curves only, frozen at Nov 2025", LB, (1.46, 1.66, 2.36)), ("curves only, refitted monthly", NEUT, (1.43, 1.53, 2.04)), ("with the layer, frozen", BLUE, (0.61, 0.81, 1.51)), ("with the layer, refitted", INK, (0.61, 0.81, 1.52)))
for j, (lb, col, v) in enumerate(ser):
    ax.bar(x + (j - 1.5) * (w + 0.015), v, w, color=col, linewidth=0, label=lb)
    for xi, a_ in zip(x, v): ax.text(xi + (j - 1.5) * (w + 0.015), a_ + 0.03, f"{a_:.2f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels(per); ax.set_ylim(0, 3.2); ax.set_ylabel("night slots of D+2, price error, ct/kWh (published load given)"); ax.legend(frameon=False, loc="upper left", fontsize=7.5); clean(ax)
ttl(ax, "Refitting helps the bare market model and does\nnothing once the layer is on")
ax = fig.add_subplot(gs[0, 2]); w = 0.19; ser = (("candidate's curve + layer", ORANGE, (0.86, 0.89, 1.59)), ("... without Europe-wide weather", LO, (0.82, 0.97, 1.67)), ("market model + layer", BLUE, (0.81, 1.06, 1.73)), ("plain curve + layer", NEUT, (0.84, 1.05, 1.70)))
for j, (lb, col, v) in enumerate(ser):
    ax.bar(x + (j - 1.5) * (w + 0.015), v, w, color=col, linewidth=0, label=lb)
    for xi, a_ in zip(x, v): ax.text(xi + (j - 1.5) * (w + 0.015), a_ + 0.02, f"{a_:.2f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels(per); ax.set_ylim(0, 2.3); ax.set_ylabel("night slots of D+2, price error, ct/kWh (Timestral's load forecast)"); ax.legend(frameon=False, loc="upper left", fontsize=7.5); clean(ax)
ttl(ax, "Europe-wide weather is about half of the candidate's edge\nin summer and autumn, and costs it in winter")
fig.suptitle("Walk-forward test of the market model, and what the candidate's curve still has that it lacks", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04)
fig.savefig(f"{SP}/out/market0e_walkforward.png", dpi=105, bbox_inches="tight"); print("figure:", f"{SP}/out/market0e_walkforward.png")
