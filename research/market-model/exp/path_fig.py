"""Figure for notes 21.64: does the path into a low-demand night matter? Fitting-period nights, scheduled (day-ahead) quantities."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
def clean(a):
    a.grid(True, color=GRID, linewidth=0.6); a.set_axisbelow(True); [a.spines[s].set_visible(False) for s in ("top", "right")]
def ttl(a, t): a.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
g = pd.read_csv(f"{SP}/out/path_nights.csv", index_col=0); g = g[g.tr]; short = (g.Dp1 - g.Dp) > 8; sust = (g.Dp1 - g.Dp) < 3; edges = np.arange(-5, 41, 5); cen = edges[:-1] + 2.5
fig, axs = plt.subplots(1, 3, figsize=(18.5, 5.0), gridspec_kw=dict(wspace=0.24))
for ax, col, yl in ((axs[0], "disp", "scheduled dispatchable output at night, GW"), (axs[1], "np_", "day-ahead net export position at night, GW")):
    ax.plot(g.Dp, g[col], "o", color=NEUT, markersize=2.5, alpha=0.5, markeredgewidth=0)
    for m, c, lab in ((short, ORANGE, "short trough: next night more than 8 GW higher"), (sust, BLUE, "sustained: next night less than 3 GW higher")):
        med = [g[col][m & (g.Dp >= a) & (g.Dp < a + 5)].median() if (m & (g.Dp >= a) & (g.Dp < a + 5)).sum() >= 8 else np.nan for a in edges[:-1]]; ax.plot(cen, med, "-o", color=c, linewidth=2, markersize=4, label=lab)
    ax.set_xlabel("demand axis at night (published residual load minus other plant), GW"); ax.set_ylabel(yl); clean(ax)
axs[0].legend(frameon=False, loc="upper left", fontsize=8); lo = g.Dp < 15; d1 = g.disp[lo & short].mean() - g.disp[lo & sust].mean(); d2 = g.np_[lo & short].mean() - g.np_[lo & sust].mean()
ttl(axs[0], f"At the same demand, a short trough leaves only {d1:.1f} GW more\nplant scheduled than a sustained one (demand under 15 GW)"); ttl(axs[1], f"Exports are {d2:.1f} GW higher in a short trough;\nthe path matters little on either side")
ax = axs[2]; ax.plot(g.Dp, g.sh, "o", color=NEUT, markersize=2.5, alpha=0.5, markeredgewidth=0); med = [g.sh[(g.Dp >= a) & (g.Dp < a + 5)].median() for a in edges[:-1]]; ax.plot(cen, med, "-o", color=INK, linewidth=2, markersize=4, label="median per 5 GW"); ax.axhline(0, color=INK, linewidth=0.7); ax.set_xlabel("demand axis at night, GW"); ax.set_ylabel("scheduled dispatchable minus the model's curves at the price, GW"); ax.legend(frameon=False, loc="upper center", fontsize=8); clean(ax)
ttl(ax, f"The static misfit is larger than the path effect: the curves are\n{med[1]:+.1f} GW off at 0-5 GW of demand and {med[-2]:+.1f} at 30-35")
fig.suptitle("Does the path into a low-demand night matter? 546 nights of the fitting period (Mar 2024 - Nov 2025), day-ahead scheduled quantities, 00:00-06:00", x=0.06, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.03); fig.savefig(f"{SP}/out/path_nights.png", dpi=100, bbox_inches="tight"); print("ok", round(d1, 2), round(d2, 2), np.round(med, 2))
