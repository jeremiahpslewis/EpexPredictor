import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; T = pd.read_csv(f"{SP}/out/durable5.csv", index_col=0); I = pd.read_csv(f"{SP}/out/durable_imports_r2.csv", index_col=0)
BLUE, ORANGE, AQUA, GREY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#a8a79f", "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor=SURF)
def style(a):
    a.set_facecolor(SURF); a.yaxis.grid(True, color=GRID); a.set_axisbelow(True); a.tick_params(length=0)
    for s in ("top", "right", "left"): a.spines[s].set_visible(False)
a = ax[0]; style(a); x = np.arange(len(T)); a.axvspan(-0.4, 6.67, color=GRID, alpha=0.6, lw=0); a.text(-0.2, 37.5, "fit window of the curves", color=INK2, fontsize=9)
for c, col, lab, dy in ((T.columns[0], ORANGE, "nothing moves", 0), (T.columns[2], BLUE, "one level moves", 0)):
    a.plot(x, T[c], color=col, lw=2.2, marker="o", ms=4); a.text(x[-1] + 0.2, T[c].iloc[-1], lab, color=col, va="center", fontsize=9.5)
a.set_xticks(x[::2]); a.set_xticklabels([s[:4] + "\n" + s[4:] for s in T.index[::2]], fontsize=8.5); a.set_xlim(-0.4, len(T) + 1.9); a.set_ylim(0, 40); a.set_ylabel("price error, % of the mean price")
a.set_title("Frozen curves and shapes keep their accuracy for 2.7 years\nif one level may move; frozen entirely, they break in 2026", loc="left", fontsize=10.5, fontweight="bold")
a = ax[1]; style(a); x = np.arange(len(I)); a.axvspan(-0.4, 1.5, color=GRID, alpha=0.6, lw=0); a.text(-0.3, 66.5, "fitted here", color=INK2, fontsize=9)
S_ = (("current form, first-year fit", ORANGE, "current curve, frozen", 0), ("spread law, actual neighbour prices", GREY, "price-difference law,\nneighbours' prices known", 3), ("current form + its own level of last week", BLUE, "current curve + level", 2.5), ("the same + the law's own level of last week", AQUA, "price-difference law,\nneighbours modelled, + level", -4))
for c, col, lab, dy in S_:
    a.plot(x, I[c], color=col, lw=2.2, marker="o", ms=4, label=lab.replace("\n", " "))
a.set_xticks(x); a.set_xticklabels([s[:4] + "\n" + s[4:] for s in I.index], fontsize=8.5); a.set_xlim(-0.4, len(I) - 0.6); a.set_ylim(-22, 70); a.set_yticks([0, 20, 40, 60]); a.legend(frameon=False, fontsize=9, loc="lower left"); a.set_ylabel("hourly variance of net imports explained, %")
a.set_title("Net imports: the frozen curve fades; with a level that\nmoves it holds as well as a law that needs the neighbours", loc="left", fontsize=10.5, fontweight="bold")
fig.text(0.008, 0.012, "German day-ahead market, March 2024 to November 2026. Left: supply curves fitted once (shaded), hour-of-day shape of the shift from the first year; the level is the mean shift of the 7 days before. Right: all fitted on March 2024 to February 2025 only; level = mean error of the previous week.", fontsize=8, color=INK2, wrap=True)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{SP}/out/durable2.png", dpi=200, facecolor=SURF)
